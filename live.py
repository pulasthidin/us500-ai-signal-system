"""
US500 Intraday Trading Signal App — main entry point.

Startup sequence:
  1. Self-test all 10 dependencies (critical → halt, non-critical → warn)
  2. Fix partial DB saves
  3. Catch-up missed outcomes
  4. Check missed training / morning brief
  5. Send full startup report to Telegram
  6. Arm schedulers and enter main loop
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Tuple

import schedule
from dotenv import load_dotenv

# ─── force UTF-8 console on Windows (cp1252 can't handle emoji) ──
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("US500 Signal App")
    except Exception:
        pass

# ─── environment & directories ───────────────────────────────
load_dotenv()

for d in ("data/raw", "data/features", "models", "logs", "database"):
    os.makedirs(d, exist_ok=True)

# ─── root logger with rotating file ─────────────────────────
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
rfh = RotatingFileHandler("logs/app.log", maxBytes=5_000_000, backupCount=5)
rfh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s"))
root_logger.addHandler(rfh)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
root_logger.addHandler(console)

logger = logging.getLogger("live")

# ─── imports from src ────────────────────────────────────────
from config import (
    HEALTH_CHECK_INTERVAL_MINUTES,
    MODEL_RETRAIN_DAY,
    MODEL_RETRAIN_TIME_SL,
    MORNING_BRIEF_TIME_UTC,
    OUTCOME_CHECK_INTERVAL_MINUTES,
    SIGNAL_CHECK_INTERVAL_SECONDS,
    WEEKLY_REPORT_TIME_UTC,
    SL_UTC_OFFSET,
)
from src.alert_bot import AlertBot
from src.calendar_checker import CalendarChecker
from src.checklist_engine import ChecklistEngine
from src.ctrader_connection import CTraderConnection
from src.entry_checker import EntryChecker
from src.evolution_manager import EvolutionManager
from src.health_monitor import HealthMonitor
from src.macro_checker import MacroChecker
from src.model_predictor import ModelPredictor
from src.model_trainer import ModelTrainer
from src.orderflow_analyzer import OrderFlowAnalyzer
from src.outcome_tracker import OutcomeTracker
from src.pattern_scanner import PatternScanner
from src.signal_logger import SignalLogger
from src.structure_analyzer import StructureAnalyzer
from src.zone_calculator import ZoneCalculator

# ─── globals ─────────────────────────────────────────────────
# All initialised to None so guards like `if x is None` are safe before startup completes.
alert_bot = None
ctrader = None
us500_id: int = 0
ustec_id: int = 0
signal_logger = None
macro_checker = None
calendar_checker = None
structure_analyzer = None
zone_calculator = None
orderflow_analyzer = None
entry_checker = None
checklist_engine = None
outcome_tracker = None
health_monitor = None
model_predictor = None
model_trainer = None
evolution_manager = None
pattern_scanner = None

HEARTBEAT_PATH = os.path.join("data", "heartbeat.json")
TEST_TIMEOUT = 15


# ══════════════════════════════════════════════════════════════
# SELF-TEST ENGINE
# ══════════════════════════════════════════════════════════════

class StartupTest:
    """Result container for one self-test."""
    def __init__(self, name: str, critical: bool):
        self.name = name
        self.critical = critical
        self.passed = False
        self.detail = ""
        self.error = ""

    @property
    def icon(self) -> str:
        if self.passed:
            return "\u2705"
        return "\U0001f534" if self.critical else "\u26a0\ufe0f"

    @property
    def status(self) -> str:
        if self.passed:
            return f"PASS{(' (' + self.detail + ')') if self.detail else ''}"
        return f"FAIL: {self.error}" if self.critical else f"WARN: {self.error}"


def _run_test_database() -> StartupTest:
    t = StartupTest("Database", critical=True)
    try:
        global signal_logger
        signal_logger = SignalLogger()
        signal_logger.init_db()
        count = signal_logger.get_signal_count()
        t.passed = True
        t.detail = f"{count} labelled signals"
    except Exception as exc:
        t.error = str(exc)
    return t


def _is_weekend() -> bool:
    """Check if today is Saturday (5) or Sunday (6) in UTC."""
    return datetime.now(timezone.utc).weekday() >= 5


def _run_test_ctrader() -> StartupTest:
    t = StartupTest("cTrader", critical=True)
    try:
        global ctrader, us500_id, ustec_id
        ctrader = CTraderConnection(alert_bot=alert_bot)
        ctrader.connect()
        ctrader.authenticate_app()
        ctrader.authenticate_account()
        ctrader.refresh_token()

        us500_id = ctrader.get_symbol_id("US500") or 0
        ustec_id = ctrader.get_symbol_id("USTEC") or 0
        if us500_id == 0:
            raise ConnectionError("US500 symbol not found")

        bars = ctrader.fetch_bars(us500_id, "M15", 5)
        if bars is None or bars.empty:
            if _is_weekend():
                t.passed = True
                t.detail = f"US500:{us500_id} (weekend — no bars expected)"
                return t
            raise ConnectionError("No M15 bars returned")

        price = float(bars["close"].iloc[-1])
        if price < 1000:
            raise ValueError(f"Price {price} too low — decoding may be wrong")

        t.passed = True
        t.detail = f"US500:{us500_id} price:{price:.2f}"
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_trade_bot() -> StartupTest:
    t = StartupTest("Trade Bot", critical=True)
    try:
        alert_bot.send_trade_message("\U0001f527 System starting up...")
        t.passed = True
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_system_bot() -> StartupTest:
    t = StartupTest("System Bot", critical=True)
    try:
        alert_bot.send_system_alert("INFO", "selftest", "\U0001f527 System health bot active")
        t.passed = True
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_yfinance() -> StartupTest:
    t = StartupTest("Macro Data", critical=True)
    try:
        import yfinance as yf
        ticker = yf.Ticker("^VIX")
        info = ticker.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
        if price is None or price <= 0:
            raise ValueError("VIX price not available")
        t.passed = True
        t.detail = f"VIX:{price:.2f}"
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_calendar() -> StartupTest:
    t = StartupTest("Calendar", critical=False)
    try:
        global calendar_checker
        calendar_checker = CalendarChecker(alert_bot=alert_bot)
        calendar_checker.fetch_weekly_calendar()
        t.passed = True
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_groq() -> StartupTest:
    t = StartupTest("Groq", critical=False)
    try:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            t.error = "no API key"
            return t
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Reply OK"}],
            max_tokens=3,
        )
        if resp and resp.choices:
            t.passed = True
        else:
            t.error = "empty response"
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_ml_models() -> StartupTest:
    t = StartupTest("ML Models", critical=False)
    try:
        global model_predictor
        model_predictor = ModelPredictor(signal_logger=signal_logger)
        model_predictor.load_all_models()
        loaded = [k for k in ("model1", "model2", "model3") if k in model_predictor.models]
        if loaded:
            t.passed = True
            t.detail = " ".join(k.upper().replace("MODEL", "M") for k in loaded)
        else:
            t.error = "no models found (rule-based only)"
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_smc() -> StartupTest:
    t = StartupTest("SMC Library", critical=True)
    try:
        from smartmoneyconcepts import smc
        bars = ctrader.fetch_bars(us500_id, "M5", 50)
        if bars is None or bars.empty:
            if _is_weekend():
                t.passed = True
                t.detail = "weekend — skipped bar test"
                return t
            raise ConnectionError("No M5 bars for SMC test")
        smc.fvg(bars)
        swing = smc.swing_highs_lows(bars, swing_length=5)
        smc.bos_choch(bars, swing, close_break=True)
        t.passed = True
    except Exception as exc:
        t.error = str(exc)
    return t


def _run_test_pandas_ta() -> StartupTest:
    t = StartupTest("pandas-ta", critical=True)
    try:
        import pandas_ta as ta
        bars = ctrader.fetch_bars(us500_id, "H4", 100)
        if bars is None or bars.empty:
            if _is_weekend():
                t.passed = True
                t.detail = "weekend — skipped bar test"
                return t
            raise ConnectionError("No H4 bars for EMA test")
        ema50 = ta.ema(bars["close"], length=50)
        if ema50 is None or ema50.dropna().empty:
            raise ValueError(f"EMA50 returned None ({len(bars)} bars, need 50+)")
        val50 = float(ema50.dropna().iloc[-1])
        t.detail = f"EMA50:{val50:.0f} ({len(bars)} bars)"
        t.passed = True
    except Exception as exc:
        t.error = str(exc)
    return t


def run_all_self_tests() -> List[StartupTest]:
    """
    Run all 10 startup tests.  Tests 1-5 must run sequentially (DB → cTrader → bots → yfinance).
    Tests 6-8 run in parallel (non-critical, independent).
    Tests 9-10 require cTrader and run after it.
    """
    results: List[StartupTest] = []

    sequential_tests = [
        _run_test_database,
        _run_test_ctrader,
        _run_test_trade_bot,
        _run_test_system_bot,
        _run_test_yfinance,
    ]

    for fn in sequential_tests:
        t = fn()
        results.append(t)
        logger.info("Self-test [%s]: %s", t.name, t.status)
        if t.critical and not t.passed:
            return results

    parallel_tests = [_run_test_calendar, _run_test_groq, _run_test_ml_models]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): fn.__name__ for fn in parallel_tests}
        try:
            for future in as_completed(futures, timeout=TEST_TIMEOUT):
                try:
                    t = future.result(timeout=TEST_TIMEOUT)
                except Exception as exc:
                    name = futures[future].replace("_run_test_", "")
                    t = StartupTest(name, critical=False)
                    t.error = f"timeout/error: {exc}"
                results.append(t)
                logger.info("Self-test [%s]: %s", t.name, t.status)
        except TimeoutError:
            for future, name in futures.items():
                if not future.done():
                    clean_name = name.replace("_run_test_", "")
                    t = StartupTest(clean_name, critical=False)
                    t.error = "timed out"
                    results.append(t)
                    logger.warning("Self-test [%s]: timed out", clean_name)

    post_ctrader = [_run_test_smc, _run_test_pandas_ta]
    for fn in post_ctrader:
        t = fn()
        results.append(t)
        logger.info("Self-test [%s]: %s", t.name, t.status)
        if t.critical and not t.passed:
            return results

    return results


def format_test_report(
    results: List[StartupTest],
    offline: str,
    catchup: Dict[str, int],
    partial_fixed: int,
    missed_training: str,
) -> str:
    """Build the startup Telegram report."""
    lines = ["\u2501" * 20, "System Tests:"]
    for t in results:
        lines.append(f"{t.name:14s}{t.icon} {t.status}")

    lines.append("\u2501" * 20)
    lines.append(f"Offline: {offline}")

    cu_total = catchup.get("checked", 0)
    lines.append(f"Catch-up: {cu_total} outcomes filled")
    if cu_total > 0:
        lines.append(f"  \u2705 WIN:{catchup.get('wins',0)} \u274c LOSS:{catchup.get('losses',0)} \u23f1 TIMEOUT:{catchup.get('timeouts',0)}")

    lines.append(f"Partial saves fixed: {partial_fixed}")
    lines.append(f"Training: {missed_training}")
    lines.append("\u2501" * 20)

    mp_models = model_predictor.models if model_predictor is not None else {}
    m1 = "\u2705" if "model1" in mp_models else "\u274c"
    m2 = "\u2705" if "model2" in mp_models else "\u274c"
    m3 = "\u2705" if "model3" in mp_models else "\u274c"
    evo_stage = evolution_manager.get_stage_info().get("stage", 0) if evolution_manager is not None else 0
    evo_count = signal_logger.get_signal_count() if signal_logger is not None else 0
    next_t = {0: 200, 1: 500, 2: 1000}.get(evo_stage, None)

    lines.append(f"Models: M1{m1} M2{m2} M3{m3}")
    lines.append(f"Stage: {evo_stage} | Signals: {evo_count}")
    if next_t:
        lines.append(f"Next stage: {next_t} signals")
    lines.append("\u2501" * 20)
    lines.append("Sessions today (SL):")
    lines.append("London:  12:30 \u2013 21:30")
    lines.append("NY Open: 18:30 \u2013 21:30")
    lines.append("NY:      21:30 \u2013 02:30")

    critical_fail = any(t.critical and not t.passed for t in results)
    if critical_fail:
        failed = [t for t in results if t.critical and not t.passed]
        lines.insert(0, "\U0001f534 STARTUP FAILED")
        lines.append("\u2501" * 20)
        for t in failed:
            lines.append(f"Failed: {t.name}")
            lines.append(f"Error: {t.error}")
        lines.append("Fix required before trading.")
        lines.append("App stopped.")
    else:
        lines.insert(0, "\u2705 STARTUP COMPLETE")
        lines.append("\u2501" * 20)
        lines.append("\U0001f7e2 ALL SYSTEMS GO")
        lines.append("Monitoring active...")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# HEARTBEAT / SHUTDOWN / MISSED BRIEF
# ══════════════════════════════════════════════════════════════

def _save_heartbeat() -> None:
    try:
        tmp = HEARTBEAT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"last_seen": datetime.now(timezone.utc).isoformat()}, f)
        os.replace(tmp, HEARTBEAT_PATH)
    except Exception:
        pass


def _read_offline_duration(startup_time: datetime) -> str:
    try:
        if not os.path.exists(HEARTBEAT_PATH):
            return "unknown (first run)"
        with open(HEARTBEAT_PATH) as f:
            data = json.load(f)
        last_seen = datetime.fromisoformat(data["last_seen"])
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        delta = startup_time - last_seen
        hours = delta.total_seconds() / 3600
        if hours < 0:
            return "0m"
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m"
        if hours < 48:
            return f"{hours:.1f}h"
        return f"{delta.days}d {int(hours % 24)}h"
    except Exception:
        return "unknown"


def _shutdown_hook() -> None:
    try:
        _save_heartbeat()
        logger.info("Clean shutdown — heartbeat saved")
    except Exception:
        pass


def _check_missed_morning_brief() -> None:
    try:
        sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
        sl_now = datetime.now(sl_tz)
        sl_minutes = sl_now.hour * 60 + sl_now.minute
        if not (330 <= sl_minutes <= 600):
            return
        brief_flag = os.path.join("data", "brief_sent.json")
        today_str = sl_now.strftime("%Y-%m-%d")
        try:
            if os.path.exists(brief_flag):
                with open(brief_flag) as f:
                    flag = json.load(f)
                if flag.get("date") == today_str:
                    return
        except Exception:
            pass
        logger.info("Missed morning brief detected — sending now")
        send_morning_brief()
    except Exception as exc:
        logger.warning("Missed morning brief check failed: %s", exc)


# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════

def startup() -> None:
    global alert_bot, ctrader, us500_id, ustec_id
    global signal_logger, macro_checker, calendar_checker
    global structure_analyzer, zone_calculator, orderflow_analyzer
    global entry_checker, checklist_engine, outcome_tracker
    global health_monitor, model_predictor, model_trainer
    global evolution_manager, pattern_scanner

    startup_time = datetime.now(timezone.utc)
    logger.info("=" * 50)
    logger.info("US500 Signal App starting ...")
    logger.info("=" * 50)

    # ── Step 1: Telegram early (needed for test reports) ─────
    alert_bot = AlertBot()

    # ── Step 2: Run all self-tests ───────────────────────────
    logger.info("Running startup self-tests...")
    test_results = run_all_self_tests()

    critical_fail = any(t.critical and not t.passed for t in test_results)
    if critical_fail:
        failed = [t for t in test_results if t.critical and not t.passed]
        fail_msg = (
            "\U0001f534 STARTUP FAILED\n"
            + "\n".join(f"Failed: {t.name} — {t.error}" for t in failed)
            + "\nFix required. App stopped."
        )
        logger.critical(fail_msg)
        alert_bot.send_system_alert("CRITICAL", "startup", fail_msg)
        print(f"\n{'='*50}\nCRITICAL: {fail_msg}\n{'='*50}")
        raise SystemExit(1)

    # ── Step 3: Fix partial saves ────────────────────────────
    partial_fixed = signal_logger.fix_partial_saves()
    offline_duration = _read_offline_duration(startup_time)

    # ── Step 4: Catch-up outcomes ────────────────────────────
    outcome_tracker = OutcomeTracker(ctrader, us500_id, signal_logger, alert_bot=alert_bot)
    catchup = outcome_tracker.run_startup_catchup()
    logger.info("Catch-up complete: %s", catchup)

    # ── Step 5: ML trainer + missed training ─────────────────
    model_trainer = ModelTrainer(signal_logger=signal_logger, alert_bot=alert_bot)
    if model_predictor is None:
        model_predictor = ModelPredictor(signal_logger=signal_logger)
    model_predictor._trainer = model_trainer
    missed_training = "none"
    try:
        evo_temp = EvolutionManager(signal_logger, model_trainer, alert_bot=alert_bot)
        evo_info = evo_temp.check_evolution_stage()
        if evo_info.get("stage", 0) >= 1:
            model_predictor.load_all_models()
            missed_training = "checked"
    except Exception as exc:
        logger.warning("Missed training check failed: %s", exc)
        missed_training = "error"

    # ── Step 6: Init remaining analyzers ─────────────────────
    macro_checker = MacroChecker(alert_bot=alert_bot)
    if calendar_checker is None:
        calendar_checker = CalendarChecker(alert_bot=alert_bot)

    structure_analyzer = StructureAnalyzer(ctrader, us500_id, alert_bot=alert_bot)
    zone_calculator = ZoneCalculator(ctrader, us500_id, alert_bot=alert_bot)
    orderflow_analyzer = OrderFlowAnalyzer(ctrader, us500_id, alert_bot=alert_bot)
    entry_checker = EntryChecker(ctrader, us500_id, ustec_id, alert_bot=alert_bot)

    checklist_engine = ChecklistEngine(
        macro_checker=macro_checker,
        structure_analyzer=structure_analyzer,
        zone_calculator=zone_calculator,
        orderflow_analyzer=orderflow_analyzer,
        entry_checker=entry_checker,
        signal_logger=signal_logger,
        alert_bot=alert_bot,
    )
    health_monitor = HealthMonitor(ctrader, signal_logger, alert_bot=alert_bot)
    evolution_manager = EvolutionManager(signal_logger, model_trainer, alert_bot=alert_bot)
    pattern_scanner = PatternScanner(
        macro_checker=macro_checker,
        zone_calculator=zone_calculator,
        orderflow_analyzer=orderflow_analyzer,
        checklist_engine=checklist_engine,
        ctrader_connection=ctrader,
        us500_id=us500_id,
        signal_logger=signal_logger,
        model_predictor=model_predictor,
        alert_bot=alert_bot,
    )

    # ── Step 7: Send startup report ──────────────────────────
    report = format_test_report(test_results, offline_duration, catchup, partial_fixed, missed_training)
    logger.info(report)
    alert_bot.send_system_alert("INFO", "startup", report)

    # ── Step 8: Trader-facing "system alive" message ─────────
    sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
    sl_time = datetime.now(sl_tz).strftime("%H:%M SL")
    session = checklist_engine.get_current_session()
    alert_bot.send_trade_message(
        f"\U0001f7e2 Signal monitoring active\n"
        f"{session} session\n"
        f"{sl_time}\n"
        f"Watching US500..."
    )

    # ── Step 9: Hooks ────────────────────────────────────────
    atexit.register(_shutdown_hook)
    _save_heartbeat()
    _check_missed_morning_brief()


# ══════════════════════════════════════════════════════════════
# SCHEDULED JOBS
# ══════════════════════════════════════════════════════════════

def main_signal_check() -> None:
    try:
        if not checklist_engine.is_trading_session():
            return

        macro_data = macro_checker.get_layer1_result()
        news_data = calendar_checker.get_news_status()
        current_price = ctrader.get_current_price(us500_id)

        if current_price is None:
            health_monitor.record_error("price_fetch_failed")
            return

        checklist_result = checklist_engine.run_full_checklist(current_price, macro_data, news_data)

        if checklist_engine.should_send_alert(checklist_result):
            direction = checklist_result.get("direction")
            decision = checklist_result.get("decision")

            # Inject context features before ML so the model sees the same
            # features it was trained on (signals_last_60min, losses_last_60min, etc.)
            is_real_signal = decision in ("FULL_SEND", "HALF_SIZE")
            if is_real_signal and direction:
                context = signal_logger.get_recent_context(direction)
                checklist_result.update(context)

            ml_result = None
            if is_real_signal:
                ml_result = model_predictor.get_ml_enhancement(checklist_result, macro_data)
                checklist_result["ml"] = ml_result

            pa_match = None
            if is_real_signal and direction:
                pa_match = signal_logger.get_recent_unmatched_pattern_alert(direction, minutes=30)
                if pa_match:
                    try:
                        pa_dt = datetime.fromisoformat(pa_match["timestamp"])
                        if pa_dt.tzinfo is None:
                            pa_dt = pa_dt.replace(tzinfo=timezone.utc)
                        checklist_result["pattern_called_minutes_ago"] = (datetime.now(timezone.utc) - pa_dt).total_seconds() / 60.0
                    except Exception:
                        pass

            if is_real_signal:
                signal_id = signal_logger.log_signal(checklist_result)
                if signal_id is not None:
                    checklist_result["signal_id"] = signal_id

            alert_bot.send_trade_alert(checklist_result, ml_result)

            if is_real_signal and pa_match and checklist_result.get("signal_id") is not None:
                signal_logger.update_pattern_alert_match(pa_match["id"], checklist_result["signal_id"])

    except Exception as exc:
        logger.error("Signal check error: %s", exc, exc_info=True)
        alert_bot.send_system_alert("WARNING", "signal_loop", str(exc))


def send_morning_brief() -> None:
    try:
        macro_data = macro_checker.get_layer1_result()
        news_data = calendar_checker.get_news_status()
        current_price = ctrader.get_current_price(us500_id)
        zone_data = zone_calculator.get_all_levels(current_price) if current_price else []
        ml_stats = {"signal_count": signal_logger.get_signal_count(), "win_rate": signal_logger.get_win_rate()}
        alert_bot.send_morning_brief(macro_data, news_data, zone_data, ml_stats)
        logger.info("Morning brief sent")
        try:
            sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
            with open(os.path.join("data", "brief_sent.json"), "w") as f:
                json.dump({"date": datetime.now(sl_tz).strftime("%Y-%m-%d")}, f)
        except Exception:
            pass
    except Exception as exc:
        logger.error("Morning brief failed: %s", exc, exc_info=True)
        alert_bot.send_system_alert("WARNING", "morning_brief", str(exc))


def weekly_report_and_retrain() -> None:
    evo_result = {}
    try:
        model_trainer.check_and_retrain()
        model_predictor.load_all_models()
        evo_result = evolution_manager.run_evolution_check()
    except Exception as exc:
        logger.error("Weekly retrain failed: %s", exc, exc_info=True)
        alert_bot.send_system_alert("WARNING", "weekly_retrain", str(exc))

    try:
        now = datetime.now(timezone.utc)
        this_week_iso  = (now - timedelta(days=7)).isoformat()
        last_week_iso  = (now - timedelta(days=14)).isoformat()
        stats = {
            "win_rate": signal_logger.get_win_rate(),
            "by_session": signal_logger.get_stats_by_session(),
            "by_grade": signal_logger.get_stats_by_grade(),
            "by_direction": signal_logger.get_stats_by_direction(),
            "tp_source_breakdown": signal_logger.get_tp_source_breakdown(),
            "weekly_summary": signal_logger.get_weekly_summary(this_week_iso),
            "prev_week_win_rate": signal_logger.get_win_rate_since(last_week_iso),
            "this_week_win_rate": signal_logger.get_win_rate_since(this_week_iso),
            "system_health": health_monitor.get_system_status(),
            "shap_report": model_trainer.get_shap_report(),
            "evolution": evo_result,
            "pattern_accuracy": signal_logger.get_pattern_alert_accuracy(),
        }
        alert_bot.send_weekly_report(stats)
        logger.info("Weekly report sent")
    except Exception as exc:
        logger.error("Weekly report failed: %s", exc, exc_info=True)
        alert_bot.send_system_alert("WARNING", "weekly_report", str(exc))


def daily_retrain_midnight() -> None:
    try:
        if datetime.now(timezone.utc).strftime("%A").lower() == MODEL_RETRAIN_DAY.lower():
            logger.info("Daily retrain skipped — weekly retrain already covers %s", MODEL_RETRAIN_DAY)
            return
        stage_info = evolution_manager.get_stage_info()
        if not stage_info.get("daily_training_active"):
            return
        logger.info("Daily retrain triggered (Stage 3)")
        model_trainer.train_meta_label_model()
        model_predictor.load_all_models()
    except Exception as exc:
        logger.error("Daily retrain failed: %s", exc, exc_info=True)
        alert_bot.send_system_alert("WARNING", "daily_retrain", str(exc))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def _utc_to_local_schedule_time(utc_hhmm: str) -> str:
    """
    Convert a UTC HH:MM string to local system time HH:MM for the schedule library.
    The schedule library uses the OS local clock, not UTC.  All config times are
    expressed in UTC, so we convert them here once at startup.
    """
    h, m = map(int, utc_hhmm.split(":"))
    local_offset_sec = -(time.altzone if time.localtime().tm_isdst > 0 else time.timezone)
    local_minutes = (h * 60 + m + local_offset_sec // 60) % 1440
    return f"{local_minutes // 60:02d}:{local_minutes % 60:02d}"


def _sl_time_to_utc(sl_hhmm: str) -> str:
    """Convert an SL-timezone HH:MM to a UTC HH:MM string."""
    h, m = map(int, sl_hhmm.split(":"))
    utc_minutes = int((h * 60 + m - SL_UTC_OFFSET * 60) % 1440)
    return f"{utc_minutes // 60:02d}:{utc_minutes % 60:02d}"

_DAILY_RETRAIN_UTC = _sl_time_to_utc(MODEL_RETRAIN_TIME_SL)


def main() -> None:
    _VALID_SCHEDULE_DAYS = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    }
    if MODEL_RETRAIN_DAY.lower() not in _VALID_SCHEDULE_DAYS:
        raise ValueError(
            f"MODEL_RETRAIN_DAY='{MODEL_RETRAIN_DAY}' is not a valid day name. "
            f"Must be one of: {sorted(_VALID_SCHEDULE_DAYS)}"
        )

    startup()

    # Convert all UTC schedule times to local system time once at startup.
    # NOTE: on DST-aware systems these drift by 1h at DST boundaries; restart to fix.
    # Sri Lanka has no DST so this is safe for the primary deployment target.
    morning_brief_local   = _utc_to_local_schedule_time(MORNING_BRIEF_TIME_UTC)
    weekly_report_local   = _utc_to_local_schedule_time(WEEKLY_REPORT_TIME_UTC)
    daily_retrain_local   = _utc_to_local_schedule_time(_DAILY_RETRAIN_UTC)
    logger.info(
        "Scheduled times (local): morning_brief=%s  weekly_report=%s  daily_retrain=%s",
        morning_brief_local, weekly_report_local, daily_retrain_local,
    )

    schedule.every(SIGNAL_CHECK_INTERVAL_SECONDS).seconds.do(main_signal_check)
    schedule.every(HEALTH_CHECK_INTERVAL_MINUTES).minutes.do(health_monitor.monitor_loop)
    schedule.every(OUTCOME_CHECK_INTERVAL_MINUTES).minutes.do(outcome_tracker.track_pending_outcomes)
    schedule.every().day.at(morning_brief_local).do(send_morning_brief)
    getattr(schedule.every(), MODEL_RETRAIN_DAY.lower()).at(weekly_report_local).do(weekly_report_and_retrain)
    schedule.every(5).minutes.do(pattern_scanner.run_pattern_scan)
    schedule.every(5).minutes.do(_save_heartbeat)
    schedule.every().day.at(daily_retrain_local).do(daily_retrain_midnight)
    schedule.every().day.at(morning_brief_local).do(lambda: signal_logger.cleanup_old_pattern_alerts(days=7))

    logger.info("Scheduler armed — entering main loop")
    print("App running. Press Ctrl+C to stop.")

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            alert_bot.send_system_alert("INFO", "app", "App stopped by user")
            break
        except Exception as exc:
            logger.critical("Main loop crash: %s", exc, exc_info=True)
            try:
                alert_bot.send_system_alert("CRITICAL", "main_loop", str(exc))
            except Exception:
                logger.error("Failed to send crash alert to Telegram")
            time.sleep(5)
            continue


if __name__ == "__main__":
    main()
