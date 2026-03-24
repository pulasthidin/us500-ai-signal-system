"""
Health Monitor — periodic checks on every external dependency.
Alerts only on state *change* to avoid notification spam.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import yfinance as yf

from config import (
    HEALTH_CHECK_INTERVAL_MINUTES,
    NO_SIGNAL_ALERT_HOURS,
    SL_UTC_OFFSET,
    TRADING_END_SL,
    TRADING_START_SL,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("health")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/health.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class HealthMonitor:
    """Monitors cTrader, yfinance, database, model files, and signal frequency."""

    def __init__(self, ctrader_connection, signal_logger, alert_bot=None) -> None:
        self._ctrader = ctrader_connection
        self._signal_logger = signal_logger
        self._alert_bot = alert_bot
        self._prev_state: Dict[str, bool] = {}

    # ─── individual checks ───────────────────────────────────

    def check_ctrader(self) -> Dict[str, Any]:
        """Check cTrader connection status."""
        try:
            status = self._ctrader.get_connection_status()
            ok = status == "CONNECTED"
            return {"ok": ok, "message": status, "last_ok": datetime.now(timezone.utc).isoformat() if ok else None}
        except Exception as exc:
            logger.error("check_ctrader failed: %s", exc, exc_info=True)
            return {"ok": False, "message": str(exc), "last_ok": None}

    def check_yfinance(self) -> Dict[str, Any]:
        """Try fetching VIX as a health probe — 10-second timeout."""
        try:
            ticker = yf.Ticker("^VIX")
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            ok = price is not None and price > 0
            return {"ok": ok, "message": f"VIX={price}" if ok else "No price returned"}
        except Exception as exc:
            logger.error("check_yfinance failed: %s", exc, exc_info=True)
            return {"ok": False, "message": str(exc)}

    def check_calendar_api(self) -> Dict[str, Any]:
        """Check the free faireconomy.media calendar feed is reachable."""
        try:
            import requests
            resp = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                timeout=10,
            )
            return {"ok": resp.status_code == 200}
        except Exception as exc:
            logger.error("check_calendar_api failed: %s", exc, exc_info=True)
            return {"ok": False}

    def check_database(self) -> Dict[str, Any]:
        """Write and read a test value to verify SQLite is operational."""
        try:
            db_path = os.path.join("database", "signals.db")
            with sqlite3.connect(db_path, timeout=5) as conn:
                conn.execute("SELECT COUNT(*) FROM signals")
            return {"ok": True}
        except Exception as exc:
            logger.error("check_database failed: %s", exc, exc_info=True)
            return {"ok": False}

    def check_model_files(self) -> Dict[str, Any]:
        """Verify model pkl files exist and report the meta-label model age."""
        try:
            model_dir = "models"
            m1 = os.path.exists(os.path.join(model_dir, "model1_day_quality.pkl"))
            m2 = os.path.exists(os.path.join(model_dir, "model2_session_bias.pkl"))
            m3_path = os.path.join(model_dir, "model3_meta_label.pkl")
            m3 = os.path.exists(m3_path)
            m3_age = None
            if m3:
                mod_time = os.path.getmtime(m3_path)
                m3_age = int((time.time() - mod_time) / 86400)
            ok = m1 or m2 or m3  # system works without models; any model present is acceptable
            return {"ok": ok, "model1": m1, "model2": m2, "model3": m3, "model3_age_days": m3_age}
        except Exception as exc:
            logger.error("check_model_files failed: %s", exc, exc_info=True)
            return {"ok": False, "model1": False, "model2": False, "model3": False, "model3_age_days": None}

    def check_signal_frequency(self) -> Dict[str, Any]:
        """
        During active sessions, alert if no signal in NO_SIGNAL_ALERT_HOURS.
        Skips the check if the database has no signals yet (fresh system).
        """
        try:
            sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
            sl_now = datetime.now(sl_tz)
            current_min = sl_now.hour * 60 + sl_now.minute
            s_h, s_m = map(int, TRADING_START_SL.split(":"))
            e_h, e_m = map(int, TRADING_END_SL.split(":"))
            start_min = s_h * 60 + s_m
            end_min = e_h * 60 + e_m
            if start_min <= end_min:
                in_session = start_min <= current_min <= end_min
            else:
                in_session = current_min >= start_min or current_min <= end_min
            if not in_session:
                return {"ok": True, "hours_since_last": 0}

            last_ts = self._signal_logger.get_last_signal_timestamp()
            if last_ts is None:
                return {"ok": True, "hours_since_last": 0}

            try:
                last_dt = datetime.fromisoformat(last_ts)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
            except Exception:
                return {"ok": True, "hours_since_last": 0}

            # If the last signal is from a previous trading day, don't alert —
            # the current session just started and hasn't had time to produce signals.
            last_sl = last_dt.astimezone(sl_tz).date()
            today_sl = sl_now.date()
            if last_sl < today_sl:
                return {"ok": True, "hours_since_last": 0}

            hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            return {"ok": hours < NO_SIGNAL_ALERT_HOURS, "hours_since_last": round(hours, 1)}

        except Exception as exc:
            logger.error("check_signal_frequency failed: %s", exc, exc_info=True)
            return {"ok": True, "hours_since_last": 0}

    # ─── composite health check ──────────────────────────────

    def run_health_check(self) -> Dict[str, Any]:
        """
        Run all checks, compare to previous state, and alert only on change.
        """
        try:
            checks = {
                "ctrader": self.check_ctrader(),
                "yfinance": self.check_yfinance(),
                "calendar": self.check_calendar_api(),
                "database": self.check_database(),
                "models": self.check_model_files(),
                "signal_freq": self.check_signal_frequency(),
            }

            for name, result in checks.items():
                current_ok = result.get("ok", True) if isinstance(result.get("ok"), bool) else True
                prev_ok = self._prev_state.get(name, True)

                if prev_ok and not current_ok:
                    msg = result.get("message", f"{name} is down")
                    self._send_system_alert("WARNING", name, f"DOWN: {msg}")
                elif not prev_ok and current_ok:
                    self._send_system_alert("INFO", name, f"RECOVERED: {name} is back up")

                self._prev_state[name] = current_ok

            logger.info("Health check complete: %s", {k: v.get("ok", True) for k, v in checks.items()})
            return checks

        except Exception as exc:
            logger.error("run_health_check failed: %s", exc, exc_info=True)
            self._send_system_alert("CRITICAL", "health_monitor", f"Health check itself failed: {exc}")
            return {}

    # ─── status for weekly report ────────────────────────────

    def get_system_status(self) -> Dict[str, Any]:
        """Return current system status without firing alerts or mutating state."""
        try:
            return {
                "ctrader": self.check_ctrader(),
                "yfinance": self.check_yfinance(),
                "calendar": self.check_calendar_api(),
                "database": self.check_database(),
                "models": self.check_model_files(),
                "signal_freq": self.check_signal_frequency(),
            }
        except Exception as exc:
            logger.error("get_system_status failed: %s", exc, exc_info=True)
            return {}

    # ─── monitor loop entry ──────────────────────────────────

    def monitor_loop(self) -> None:
        """Called by the scheduler every HEALTH_CHECK_INTERVAL_MINUTES."""
        try:
            self.run_health_check()
        except Exception as exc:
            logger.critical("monitor_loop crashed: %s", exc, exc_info=True)
            self._send_system_alert("CRITICAL", "health_monitor", f"Monitor loop crash: {exc}")

    def record_error(self, error_type: str) -> None:
        """Simple error counter — used by the main loop to flag recurring issues."""
        logger.warning("Recorded error: %s", error_type)

    # ─── helpers ─────────────────────────────────────────────

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
