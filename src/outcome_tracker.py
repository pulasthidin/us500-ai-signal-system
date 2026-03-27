"""
Outcome Tracker — checks pending signals against price data to resolve
WIN / LOSS / TIMEOUT via the triple-barrier method.

Includes crash-recovery catch-up: on startup, resolves ALL missed signals
(even weeks old) using historical bars with H1 fallback.
Uses a checkpoint file so interrupted checks can resume cleanly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from config import (
    OUTCOME_CHECK_DELAY_SECONDS,
    OUTCOME_MIN_H1_BARS_FOR_TIMEOUT,
    OUTCOME_MIN_M5_BARS_FOR_TIMEOUT,
)

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logger = logging.getLogger("outcome")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/outcome.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

CHECKPOINT_PATH = os.path.join("data", "checkpoint.json")


class OutcomeTracker:
    """Resolves open signals to WIN/LOSS/TIMEOUT using a triple-barrier approach."""

    def __init__(self, ctrader_connection, us500_id: int, signal_logger, alert_bot=None) -> None:
        self._ctrader = ctrader_connection
        self._us500_id = us500_id
        self._signal_logger = signal_logger
        self._alert_bot = alert_bot

    # ══════════════════════════════════════════════════════════
    # STARTUP CATCH-UP — resolve ALL missed signals
    # ══════════════════════════════════════════════════════════

    def run_startup_catchup(self) -> Dict[str, int]:
        """
        Called once on app startup.  Resolves:
        1. ALL signals with outcome IS NULL (Phase 1 — TP1/TP2/SL check)
        2. ALL signals stuck in PARTIAL_WIN (Phase 2 — TP2 vs breakeven)
        Returns a summary dict {checked, wins, losses, timeouts, skipped, upgraded}.
        """
        summary = {
            "checked": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "skipped": 0, "estimated": 0, "upgraded": 0,
        }
        try:
            self._resume_interrupted_checkpoint()

            all_null = self._signal_logger.get_all_null_outcome_signals()
            partial_wins = self._signal_logger.get_partial_win_signals()

            if not all_null and not partial_wins:
                logger.info("Catch-up: no missed signals")
                return summary

            now = datetime.now(timezone.utc)
            logger.info(
                "Catch-up: found %d NULL-outcome + %d PARTIAL_WIN signals",
                len(all_null), len(partial_wins),
            )

            for signal in all_null:
                try:
                    signal_age = self._signal_age_hours(signal, now)

                    if signal_age < (OUTCOME_CHECK_DELAY_SECONDS / 3600):
                        summary["skipped"] += 1
                        continue

                    self._save_checkpoint(signal["id"])
                    result = self._resolve_signal_with_fallback(signal)
                    self._clear_checkpoint()

                    if result is None:
                        summary["skipped"] += 1
                        continue

                    summary["checked"] += 1
                    outcome = result["outcome"]
                    if outcome in ("WIN", "ESTIMATED_WIN"):
                        summary["wins"] += 1
                    elif "PARTIAL_WIN" in outcome:
                        summary["wins"] += 1
                    elif outcome in ("LOSS", "ESTIMATED_LOSS"):
                        summary["losses"] += 1
                    elif outcome in ("TIMEOUT", "ESTIMATED_TIMEOUT"):
                        summary["timeouts"] += 1
                    if result.get("estimated"):
                        summary["estimated"] += 1

                except Exception as exc:
                    logger.error("Catch-up failed for signal %s: %s", signal.get("id"), exc, exc_info=True)
                    self._clear_checkpoint()

            for signal in partial_wins:
                try:
                    self._save_checkpoint(signal["id"])
                    result = self._resolve_signal_with_fallback(signal)
                    self._clear_checkpoint()

                    if result is None:
                        summary["skipped"] += 1
                        continue

                    summary["checked"] += 1
                    if result.get("upgraded_from_partial"):
                        summary["upgraded"] += 1
                        summary["wins"] += 1
                        logger.info("Catch-up: signal %s upgraded PARTIAL_WIN -> WIN", signal.get("id"))
                    elif result["outcome"] == "PARTIAL_WIN_FINAL":
                        logger.info("Catch-up: signal %s finalized as PARTIAL_WIN_FINAL", signal.get("id"))

                except Exception as exc:
                    logger.error("Catch-up Phase 2 failed for signal %s: %s", signal.get("id"), exc, exc_info=True)
                    self._clear_checkpoint()

            logger.info("Catch-up complete: %s", summary)
            return summary

        except Exception as exc:
            logger.error("run_startup_catchup failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "catchup", str(exc))
            return summary

    # ──────────────────────────────────────────────────────────
    # SCHEDULED LOOP — normal 30-min cycle (unchanged logic)
    # ──────────────────────────────────────────────────────────

    def track_pending_outcomes(self) -> None:
        """Retrieve pending signals (>2 hrs old, oldest first) and resolve via triple barrier."""
        try:
            pending = self._signal_logger.get_pending_signals()
            if not pending:
                return

            logger.info("Processing %d pending outcomes", len(pending))

            for signal in pending:
                try:
                    self._save_checkpoint(signal["id"])
                    self._resolve_signal_with_fallback(signal)
                    self._clear_checkpoint()
                except Exception as exc:
                    logger.error("Failed to resolve signal %s: %s", signal.get("id"), exc, exc_info=True)
                    self._clear_checkpoint()

        except Exception as exc:
            logger.error("track_pending_outcomes failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "outcome_tracker", str(exc))

    # ──────────────────────────────────────────────────────────
    # RESOLVE WITH FALLBACK — M5 -> H1 -> TIMEOUT
    # ──────────────────────────────────────────────────────────

    def _resolve_signal_with_fallback(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Try M5 bars first; if not enough data, skip (return None).
        Fall back to H1 only when M5 bars are completely unavailable.
        Hard TIMEOUT only when signal is very old (>48h) and no resolution found.
        """
        sid = signal.get("id")

        bars_df = self._ctrader.fetch_bars(self._us500_id, "M5", 500)
        if bars_df is not None and not bars_df.empty:
            result = self.triple_barrier_check(signal, bars_df)
            if result is not None:
                if result.get("tp1_hit") and result["outcome"] == "PARTIAL_WIN":
                    self._signal_logger.update_tp1_hit(
                        sid, result["outcome_timestamp"],
                    )
                elif result.get("upgraded_from_partial"):
                    self._signal_logger.update_outcome(
                        signal_id=sid, outcome="WIN", label=1,
                        bars=result["bars_to_outcome"], pnl=result["pnl_points"],
                        timestamp=result["outcome_timestamp"],
                    )
                elif result["outcome"] == "PARTIAL_WIN_FINAL":
                    self._signal_logger.update_outcome(
                        signal_id=sid, outcome="PARTIAL_WIN_FINAL", label=1,
                        bars=result["bars_to_outcome"], pnl=result["pnl_points"],
                        timestamp=result["outcome_timestamp"],
                    )
                else:
                    self._signal_logger.update_outcome(
                        signal_id=sid, outcome=result["outcome"], label=result["label"],
                        bars=result["bars_to_outcome"], pnl=result["pnl_points"],
                        timestamp=result["outcome_timestamp"],
                    )
                return result
            logger.info("Signal %s: M5 check returned None (not enough bars yet), will retry", sid)
            return None

        logger.warning("M5 bars unavailable for signal %s — trying H1 fallback", sid)
        h1_df = self._ctrader.fetch_bars(self._us500_id, "H1", 200)
        if h1_df is not None and not h1_df.empty:
            result = self.triple_barrier_check(signal, h1_df, min_bars_for_timeout=OUTCOME_MIN_H1_BARS_FOR_TIMEOUT)
            if result is not None:
                if result.get("tp1_hit") and result["outcome"] == "PARTIAL_WIN":
                    self._signal_logger.update_tp1_hit(
                        sid, result["outcome_timestamp"],
                    )
                    logger.info("Signal %s: H1 fallback found TP1 hit — PARTIAL_WIN (Phase 2 will follow)", sid)
                    return result
                result["outcome"] = f"ESTIMATED_{result['outcome']}"
                result["estimated"] = True
                self._signal_logger.update_outcome(
                    signal_id=sid, outcome=result["outcome"], label=result["label"],
                    bars=result["bars_to_outcome"], pnl=result["pnl_points"],
                    timestamp=result["outcome_timestamp"],
                )
                logger.info("Signal %s resolved via H1 fallback: %s", sid, result["outcome"])
                return result

        # Only hard-TIMEOUT when signal is very old and no bars at all
        signal_age = self._signal_age_hours(signal, datetime.now(timezone.utc))
        if signal_age > 48:
            logger.warning("Signal %s is %.1f hours old with no resolution — marking TIMEOUT", sid, signal_age)
            result = {
                "outcome": "TIMEOUT", "label": 0, "bars_to_outcome": 0,
                "pnl_points": 0.0, "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._signal_logger.update_outcome(
                signal_id=sid, outcome="TIMEOUT", label=0, bars=0, pnl=0.0,
                timestamp=result["outcome_timestamp"],
            )
            return result

        logger.info("Signal %s: no resolution yet (%.1fh old), will retry next cycle", sid, signal_age)
        return None

    # ──────────────────────────────────────────────────────────
    # TRIPLE BARRIER
    # ──────────────────────────────────────────────────────────

    def _get_bars_after_entry(self, signal: Dict[str, Any], bars_df: pd.DataFrame):
        """Parse signal timestamp and filter bars to only those after entry."""
        signal_ts = signal.get("timestamp", "")
        entry_time = None
        try:
            entry_time = pd.Timestamp(signal_ts)
            if entry_time.tzinfo is None:
                entry_time = entry_time.tz_localize("UTC")
        except Exception:
            logger.error(
                "Signal %s has unparseable timestamp: %r — skipping outcome check",
                signal.get("id"), signal_ts,
            )
            return None, None

        bars_df = bars_df.copy()
        ts_col = None
        for candidate in ("timestamp", "time", "datetime"):
            if candidate in bars_df.columns:
                ts_col = candidate
                break

        if ts_col:
            bars_df.sort_values(ts_col, inplace=True)
        bars_df.reset_index(drop=True, inplace=True)

        if entry_time and ts_col:
            bars_df[ts_col] = pd.to_datetime(bars_df[ts_col], utc=True)
            bars_after = bars_df[bars_df[ts_col] >= entry_time]
        else:
            bars_after = bars_df

        return bars_after, ts_col

    def triple_barrier_check(
        self, signal: Dict[str, Any], bars_df: pd.DataFrame,
        min_bars_for_timeout: int = OUTCOME_MIN_M5_BARS_FOR_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        """
        Quad barrier check with TP1 support.

        Phase 1 (outcome IS NULL): Check TP1 and SL
          - SL hit first -> LOSS
          - TP1 hit first -> PARTIAL_WIN (will be rechecked for TP2 later)
          - TP2 hit first -> WIN (skipped TP1, full target reached)

        Phase 2 (outcome = PARTIAL_WIN): Check TP2 and breakeven SL
          - TP2 hit -> upgrade to WIN
          - Breakeven SL (entry price) hit -> keep PARTIAL_WIN, finalize
          - Timeout -> keep PARTIAL_WIN, finalize
        """
        try:
            direction = signal.get("direction")
            sl_price = signal.get("sl_price")
            tp_price = signal.get("tp_price")
            tp1_price = signal.get("tp1_price")
            entry_price = signal.get("entry_price")
            current_outcome = signal.get("outcome")

            if direction is None or sl_price is None or tp_price is None or entry_price is None:
                logger.warning("Signal %s missing SL/TP/entry — skipping", signal.get("id"))
                return None

            bars_after, ts_col = self._get_bars_after_entry(signal, bars_df)
            if bars_after is None or bars_after.empty:
                return None

            is_partial_win = current_outcome == "PARTIAL_WIN"

            if is_partial_win:
                return self._check_phase2(signal, bars_after, ts_col, min_bars_for_timeout)
            else:
                return self._check_phase1(signal, bars_after, ts_col, min_bars_for_timeout)

        except Exception as exc:
            logger.error("triple_barrier_check failed: %s", exc, exc_info=True)
            return None

    def _check_phase1(self, signal, bars_after, ts_col, min_bars_for_timeout):
        """Phase 1: Check TP1/TP2/SL for signals with no outcome yet."""
        direction = signal["direction"]
        sl_price = signal["sl_price"]
        tp_price = signal["tp_price"]
        tp1_price = signal.get("tp1_price")
        entry_price = signal["entry_price"]

        for bar_num, (_, bar) in enumerate(bars_after.iterrows(), start=1):
            if direction == "SHORT":
                sl_hit = bar["high"] >= sl_price
                tp2_hit = bar["low"] <= tp_price
                tp1_hit = tp1_price is not None and bar["low"] <= tp1_price
            elif direction == "LONG":
                sl_hit = bar["low"] <= sl_price
                tp2_hit = bar["high"] >= tp_price
                tp1_hit = tp1_price is not None and bar["high"] >= tp1_price
            else:
                continue

            if sl_hit and tp2_hit:
                return self._barrier_result("LOSS", -1, bar_num, signal, bar, ts_col or "timestamp")
            if sl_hit and tp1_hit:
                return self._barrier_result("LOSS", -1, bar_num, signal, bar, ts_col or "timestamp")
            if sl_hit:
                return self._barrier_result("LOSS", -1, bar_num, signal, bar, ts_col or "timestamp")
            if tp2_hit:
                pnl = self._calc_full_pnl_after_tp1(signal) if tp1_price else self.calculate_pnl(signal, "WIN")
                ts = str(bar.get(ts_col, "")) if ts_col else datetime.now(timezone.utc).isoformat()
                return {"outcome": "WIN", "label": 1, "bars_to_outcome": bar_num, "pnl_points": pnl, "outcome_timestamp": ts}
            if tp1_hit:
                ts = str(bar.get(ts_col, "")) if ts_col else datetime.now(timezone.utc).isoformat()
                pnl = self._calc_tp1_pnl(signal)
                return {
                    "outcome": "PARTIAL_WIN", "label": 1,
                    "bars_to_outcome": bar_num, "pnl_points": pnl,
                    "outcome_timestamp": ts, "tp1_hit": True,
                }

        if len(bars_after) < min_bars_for_timeout:
            logger.info(
                "Signal %s: only %d bars after entry (need %d) — skipping",
                signal.get("id"), len(bars_after), min_bars_for_timeout,
            )
            return None

        return {
            "outcome": "TIMEOUT", "label": 0, "bars_to_outcome": len(bars_after),
            "pnl_points": 0.0, "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _check_phase2(self, signal, bars_after, ts_col, min_bars_for_timeout):
        """Phase 2: Signal already hit TP1 (PARTIAL_WIN). Check TP2 vs breakeven."""
        direction = signal["direction"]
        tp_price = signal["tp_price"]
        entry_price = signal["entry_price"]
        tp1_ts = signal.get("tp1_hit_timestamp", signal.get("outcome_timestamp", ""))

        start_time = None
        if tp1_ts:
            try:
                start_time = pd.Timestamp(tp1_ts)
                if start_time.tzinfo is None:
                    start_time = start_time.tz_localize("UTC")
            except Exception:
                pass

        if start_time is not None and ts_col and ts_col in bars_after.columns:
            phase2_bars = bars_after[bars_after[ts_col] >= start_time]
        else:
            phase2_bars = bars_after

        if phase2_bars.empty:
            return None

        for bar_num, (_, bar) in enumerate(phase2_bars.iterrows(), start=1):
            if direction == "SHORT":
                tp2_hit = bar["low"] <= tp_price
                be_hit = bar["high"] >= entry_price
            elif direction == "LONG":
                tp2_hit = bar["high"] >= tp_price
                be_hit = bar["low"] <= entry_price
            else:
                continue

            if tp2_hit and be_hit:
                pnl = self._calc_tp1_pnl(signal)
                ts = str(bar.get(ts_col, "")) if ts_col else datetime.now(timezone.utc).isoformat()
                logger.info("Signal %s: TP2+BE same bar after PARTIAL_WIN -> conservative BE (finalized)", signal.get("id"))
                return {
                    "outcome": "PARTIAL_WIN_FINAL", "label": 1,
                    "bars_to_outcome": bar_num, "pnl_points": pnl,
                    "outcome_timestamp": ts,
                }
            if tp2_hit:
                pnl = self._calc_full_pnl_after_tp1(signal)
                ts = str(bar.get(ts_col, "")) if ts_col else datetime.now(timezone.utc).isoformat()
                logger.info("Signal %s: TP2 hit after PARTIAL_WIN -> upgrading to WIN", signal.get("id"))
                return {
                    "outcome": "WIN", "label": 1,
                    "bars_to_outcome": bar_num, "pnl_points": pnl,
                    "outcome_timestamp": ts, "upgraded_from_partial": True,
                }
            if be_hit:
                pnl = self._calc_tp1_pnl(signal)
                ts = str(bar.get(ts_col, "")) if ts_col else datetime.now(timezone.utc).isoformat()
                logger.info("Signal %s: breakeven hit after PARTIAL_WIN -> finalized", signal.get("id"))
                return {
                    "outcome": "PARTIAL_WIN_FINAL", "label": 1,
                    "bars_to_outcome": bar_num, "pnl_points": pnl,
                    "outcome_timestamp": ts,
                }

        if len(phase2_bars) < min_bars_for_timeout:
            return None

        pnl = self._calc_tp1_pnl(signal)
        return {
            "outcome": "PARTIAL_WIN_FINAL", "label": 1,
            "bars_to_outcome": len(phase2_bars), "pnl_points": pnl,
            "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _calc_tp1_pnl(self, signal):
        """PnL for half position at TP1: tp1_distance / 2."""
        tp1_price = signal.get("tp1_price")
        entry = signal.get("entry_price", 0)
        if tp1_price is None:
            return 0.0
        if signal.get("direction") == "SHORT":
            return round((entry - tp1_price) / 2, 2)
        return round((tp1_price - entry) / 2, 2)

    def _calc_full_pnl_after_tp1(self, signal):
        """PnL for half at TP1 + half at TP2."""
        entry = signal.get("entry_price", 0)
        tp1 = signal.get("tp1_price", entry)
        tp2 = signal.get("tp_price", entry)
        if signal.get("direction") == "SHORT":
            return round(((entry - tp1) + (entry - tp2)) / 2, 2)
        return round(((tp1 - entry) + (tp2 - entry)) / 2, 2)

    def _barrier_result(self, outcome: str, label: int, bar_num: int, signal: Dict, bar, ts_col: str = "timestamp") -> Dict[str, Any]:
        pnl = self.calculate_pnl(signal, outcome)
        ts = str(bar.get(ts_col, bar.get("timestamp", datetime.now(timezone.utc).isoformat())))
        return {"outcome": outcome, "label": label, "bars_to_outcome": bar_num, "pnl_points": pnl, "outcome_timestamp": ts}

    # ──────────────────────────────────────────────────────────
    # PnL
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def calculate_pnl(signal: Dict[str, Any], outcome: str) -> float:
        try:
            entry = signal.get("entry_price", 0)
            sl = signal.get("sl_price", 0)
            tp = signal.get("tp_price", 0)
            direction = signal.get("direction", "")

            if outcome in ("TIMEOUT", "ESTIMATED_TIMEOUT"):
                return 0.0
            clean_outcome = outcome.replace("ESTIMATED_", "")
            if direction == "SHORT":
                return round(entry - tp, 2) if clean_outcome == "WIN" else round(entry - sl, 2)
            else:
                return round(tp - entry, 2) if clean_outcome == "WIN" else round(sl - entry, 2)
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────
    # CHECKPOINT FILE — resume after crash
    # ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, signal_id: int) -> None:
        try:
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump({"checking_signal_id": signal_id, "started": datetime.now(timezone.utc).isoformat()}, f)
        except Exception:
            pass

    def _clear_checkpoint(self) -> None:
        try:
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
        except Exception:
            pass

    def _resume_interrupted_checkpoint(self) -> None:
        """If a checkpoint file exists from a previous crash, re-check that signal."""
        try:
            if not os.path.exists(CHECKPOINT_PATH):
                return
            with open(CHECKPOINT_PATH) as f:
                data = json.load(f)
            signal_id = data.get("checking_signal_id")
            if signal_id is None:
                self._clear_checkpoint()
                return

            logger.warning("Resuming interrupted outcome check for signal %s", signal_id)
            all_null = self._signal_logger.get_all_null_outcome_signals()
            for sig in all_null:
                if sig.get("id") == signal_id:
                    self._resolve_signal_with_fallback(sig)
                    break
            self._clear_checkpoint()
        except Exception as exc:
            logger.error("_resume_interrupted_checkpoint failed: %s", exc, exc_info=True)
            self._clear_checkpoint()

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _signal_age_hours(signal: Dict[str, Any], now: datetime) -> float:
        try:
            ts = signal.get("timestamp", "")
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt).total_seconds() / 3600
        except Exception:
            return 999.0

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
