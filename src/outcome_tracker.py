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

from config import OUTCOME_CHECK_DELAY_SECONDS

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
        Called once on app startup.  Finds ALL signals with outcome IS NULL
        (no time limit), resolves any that are old enough (>= 2 hrs),
        and returns a summary dict {checked, wins, losses, timeouts, skipped}.
        """
        summary = {"checked": 0, "wins": 0, "losses": 0, "timeouts": 0, "skipped": 0, "estimated": 0}
        try:
            self._resume_interrupted_checkpoint()

            all_null = self._signal_logger.get_all_null_outcome_signals()
            if not all_null:
                logger.info("Catch-up: no missed signals")
                return summary

            now = datetime.now(timezone.utc)
            logger.info("Catch-up: found %d signals with NULL outcome", len(all_null))

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
                    if outcome == "WIN":
                        summary["wins"] += 1
                    elif outcome == "LOSS":
                        summary["losses"] += 1
                    elif outcome in ("TIMEOUT", "ESTIMATED_TIMEOUT"):
                        summary["timeouts"] += 1
                    if result.get("estimated"):
                        summary["estimated"] += 1

                except Exception as exc:
                    logger.error("Catch-up failed for signal %s: %s", signal.get("id"), exc, exc_info=True)
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
        """Try M5 bars first; if unavailable fall back to H1; if that fails mark TIMEOUT."""
        sid = signal.get("id")

        bars_df = self._ctrader.fetch_bars(self._us500_id, "M5", 500)
        if bars_df is not None and not bars_df.empty:
            result = self.triple_barrier_check(signal, bars_df)
            if result is not None:
                self._signal_logger.update_outcome(
                    signal_id=sid, outcome=result["outcome"], label=result["label"],
                    bars=result["bars_to_outcome"], pnl=result["pnl_points"],
                    timestamp=result["outcome_timestamp"],
                )
                return result

        logger.warning("M5 bars unavailable for signal %s — trying H1 fallback", sid)
        h1_df = self._ctrader.fetch_bars(self._us500_id, "H1", 200)
        if h1_df is not None and not h1_df.empty:
            result = self.triple_barrier_check(signal, h1_df)
            if result is not None:
                result["outcome"] = f"ESTIMATED_{result['outcome']}"
                result["estimated"] = True
                self._signal_logger.update_outcome(
                    signal_id=sid, outcome=result["outcome"], label=result["label"],
                    bars=result["bars_to_outcome"], pnl=result["pnl_points"],
                    timestamp=result["outcome_timestamp"],
                )
                logger.info("Signal %s resolved via H1 fallback: %s", sid, result["outcome"])
                return result

        logger.warning("No bars available for signal %s — marking TIMEOUT", sid)
        result = {
            "outcome": "TIMEOUT", "label": 0, "bars_to_outcome": 0,
            "pnl_points": 0.0, "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._signal_logger.update_outcome(
            signal_id=sid, outcome="TIMEOUT", label=0, bars=0, pnl=0.0,
            timestamp=result["outcome_timestamp"],
        )
        return result

    # ──────────────────────────────────────────────────────────
    # TRIPLE BARRIER
    # ──────────────────────────────────────────────────────────

    def triple_barrier_check(self, signal: Dict[str, Any], bars_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Walk through bars chronologically.  Whichever barrier is hit first wins:
          SL barrier -> LOSS,  TP barrier -> WIN,  end of bars -> TIMEOUT
        """
        try:
            direction = signal.get("direction")
            sl_price = signal.get("sl_price")
            tp_price = signal.get("tp_price")
            entry_price = signal.get("entry_price")

            if direction is None or sl_price is None or tp_price is None or entry_price is None:
                logger.warning("Signal %s missing SL/TP/entry — skipping", signal.get("id"))
                return None

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
                return None

            bars_df = bars_df.copy()
            # Find the timestamp column flexibly — cTrader may use different names
            ts_col = None
            for candidate in ("timestamp", "time", "datetime"):
                if candidate in bars_df.columns:
                    ts_col = candidate
                    break

            if ts_col:
                bars_df.sort_values(ts_col, inplace=True)
            else:
                logger.warning("No timestamp column in bars — using row order")
            bars_df.reset_index(drop=True, inplace=True)

            if entry_time and ts_col:
                bars_df[ts_col] = pd.to_datetime(bars_df[ts_col], utc=True)
                bars_after = bars_df[bars_df[ts_col] >= entry_time]
            else:
                bars_after = bars_df

            if bars_after.empty:
                return None

            for bar_num, (_, bar) in enumerate(bars_after.iterrows(), start=1):
                if direction == "SHORT":
                    sl_hit = bar["high"] >= sl_price
                    tp_hit = bar["low"] <= tp_price
                elif direction == "LONG":
                    sl_hit = bar["low"] <= sl_price
                    tp_hit = bar["high"] >= tp_price
                else:
                    continue

                if sl_hit and tp_hit:
                    # Both barriers touched in the same bar.  SL (1.5×ATR) is
                    # closer to entry than TP (2.5×ATR), so conservatively
                    # assume SL was hit first → LOSS.
                    return self._barrier_result("LOSS", -1, bar_num, signal, bar)
                if sl_hit:
                    return self._barrier_result("LOSS", -1, bar_num, signal, bar)
                if tp_hit:
                    return self._barrier_result("WIN", 1, bar_num, signal, bar)

            return {
                "outcome": "TIMEOUT", "label": 0, "bars_to_outcome": len(bars_after),
                "pnl_points": 0.0, "outcome_timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            logger.error("triple_barrier_check failed: %s", exc, exc_info=True)
            return None

    def _barrier_result(self, outcome: str, label: int, bar_num: int, signal: Dict, bar) -> Dict[str, Any]:
        pnl = self.calculate_pnl(signal, outcome)
        ts = str(bar.get("timestamp", datetime.now(timezone.utc).isoformat()))
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
