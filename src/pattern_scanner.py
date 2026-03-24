"""
Pattern Scanner — proactive early-warning system (Stage 2+).
Runs every 5 minutes during trading sessions.  Uses Model 3 to find
market conditions that closely resemble YOUR historical winning setups,
and sends an early heads-up alert before the full checklist fires.

Never replaces the checklist — early warning only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("pattern_scanner")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/pattern_scanner.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

STAGE_FILE = os.path.join("models", "stage.json")
PATTERN_WIN_PROB_THRESHOLD = 0.80
PATTERN_MIN_SIMILAR_WINS = 10
PATTERN_DEDUP_MINUTES = 30


class PatternScanner:
    """Scans current conditions against historical winning patterns."""

    def __init__(
        self,
        macro_checker,
        zone_calculator,
        orderflow_analyzer,
        checklist_engine,
        ctrader_connection,
        us500_id: int,
        signal_logger,
        model_predictor,
        alert_bot=None,
    ) -> None:
        self._macro = macro_checker
        self._zones = zone_calculator
        self._orderflow = orderflow_analyzer
        self._checklist = checklist_engine
        self._ctrader = ctrader_connection
        self._us500_id = us500_id
        self._signal_logger = signal_logger
        self._model_predictor = model_predictor
        self._alert_bot = alert_bot
        self._last_alert_time: float = 0.0
        self._last_alert_direction: Optional[str] = None
        self._wins_cache: Optional[pd.DataFrame] = None
        self._wins_cache_time: float = 0.0

    # ─── activation gate ─────────────────────────────────────

    def is_active(self) -> bool:
        """True only when stage.json shows stage >= 2 (500+ signals)."""
        try:
            if not os.path.exists(STAGE_FILE):
                return False
            with open(STAGE_FILE) as f:
                data = json.load(f)
            return data.get("stage", 0) >= 2
        except Exception:
            return False

    # ─── market snapshot ─────────────────────────────────────

    def get_current_market_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Collect all current market conditions in the same numeric format
        that Model 3 was trained on, so it can score them directly.
        """
        try:
            macro = self._macro.get_layer1_result()
            current_price = self._ctrader.get_current_price(self._us500_id)
            if current_price is None:
                return None

            zone_result = self._zones.get_layer3_result(current_price)
            orderflow = self._orderflow.get_layer4_result(
                macro.get("bias"), macro,
            )
            session = self._checklist.get_current_session()
            day_flags = self._checklist.get_day_flags()

            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_price": round(current_price, 2),
                "session": session,
                "day_name": day_flags.get("day_name", ""),
                "is_monday": day_flags.get("is_monday", False),
                "is_friday": day_flags.get("is_friday", False),
                "is_news_day": False,

                "vix_level": macro.get("vix_value", 0),
                "vix_pct": macro.get("vix_pct", 0),
                "vix_bucket": macro.get("vix_bucket", "normal"),
                "vix_direction_bias": macro.get("vix_direction_bias", "NEUTRAL"),
                "macro_bias": macro.get("bias", "MIXED"),
                "us10y_direction": macro.get("us10y_direction", "flat"),
                "oil_direction": macro.get("oil_direction", "stable"),
                "dxy_direction": macro.get("dxy_direction", "flat"),
                "rut_direction": macro.get("rut_direction", "neutral"),
                "groq_sentiment": macro.get("groq_sentiment", "NEUTRAL"),
                "bullish_count": macro.get("bullish_count", 0),
                "bearish_count": macro.get("bearish_count", 0),

                "at_zone": zone_result.get("at_zone", False),
                "zone_type": zone_result.get("zone_type"),
                "zone_level": zone_result.get("zone_level"),
                "zone_distance": zone_result.get("distance", 999),
                "zone_direction": zone_result.get("zone_direction"),

                "delta_direction": orderflow.get("delta_direction", "mixed"),
                "divergence": orderflow.get("divergence", "none"),
                "vix_spiking_now": orderflow.get("vix_spiking_now", False),
                "confirms_bias": orderflow.get("confirms_bias", False),

                "score": 0,
                "direction": macro.get("bias") if macro.get("bias") in ("LONG", "SHORT") else None,
            }

            return snapshot

        except Exception as exc:
            logger.error("get_current_market_snapshot failed: %s", exc, exc_info=True)
            return None

    # ─── similarity scoring ──────────────────────────────────

    def calculate_pattern_similarity(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Model 3 to predict win probability on the current snapshot,
        and find the most similar historical winning signals.
        """
        try:
            if not self._model_predictor or "model3" not in self._model_predictor.models:
                return self._empty_similarity()

            checklist_like = self._snapshot_to_checklist(snapshot)
            win_prob = self._model_predictor.predict_win_probability(checklist_like)

            winning_signals = self._get_winning_signals()
            similar_count = 0
            top_features: List[str] = []
            nearest_win: Optional[Dict] = None

            if not winning_signals.empty and win_prob is not None:
                similar_count, nearest_win, top_features = self._find_similar_wins(
                    snapshot, winning_signals,
                )

            shap_feats = self._model_predictor.get_shap_explanation(checklist_like)

            return {
                "win_probability": win_prob or 0.0,
                "similar_signals_count": similar_count,
                "top_matching_features": shap_feats or top_features,
                "nearest_historical_win": nearest_win,
            }

        except Exception as exc:
            logger.error("calculate_pattern_similarity failed: %s", exc, exc_info=True)
            return self._empty_similarity()

    def _get_winning_signals(self) -> pd.DataFrame:
        try:
            import time as _time
            if self._wins_cache is not None and (_time.time() - self._wins_cache_time) < 1800:
                return self._wins_cache
            data = self._signal_logger.get_training_data()
            if data.empty:
                return data
            self._wins_cache = data[data["outcome_label"] == 1].copy()
            self._wins_cache_time = _time.time()
            return self._wins_cache
        except Exception:
            return pd.DataFrame()

    def _find_similar_wins(
        self, snapshot: Dict, wins_df: pd.DataFrame
    ) -> tuple:
        """
        Count how many historical wins share the same key conditions,
        and identify which features are matching.
        """
        try:
            str_match_cols = {
                "macro_bias": snapshot.get("macro_bias"),
                "vix_bucket": snapshot.get("vix_bucket"),
                "delta_direction": snapshot.get("delta_direction"),
            }
            int_match_cols = {
                "at_zone": 1 if snapshot.get("at_zone") else 0,
            }

            session_map = {"Asia": 0, "London": 1, "NY_Open_Killzone": 2, "NY_Session": 3, "Off_Hours": 4}
            snap_session = session_map.get(snapshot.get("session"), -1)

            mask = pd.Series(True, index=wins_df.index)
            matching_features: List[str] = []

            for col, val in str_match_cols.items():
                if col in wins_df.columns and val is not None:
                    col_mask = wins_df[col].astype(str) == str(val)
                    mask &= col_mask
                    if col_mask.any():
                        matching_features.append(f"{col}={val}")

            for col, val in int_match_cols.items():
                if col in wins_df.columns:
                    col_mask = wins_df[col].fillna(-1).astype(int) == int(val)
                    mask &= col_mask
                    if col_mask.any():
                        matching_features.append(f"{col}={val}")

            if "session" in wins_df.columns and snap_session >= 0:
                sess_names = {0: "Asia", 1: "London", 2: "NY_Open_Killzone", 3: "NY_Session", 4: "Off_Hours"}
                sess_mask = wins_df["session"] == sess_names.get(snap_session, "")
                if sess_mask.any():
                    mask &= sess_mask
                    matching_features.append(f"session={snapshot.get('session')}")

            similar = wins_df[mask]
            count = len(similar)

            nearest = None
            if not similar.empty:
                row = similar.iloc[-1]
                nearest = {
                    "signal_id": int(row.get("id", 0)),
                    "direction": row.get("direction"),
                    "session": row.get("session"),
                    "pnl_points": row.get("pnl_points"),
                }

            return count, nearest, matching_features[:5]

        except Exception as exc:
            logger.error("_find_similar_wins failed: %s", exc, exc_info=True)
            return 0, None, []

    # ─── alert gating ────────────────────────────────────────

    def should_send_early_warning(self, similarity: Dict[str, Any], snapshot: Dict[str, Any]) -> bool:
        """Gate early-warning alerts: high probability, enough data, not spammed, no active signal."""
        try:
            if similarity.get("win_probability", 0) < PATTERN_WIN_PROB_THRESHOLD:
                return False
            if similarity.get("similar_signals_count", 0) < PATTERN_MIN_SIMILAR_WINS:
                return False
            if not self._checklist.is_trading_session():
                return False

            direction = snapshot.get("direction")
            if direction and self._signal_logger.is_duplicate(direction, snapshot.get("current_price", 0)):
                return False

            now = time.time()
            if (now - self._last_alert_time < PATTERN_DEDUP_MINUTES * 60
                    and self._last_alert_direction == direction):
                return False

            return True

        except Exception:
            return False

    # ─── alert formatting & sending ──────────────────────────

    def send_early_warning_alert(self, snapshot: Dict[str, Any], similarity: Dict[str, Any]) -> None:
        """Format and dispatch the pattern-forming alert to the TRADE channel."""
        try:
            wp = similarity.get("win_probability", 0)
            n = similarity.get("similar_signals_count", 0)
            features = similarity.get("top_matching_features", [])
            direction = snapshot.get("direction", "?")
            zone_level = snapshot.get("zone_level", "?")
            zone_type = snapshot.get("zone_type", "?")
            session = snapshot.get("session", "?")

            from config import SL_UTC_OFFSET
            sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
            sl_time = datetime.now(sl_tz).strftime("%H:%M SL")

            lines = [
                "\u26a1 PATTERN FORMING",
                "\u2501" * 20,
                f"Similarity to your winning setups: {wp:.0%}",
                f"Based on {n} historical wins",
                "\u2501" * 20,
                f"Watch: {zone_level} ({zone_type})",
                f"Direction: {direction}",
                "\u2501" * 20,
                "Matching conditions:",
            ]
            for feat in features[:3]:
                lines.append(f"\u2713 {feat}")
            lines.append("\u2501" * 20)
            lines.append("\u23f3 Not entry yet")
            lines.append("Wait for checklist confirmation")
            lines.append(f"{sl_time} | {session}")

            msg = "\n".join(lines)

            if self._alert_bot:
                self._alert_bot.send_trade_message(msg)

            self._last_alert_time = time.time()
            self._last_alert_direction = snapshot.get("direction")

            self._signal_logger.log_pattern_alert({
                "timestamp": snapshot.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "win_probability": wp,
                "snapshot_json": json.dumps(snapshot, default=str),
                "direction": direction,
            })

            logger.info("Early warning sent: prob=%.0f%% n=%d dir=%s", wp * 100, n, direction)

        except Exception as exc:
            logger.error("send_early_warning_alert failed: %s", exc, exc_info=True)

    # ─── main entry point ────────────────────────────────────

    def run_pattern_scan(self) -> None:
        """
        Called every 5 minutes.  Exits silently if stage < 2
        or Model 3 is not loaded.
        """
        try:
            if not self.is_active():
                return

            if not self._checklist.is_trading_session():
                return

            snapshot = self.get_current_market_snapshot()
            if snapshot is None:
                return

            similarity = self.calculate_pattern_similarity(snapshot)

            if self.should_send_early_warning(similarity, snapshot):
                self.send_early_warning_alert(snapshot, similarity)

        except Exception as exc:
            logger.error("run_pattern_scan failed: %s", exc, exc_info=True)

    # ─── outcome tracking ────────────────────────────────────

    def track_pattern_alert_outcome(self) -> Optional[Dict[str, Any]]:
        """
        Check if any recent pattern alert was followed by a real
        checklist signal within 30 minutes in the same direction.
        Returns match info or None.
        """
        try:
            recent_patterns = self._signal_logger.get_recent_pattern_alerts(10)
            if not recent_patterns:
                return None

            recent_signals = self._signal_logger.get_recent_signals(20)
            if not recent_signals:
                return None

            for pa in recent_patterns:
                if pa.get("matched_signal_id"):
                    continue

                pa_ts = pa.get("timestamp", "")
                pa_dir = pa.get("direction")
                try:
                    pa_dt = datetime.fromisoformat(pa_ts)
                    if pa_dt.tzinfo is None:
                        pa_dt = pa_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                for sig in recent_signals:
                    sig_ts = sig.get("timestamp", "")
                    sig_dir = sig.get("direction")
                    try:
                        sig_dt = datetime.fromisoformat(sig_ts)
                        if sig_dt.tzinfo is None:
                            sig_dt = sig_dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        continue

                    delta_min = (sig_dt - pa_dt).total_seconds() / 60.0
                    if 0 < delta_min <= 30 and pa_dir == sig_dir:
                        self._signal_logger.update_pattern_alert_match(
                            pa["id"], sig.get("id"),
                        )
                        return {
                            "pattern_alert_id": pa["id"],
                            "signal_id": sig.get("id"),
                            "minutes_early": round(delta_min, 1),
                            "direction": pa_dir,
                        }

            return None

        except Exception as exc:
            logger.error("track_pattern_alert_outcome failed: %s", exc, exc_info=True)
            return None

    # ─── helpers ─────────────────────────────────────────────

    def _snapshot_to_checklist(self, snapshot: Dict) -> Dict[str, Any]:
        """Wrap a snapshot dict so predict_win_probability can consume it."""
        return {
            "layer1": snapshot,
            "layer2": snapshot,
            "layer3": snapshot,
            "layer4": snapshot,
            "entry": {},
            **snapshot,
        }

    @staticmethod
    def _empty_similarity() -> Dict[str, Any]:
        return {
            "win_probability": 0.0,
            "similar_signals_count": 0,
            "top_matching_features": [],
            "nearest_historical_win": None,
        }
