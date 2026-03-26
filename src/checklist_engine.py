"""
Main orchestrator — runs the 4-layer confluence checklist, applies filters,
decides grade/sizing, and determines whether to fire an alert.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

os.makedirs("logs", exist_ok=True)

from config import (
    EQH_EQL_NEARBY_POINTS,
    GRADE_A_SCORE,
    GRADE_B_SCORE,
    SESSIONS_UTC,
    SL_UTC_OFFSET,
    TRADING_END_SL,
    TRADING_START_SL,
)

logger = logging.getLogger("checklist")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/checklist.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class ChecklistEngine:
    """Orchestrates all four layers, entry check, filters, and final decision."""

    def __init__(
        self,
        macro_checker,
        structure_analyzer,
        zone_calculator,
        orderflow_analyzer,
        entry_checker,
        signal_logger,
        alert_bot=None,
    ) -> None:
        self._macro = macro_checker
        self._structure = structure_analyzer
        self._zones = zone_calculator
        self._orderflow = orderflow_analyzer
        self._entry = entry_checker
        self._signal_logger = signal_logger
        self._alert_bot = alert_bot
        self._hard_stop_active: bool = False

    # ─── session / time helpers ──────────────────────────────

    def get_current_session(self) -> str:
        """Map the current UTC time to a session label."""
        try:
            now = datetime.now(timezone.utc)
            current_minutes = now.hour * 60 + now.minute

            for name, window in SESSIONS_UTC.items():
                start_h, start_m = map(int, window["start"].split(":"))
                end_h, end_m = map(int, window["end"].split(":"))
                start_min = start_h * 60 + start_m
                end_min = end_h * 60 + end_m

                if start_min <= end_min:
                    if start_min <= current_minutes < end_min:
                        return name
                else:
                    if current_minutes >= start_min or current_minutes < end_min:
                        return name

            return "Off_Hours"
        except Exception as exc:
            logger.error("get_current_session failed: %s", exc, exc_info=True)
            return "Off_Hours"

    def get_sl_time(self) -> str:
        """Convert current UTC to Sri Lanka time string (HH:MM SL)."""
        try:
            sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
            sl_now = datetime.now(sl_tz)
            return sl_now.strftime("%H:%M SL")
        except Exception:
            return "??:?? SL"

    def is_trading_session(self) -> bool:
        """Check whether current SL time falls within the configured trading window.

        Handles midnight-spanning windows (e.g. start=22:00, end=02:00) using
        the same logic as get_current_session.
        """
        try:
            sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
            sl_now = datetime.now(sl_tz)
            current = sl_now.hour * 60 + sl_now.minute

            start_h, start_m = map(int, TRADING_START_SL.split(":"))
            end_h, end_m = map(int, TRADING_END_SL.split(":"))
            start_min = start_h * 60 + start_m
            end_min = end_h * 60 + end_m

            if start_min <= end_min:
                return start_min <= current < end_min
            else:
                # window spans midnight (e.g. 22:00 – 02:00)
                return current >= start_min or current < end_min
        except Exception as exc:
            logger.error("is_trading_session failed: %s", exc, exc_info=True)
            return False

    @staticmethod
    def _is_us_dst(dt_utc: datetime) -> bool:
        """
        Return True if *dt_utc* falls within US DST.
        DST begins: 02:00 ET on the second Sunday of March  (= 07:00 UTC)
        DST ends:   02:00 EDT on the first Sunday of November (= 06:00 UTC)
        """
        year = dt_utc.year
        # Second Sunday in March
        march1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        first_sun_march = march1 + timedelta(days=(6 - march1.weekday()) % 7)
        dst_start = first_sun_march + timedelta(weeks=1, hours=7)  # 07:00 UTC
        # First Sunday in November
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        first_sun_nov = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
        dst_end = first_sun_nov + timedelta(hours=6)               # 06:00 UTC
        return dst_start <= dt_utc < dst_end

    def get_day_flags(self) -> Dict[str, Any]:
        """Return day-of-week info and session-specific caution flags."""
        try:
            utc_now = datetime.now(timezone.utc)
            day_name = utc_now.strftime("%A")

            is_dst = self._is_us_dst(utc_now)
            ny_offset = -4 if is_dst else -5
            ny_tz = timezone(timedelta(hours=ny_offset))
            ny_now = datetime.now(ny_tz)
            ny_hour = ny_now.hour

            is_monday = day_name == "Monday"
            is_friday = day_name == "Friday"

            monday_caution = is_monday and 9 <= ny_hour < 11
            friday_caution = is_friday and ny_hour >= 14

            return {
                "day_name": day_name,
                "is_monday": is_monday,
                "is_friday": is_friday,
                "is_tuesday": day_name == "Tuesday",
                "monday_caution": monday_caution,
                "friday_caution": friday_caution,
            }
        except Exception as exc:
            logger.error("get_day_flags failed: %s", exc, exc_info=True)
            return {
                "day_name": "Unknown", "is_monday": False, "is_friday": False,
                "is_tuesday": False, "monday_caution": False, "friday_caution": False,
            }

    # ─── full checklist run ──────────────────────────────────

    def run_full_checklist(
        self,
        current_price: float,
        macro_data: Dict[str, Any],
        news_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute all 4 layers sequentially, derive direction, optionally run entry check,
        apply all filters (including range + sweep logic), and return the complete result dict.
        """
        try:
            day_flags = self.get_day_flags()
            session = self.get_current_session()
            sl_time = self.get_sl_time()

            layer1 = macro_data if "score_contribution" in macro_data else self._macro.get_layer1_result()
            layer2 = self._structure.get_layer2_result(current_price)
            layer3 = self._zones.get_layer3_result(current_price)

            direction, direction_confidence = self._derive_direction(layer1, layer2)

            layer4 = self._orderflow.get_layer4_result(direction, layer1)

            score = (
                layer1.get("score_contribution", 0)
                + layer2.get("score_contribution", 0)
                + layer3.get("score_contribution", 0)
                + layer4.get("score_contribution", 0)
            )

            range_condition = layer2.get("range_condition", {})
            liquidity_sweeps = layer3.get("liquidity_sweeps", [])
            is_ranging = range_condition.get("is_ranging", False)

            entry = None
            entry_ready = False
            entry_confidence = None
            if score >= GRADE_B_SCORE and direction:
                entry = self._entry.get_entry_result(
                    direction, current_price,
                    zone_levels=layer3.get("all_levels", []),
                    range_condition=range_condition,
                    liquidity_sweeps=liquidity_sweeps,
                )
                entry_ready = entry.get("entry_ready", False)
                entry_confidence = entry.get("entry_confidence")

            has_sweep = entry.get("has_liquidity_sweep", False) if entry else False

            if score >= GRADE_A_SCORE and entry_ready and entry_confidence == "full":
                decision = "FULL_SEND"
                grade = "A"
            elif score >= GRADE_A_SCORE and entry_ready:
                decision = "HALF_SIZE"
                grade = "B"
                logger.info(
                    "Grade A->B: entry_confidence=%s (missing %s)",
                    entry_confidence,
                    "BOS+SMT" if entry_confidence == "base" else "BOS or SMT",
                )
            elif score >= GRADE_B_SCORE and entry_ready:
                decision = "HALF_SIZE"
                grade = "B"
            elif score >= GRADE_B_SCORE and not entry_ready and direction:
                decision = "WAIT"
                grade = None
            else:
                decision = "NO_TRADE"
                grade = None

            if direction_confidence == "reduced" and grade == "A":
                grade = "B"
                decision = "HALF_SIZE"
                logger.info("Grade A downgraded to B — direction from structure alone")

            # Range + no-sweep downgrade: when trending without a sweep, still
            # allow the trade but mark it as lower confidence.  The entry_checker
            # already blocks entries in ranging markets without a sweep.
            if entry_ready and not has_sweep and not is_ranging:
                if grade == "A" and entry_confidence == "no_sweep":
                    grade = "B"
                    decision = "HALF_SIZE"
                    logger.info("Grade A->B: trending but no liquidity sweep confirmation")

            all_levels = layer3.get("all_levels", [])
            eqh_nearby = False
            eql_nearby = False
            eqh_level = None
            eql_level = None
            for lvl in all_levels:
                p = lvl.get("price")
                if p is None:
                    continue
                if lvl.get("type") == "eqh" and p > current_price and (p - current_price) <= EQH_EQL_NEARBY_POINTS:
                    eqh_nearby = True
                    eqh_level = p
                elif lvl.get("type") == "eql" and p < current_price and (current_price - p) <= EQH_EQL_NEARBY_POINTS:
                    eql_nearby = True
                    eql_level = p

            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sl_time": sl_time,
                "session": session,
                "day_name": day_flags["day_name"],
                "is_monday": day_flags["is_monday"],
                "is_friday": day_flags["is_friday"],
                "current_price": round(current_price, 2),
                "eqh_nearby": eqh_nearby,
                "eql_nearby": eql_nearby,
                "eqh_level": eqh_level,
                "eql_level": eql_level,
                "layer1": layer1,
                "layer2": layer2,
                "layer3": layer3,
                "layer4": layer4,
                "entry": entry,
                "score": score,
                "entry_ready": entry_ready,
                "entry_confidence": entry_confidence,
                "direction": direction,
                "direction_confidence": direction_confidence,
                "size_label": layer1.get("size_label", "normal"),
                "vix_bucket": layer1.get("vix_bucket", "normal"),
                "decision": decision,
                "grade": grade,
                "caution_flags": [],
                "block_reason": None,
                "is_news_day": news_data.get("is_news_day", False),
                "next_news_event": news_data.get("next_event"),
                "is_ranging": is_ranging,
                "range_condition": range_condition,
                "has_liquidity_sweep": has_sweep,
                "_day_flags": day_flags,
            }

            result = self.apply_all_filters(result, news_data)

            logger.info(
                "Checklist: score=%d dir=%s decision=%s grade=%s price=%.2f ranging=%s sweep=%s",
                score, direction, result["decision"], result["grade"],
                current_price, is_ranging, has_sweep,
            )
            return result

        except Exception as exc:
            logger.error("run_full_checklist failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "checklist", str(exc))
            return self._fallback_result(current_price)

    # ─── direction derivation ────────────────────────────────

    def _derive_direction(
        self, layer1: Dict, layer2: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Derive trade direction from macro bias, H4 structure, and VIX.

        Priority (highest → lowest confidence):
          1. macro_bias + structure_bias agree
          2. VIX direction + structure_bias agree
          3. EMA bias + macro OR VIX agree            (fallback when structure unclear)
          4. Structure alone when macro conflicts      (ICT: structure > macro)

        Returns (direction, confidence) — no side effects on instance state.
        """
        macro_bias = layer1.get("bias")
        struct_bias = layer2.get("structure_bias")
        vix_dir = layer1.get("vix_direction_bias")

        # Priority 1 — full agreement
        if macro_bias == "SHORT" and struct_bias == "short":
            return "SHORT", "high"
        if macro_bias == "LONG" and struct_bias == "long":
            return "LONG", "high"

        # Priority 2 — VIX direction + structure
        if vix_dir == "SELL_BIAS" and struct_bias == "short":
            return "SHORT", "high"
        if vix_dir == "BUY_BIAS" and struct_bias == "long":
            return "LONG", "high"

        # Priority 3 — EMA + macro/VIX when structure unclear
        if struct_bias == "unclear":
            ema_bias = layer2.get("ema_bias")
            if ema_bias == "bearish" and (macro_bias == "SHORT" or vix_dir == "SELL_BIAS"):
                return "SHORT", "medium"
            if ema_bias == "bullish" and (macro_bias == "LONG" or vix_dir == "BUY_BIAS"):
                return "LONG", "medium"

        # Priority 4 — Structure alone (macro conflicts or unavailable)
        # ICT principle: H4 structure is king. Macro modulates sizing, not direction.
        # When VIX short_only bucket aligns with bearish structure, confidence is higher.
        if struct_bias in ("long", "short"):
            short_only = layer1.get("short_only", False)
            if struct_bias == "short":
                confidence = "medium" if short_only else "reduced"
                logger.info(
                    "Direction from structure alone: SHORT (macro=%s vix=%s confidence=%s)",
                    macro_bias, vix_dir, confidence,
                )
                return "SHORT", confidence
            if struct_bias == "long" and not short_only:
                logger.info(
                    "Direction from structure alone: LONG (macro=%s vix=%s confidence=%s)",
                    macro_bias, vix_dir, "reduced",
                )
                return "LONG", "reduced"

        return None, None

    # ─── filters ─────────────────────────────────────────────

    def apply_all_filters(self, result: Dict[str, Any], news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all safety filters — VIX hard stop, news blocking, range/sweep, day-of-week cautions."""
        try:
            vix_val = result["layer1"].get("vix_value") or 0
            if vix_val >= 30:
                result["decision"] = "HARD_STOP"
                result["block_reason"] = "VIX >= 30"
                result["grade"] = None
                return result

            if news_data.get("pre_news_blocked"):
                event = news_data.get("pre_news_event", "upcoming event")
                result["decision"] = "NO_TRADE"
                result["block_reason"] = f"Pre-news: {event}"
                return result

            day_flags = result.get("_day_flags") or self.get_day_flags()
            if day_flags.get("monday_caution"):
                result["caution_flags"].append("MONDAY OPEN CAUTION")
            if day_flags.get("friday_caution"):
                result["caution_flags"].append("FRIDAY PM CAUTION")

            if news_data.get("is_news_day"):
                result["caution_flags"].append("NEWS DAY — reduced size")
                if news_data.get("post_news_caution"):
                    if result.get("grade") == "B":
                        result["decision"] = "NO_TRADE"
                        result["block_reason"] = "Post-news: Grade A only"

            layer1 = result.get("layer1", {})
            if layer1.get("short_only") and result.get("direction") == "LONG":
                result["decision"] = "NO_TRADE"
                result["block_reason"] = "VIX 25-30 short only"

            if result.get("direction_confidence") == "reduced":
                result["caution_flags"].append("STRUCT-ONLY DIRECTION — macro conflicts")

            # ── Range / consolidation cautions ──
            range_cond = result.get("range_condition", {})
            is_ranging = range_cond.get("is_ranging", False)
            range_strength = range_cond.get("range_strength", "none")

            if is_ranging:
                adx_val = range_cond.get("adx_value", "?")
                if range_strength == "strong":
                    result["caution_flags"].append(
                        f"STRONG RANGE — ADX {adx_val}, ATR compressing, sweep required"
                    )
                else:
                    result["caution_flags"].append(
                        f"MILD RANGE — ADX {adx_val}, reduced conviction"
                    )

            if not result.get("has_liquidity_sweep", False) and result.get("entry_ready") and not is_ranging:
                result["caution_flags"].append("NO LIQUIDITY SWEEP — lower probability")

            return result

        except Exception as exc:
            logger.error("apply_all_filters failed: %s", exc, exc_info=True)
            return result

    # ─── alert gating ────────────────────────────────────────

    def should_send_alert(self, result: Dict[str, Any]) -> bool:
        """Decide whether this result warrants a Telegram alert."""
        try:
            decision = result.get("decision")

            if decision == "HARD_STOP":
                if self._hard_stop_active:
                    return False
                self._hard_stop_active = True
                return True

            if self._hard_stop_active:
                self._hard_stop_active = False
                logger.info("VIX dropped below 30 — HARD_STOP cleared")

            if decision == "WAIT" and result.get("score", 0) >= GRADE_B_SCORE:
                if not result.get("direction"):
                    return False
                if result.get("block_reason"):
                    return False
                if self.is_duplicate_signal(result):
                    return False
                return True

            if decision not in ("FULL_SEND", "HALF_SIZE"):
                return False

            if not result.get("entry_ready"):
                return False

            if result.get("block_reason"):
                return False

            if self.is_duplicate_signal(result):
                return False

            return True

        except Exception as exc:
            logger.error("should_send_alert failed: %s", exc, exc_info=True)
            return False

    def is_duplicate_signal(self, result: Dict[str, Any]) -> bool:
        """Check the database for a recent identical signal to avoid spam."""
        try:
            direction = result.get("direction")
            price = result.get("current_price", 0)
            if not direction:
                return False
            return self._signal_logger.is_duplicate(direction, price)
        except Exception as exc:
            logger.error("is_duplicate_signal failed: %s", exc, exc_info=True)
            return False

    # ─── fallbacks ───────────────────────────────────────────

    def _fallback_result(self, current_price: float) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sl_time": self.get_sl_time(),
            "session": self.get_current_session(),
            "day_name": "Unknown", "is_monday": False, "is_friday": False,
            "current_price": round(current_price, 2),
            "eqh_nearby": False, "eql_nearby": False,
            "eqh_level": None, "eql_level": None,
            "layer1": {}, "layer2": {}, "layer3": {}, "layer4": {},
            "entry": None, "score": 0, "entry_ready": False, "entry_confidence": None,
            "direction": None, "direction_confidence": None,
            "size_label": "normal", "vix_bucket": "normal",
            "decision": "NO_TRADE", "grade": None,
            "caution_flags": [], "block_reason": "checklist_error",
            "is_news_day": False, "next_news_event": None,
            "is_ranging": False, "range_condition": {},
            "has_liquidity_sweep": False,
        }

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
