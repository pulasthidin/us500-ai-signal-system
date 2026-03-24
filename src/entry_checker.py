"""
M5 Entry Check — only invoked when the checklist scores 3/4 or 4/4.

Required (hard gate):
  - M5 FVG in trade direction (unfilled / unmitigated)
  - R:R ≥ MIN_RR

Bonus (confidence boosters — affect grade, not entry readiness):
  - M5 BOS confirmed in trade direction
  - USTEC SMT agreement (both indices swinging same way)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import os

import numpy as np
import pandas as pd
from smartmoneyconcepts import smc

from config import (
    ATR_PERIOD,
    MIN_RR,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("entry")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/entry.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class EntryChecker:
    """Runs M5 entry conditions plus ATR-based SL/TP calculation.

    Hard gate: FVG + R:R.  Bonus: BOS + SMT (affect confidence/grade).
    """

    def __init__(self, ctrader_connection, us500_id: int, ustec_id: int, alert_bot=None) -> None:
        self._ctrader = ctrader_connection
        self._us500_id = us500_id
        self._ustec_id = ustec_id
        self._alert_bot = alert_bot

    # ─── bar fetching ────────────────────────────────────────

    def get_m5_bars(self, count: int = 50) -> Optional[pd.DataFrame]:
        """Fetch M5 bars for US500."""
        try:
            return self._ctrader.fetch_bars(self._us500_id, "M5", count)
        except Exception as exc:
            logger.error("get_m5_bars failed: %s", exc, exc_info=True)
            return None

    def get_m5_ustec_bars(self, count: int = 50) -> Optional[pd.DataFrame]:
        """Fetch M5 bars for USTEC (NAS100)."""
        try:
            return self._ctrader.fetch_bars(self._ustec_id, "M5", count)
        except Exception as exc:
            logger.error("get_m5_ustec_bars failed: %s", exc, exc_info=True)
            return None

    # ─── FVG detection ───────────────────────────────────────

    def detect_m5_fvg(self, df: pd.DataFrame, direction: str, atr: float = 0.0) -> Dict[str, Any]:
        """
        Find the most recent unmitigated Fair Value Gap in the trade direction.

        Two-pass approach:
          1. Try the SMC library (strict wick-based FVG).
          2. Fallback to body-based imbalance detection — matches what ICT
             traders actually see on charts.  A body-based FVG exists when the
             *body* of bar[i-1] doesn't overlap with the *body* of bar[i+1],
             leaving a gap that the impulse candle (bar[i]) fills.

        Mitigation: an FVG is considered mitigated only when price has closed
        beyond the 50% midpoint of the gap (not just wicked the edge).
        """
        try:
            target_dir = "bearish" if direction.upper() == "SHORT" else "bullish"

            # ── Pass 1: SMC library (strict) ──
            result = self._detect_fvg_smc(df, target_dir, atr)
            if result["present"]:
                logger.debug("FVG found via SMC library: %s", result)
                return result

            # ── Pass 2: body-based imbalance (practical) ──
            result = self._detect_fvg_body(df, target_dir, atr)
            if result["present"]:
                logger.debug("FVG found via body-based detection: %s", result)
            return result

        except Exception as exc:
            logger.error("detect_m5_fvg failed: %s", exc, exc_info=True)
            return self._empty_fvg()

    def _detect_fvg_smc(self, df: pd.DataFrame, target_dir: str, atr: float) -> Dict[str, Any]:
        """Detect FVG using the smartmoneyconcepts library (strict wick-based)."""
        try:
            min_fvg_size = 0.25 * atr if atr > 0 else 0.0
            fvg_data = smc.fvg(df, join_consecutive=True)

            fvg_col = None
            for col in fvg_data.columns:
                if "fvg" in col.lower():
                    fvg_col = col
                    break
            if fvg_col is None:
                return self._empty_fvg()

            fvg_series = fvg_data[fvg_col].dropna()
            if fvg_series.empty:
                return self._empty_fvg()

            top_col = bottom_col = mitigated_col = None
            for col in fvg_data.columns:
                cl = col.lower()
                if "top" in cl:
                    top_col = col
                elif "bottom" in cl or "bot" in cl:
                    bottom_col = col
                elif "mitigat" in cl:
                    mitigated_col = col

            for idx in reversed(fvg_series.index.tolist()):
                val = fvg_series.loc[idx]
                if isinstance(val, str):
                    is_match = target_dir in val.lower()
                elif isinstance(val, (int, float)):
                    is_match = (val < 0 and target_dir == "bearish") or (val > 0 and target_dir == "bullish")
                else:
                    continue
                if not is_match:
                    continue

                if mitigated_col and idx in fvg_data.index:
                    mit_val = fvg_data.loc[idx, mitigated_col]
                    if mit_val and str(mit_val).lower() not in ("0", "false", "nan", ""):
                        continue

                top = float(fvg_data.loc[idx, top_col]) if top_col else None
                bottom = float(fvg_data.loc[idx, bottom_col]) if bottom_col else None
                if top is None or bottom is None:
                    continue
                if min_fvg_size > 0 and abs(top - bottom) < min_fvg_size:
                    continue

                return self._fvg_result(df, idx, top, bottom, target_dir)

            return self._empty_fvg()
        except Exception as exc:
            logger.debug("SMC FVG detection failed: %s", exc)
            return self._empty_fvg()

    def _detect_fvg_body(self, df: pd.DataFrame, target_dir: str, atr: float) -> Dict[str, Any]:
        """
        Detect FVG using candle bodies — more lenient than wick-based.

        Bearish body-FVG: the body-bottom of bar[i-1] > body-top of bar[i+1]
        (i.e. the bodies don't overlap, leaving a gap that bar[i] fills).

        Mitigation: price must *close* beyond 50% of the gap, not just wick it.
        """
        try:
            min_fvg_size = 0.15 * atr if atr > 0 else 0.0
            highs = df["high"].values
            lows = df["low"].values
            opens = df["open"].values
            closes = df["close"].values

            # Scan from most recent backwards (skip last bar — need i+1)
            for i in range(len(df) - 2, 1, -1):
                if target_dir == "bearish":
                    # Impulse candle must be bearish
                    if closes[i] >= opens[i]:
                        continue
                    # Body-based gap: body-bottom of bar[i-1] > body-top of bar[i+1]
                    prev_body_bottom = min(opens[i - 1], closes[i - 1])
                    next_body_top = max(opens[i + 1], closes[i + 1])
                    if prev_body_bottom <= next_body_top:
                        continue
                    gap_top = prev_body_bottom
                    gap_bottom = next_body_top
                else:
                    # Impulse candle must be bullish
                    if closes[i] <= opens[i]:
                        continue
                    # Body-based gap: body-top of bar[i-1] < body-bottom of bar[i+1]
                    prev_body_top = max(opens[i - 1], closes[i - 1])
                    next_body_bottom = min(opens[i + 1], closes[i + 1])
                    if next_body_bottom <= prev_body_top:
                        continue
                    gap_top = next_body_bottom
                    gap_bottom = prev_body_top

                gap_size = abs(gap_top - gap_bottom)
                if gap_size < min_fvg_size:
                    continue

                # Check mitigation: has price closed beyond the 50% midpoint?
                midpoint = (gap_top + gap_bottom) / 2
                mitigated = False
                for j in range(i + 2, len(df)):
                    if target_dir == "bearish" and closes[j] > midpoint:
                        mitigated = True
                        break
                    elif target_dir == "bullish" and closes[j] < midpoint:
                        mitigated = True
                        break

                if mitigated:
                    continue

                return self._fvg_result(df, i, gap_top, gap_bottom, target_dir)

            return self._empty_fvg()
        except Exception as exc:
            logger.debug("Body FVG detection failed: %s", exc)
            return self._empty_fvg()

    def _fvg_result(self, df: pd.DataFrame, idx: int, top: float, bottom: float, direction: str) -> Dict[str, Any]:
        """Build an FVG result dict."""
        midpoint = (top + bottom) / 2
        bars_ago = max(0, len(df) - 1 - int(idx))
        return {
            "present": True,
            "top": round(top, 2),
            "bottom": round(bottom, 2),
            "midpoint": round(midpoint, 2),
            "age_bars": int(bars_ago),
            "direction": direction,
        }

    # ─── M5 BOS ──────────────────────────────────────────────

    def detect_m5_bos(self, df: pd.DataFrame, direction: str) -> bool:
        """Check if the most recent M5 BOS matches the trade direction."""
        try:
            swing_hl = smc.swing_highs_lows(df, swing_length=5)
            bos_choch = smc.bos_choch(df, swing_hl, close_break=True)

            bos_col = None
            for col in bos_choch.columns:
                if "bos" in col.lower():
                    bos_col = col
                    break

            if bos_col is None:
                return False

            bos_series = bos_choch[bos_col].dropna()
            if bos_series.empty:
                return False

            last_val = bos_series.iloc[-1]
            target = "bullish" if direction.upper() == "LONG" else "bearish"

            if isinstance(last_val, str):
                return target in last_val.lower()
            elif isinstance(last_val, (int, float)):
                return (last_val > 0 and target == "bullish") or (last_val < 0 and target == "bearish")
            return False

        except Exception as exc:
            logger.error("detect_m5_bos failed: %s", exc, exc_info=True)
            return False

    # ─── USTEC SMT ───────────────────────────────────────────

    def check_ustec_smt(
        self, us500_df: pd.DataFrame, ustec_df: pd.DataFrame, direction: str
    ) -> Dict[str, Any]:
        """
        Smart Money Technique inter-market check.
        Both indices making the same directional swing = agrees.
        """
        try:
            if us500_df is None or ustec_df is None or us500_df.empty or ustec_df.empty:
                return {"agrees": False, "us500_swing": "unknown", "ustec_swing": "unknown", "divergence_type": None}

            lookback = min(15, len(us500_df), len(ustec_df))
            us_recent = us500_df.tail(lookback)
            ut_recent = ustec_df.tail(lookback)

            us_high_idx = us_recent["high"].idxmax()
            us_low_idx = us_recent["low"].idxmin()
            ut_high_idx = ut_recent["high"].idxmax()
            ut_low_idx = ut_recent["low"].idxmin()

            us_swing = "higher" if us_high_idx > us_low_idx else "lower"
            ut_swing = "higher" if ut_high_idx > ut_low_idx else "lower"

            if direction.upper() == "SHORT":
                mid = lookback // 2
                us_h1 = us_recent.iloc[:mid]["high"].max()
                us_h2 = us_recent.iloc[mid:]["high"].max()
                ut_h1 = ut_recent.iloc[:mid]["high"].max()
                ut_h2 = ut_recent.iloc[mid:]["high"].max()
                agrees = us_h2 < us_h1 and ut_h2 < ut_h1
            elif direction.upper() == "LONG":
                mid = lookback // 2
                us_l1 = us_recent.iloc[:mid]["low"].min()
                us_l2 = us_recent.iloc[mid:]["low"].min()
                ut_l1 = ut_recent.iloc[:mid]["low"].min()
                ut_l2 = ut_recent.iloc[mid:]["low"].min()
                agrees = us_l2 > us_l1 and ut_l2 > ut_l1
            else:
                agrees = us_swing == ut_swing

            divergence_type = None
            if us_swing != ut_swing:
                divergence_type = "SMT_divergence"

            return {
                "agrees": agrees,
                "us500_swing": us_swing,
                "ustec_swing": ut_swing,
                "divergence_type": divergence_type,
            }

        except Exception as exc:
            logger.error("check_ustec_smt failed: %s", exc, exc_info=True)
            return {"agrees": False, "us500_swing": "unknown", "ustec_swing": "unknown", "divergence_type": None}

    # ─── ATR ─────────────────────────────────────────────────

    def calculate_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """Average True Range using Wilder's exponential smoothing."""
        try:
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values

            tr = np.zeros(len(df))
            tr[0] = high[0] - low[0]
            for i in range(1, len(df)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )

            if len(tr) < period:
                return float(np.mean(tr))
            atr_val = float(np.mean(tr[:period]))
            for i in range(period, len(tr)):
                atr_val = (atr_val * (period - 1) + tr[i]) / period
            return float(atr_val)

        except Exception as exc:
            logger.error("calculate_atr failed: %s", exc, exc_info=True)
            return 0.0

    # ─── R:R calculation ─────────────────────────────────────

    def calculate_rr(
        self, current_price: float, direction: str, atr: float, fvg_edge: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute SL/TP prices and R:R ratio.

        When *fvg_edge* (the invalidation level from the FVG) is provided, SL is
        placed just beyond it with a small ATR buffer (0.3 * ATR) instead of the
        fixed ATR multiplier.  This produces a variable R:R that actually filters.
        """
        try:
            tp_distance = TP_ATR_MULTIPLIER * atr
            atr_buffer = 0.3 * atr

            if fvg_edge is not None and atr > 0:
                if direction.upper() == "SHORT":
                    sl_price = fvg_edge + atr_buffer
                    tp_price = current_price - tp_distance
                else:
                    sl_price = fvg_edge - atr_buffer
                    tp_price = current_price + tp_distance
                sl_distance = abs(current_price - sl_price)
            else:
                sl_distance = SL_ATR_MULTIPLIER * atr
                if direction.upper() == "SHORT":
                    sl_price = current_price + sl_distance
                    tp_price = current_price - tp_distance
                else:
                    sl_price = current_price - sl_distance
                    tp_price = current_price + tp_distance

            rr = (tp_distance / sl_distance) if sl_distance > 0 else 0.0

            return {
                "sl_price": round(sl_price, 2),
                "tp_price": round(tp_price, 2),
                "rr": round(rr, 2),
                "sl_points": round(sl_distance, 2),
                "tp_points": round(tp_distance, 2),
                "atr_value": round(atr, 2),
            }

        except Exception as exc:
            logger.error("calculate_rr failed: %s", exc, exc_info=True)
            return {"sl_price": 0, "tp_price": 0, "rr": 0, "sl_points": 0, "tp_points": 0, "atr_value": 0}

    # ─── composite entry check ───────────────────────────────

    def get_entry_result(self, direction: str, current_price: float) -> Dict[str, Any]:
        """
        Run M5 entry conditions.

        Hard gate (required):
          1. M5 FVG in trade direction (unfilled / unmitigated)
          2. R:R ≥ MIN_RR

        Bonus (confidence — affects grade, not entry readiness):
          3. M5 BOS confirmed in trade direction
          4. USTEC SMT agreement

        entry_confidence:
          "full"     — FVG + RR + BOS + SMT  (all 4)
          "high"     — FVG + RR + one of BOS/SMT
          "base"     — FVG + RR only (minimum entry)
        """
        try:
            if not direction or direction.upper() not in ("LONG", "SHORT"):
                return self._empty_entry()

            us500_df = self.get_m5_bars(50)
            ustec_df = self.get_m5_ustec_bars(50)

            if us500_df is None or us500_df.empty:
                return self._empty_entry()

            m15_df = self._ctrader.fetch_bars(self._us500_id, "M15", 50)
            atr = self.calculate_atr(m15_df if m15_df is not None and not m15_df.empty else us500_df)

            fvg = self.detect_m5_fvg(us500_df, direction, atr=atr)
            m5_bos = self.detect_m5_bos(us500_df, direction)
            smt = self.check_ustec_smt(us500_df, ustec_df, direction)

            fvg_edge = None
            if fvg.get("present"):
                fvg_edge = fvg.get("bottom") if direction.upper() == "LONG" else fvg.get("top")
            rr_info = self.calculate_rr(current_price, direction, atr, fvg_edge=fvg_edge)
            rr_valid = rr_info["rr"] >= MIN_RR

            # ── Hard gate: FVG + R:R ──
            entry_ready = fvg["present"] and rr_valid

            # ── Confidence tier based on bonus conditions ──
            bonus_count = sum([m5_bos, smt["agrees"]])
            if entry_ready and bonus_count == 2:
                entry_confidence = "full"      # all conditions met
            elif entry_ready and bonus_count == 1:
                entry_confidence = "high"      # FVG + RR + one bonus
            elif entry_ready:
                entry_confidence = "base"      # FVG + RR only
            else:
                entry_confidence = None

            logger.info(
                "Entry check: FVG=%s BOS=%s SMT=%s RR=%.2f(valid=%s) → %s (confidence=%s)",
                fvg["present"], m5_bos, smt["agrees"],
                rr_info["rr"], rr_valid,
                "READY" if entry_ready else "NOT_READY",
                entry_confidence,
            )

            return {
                "fvg_present": fvg["present"],
                "fvg_details": fvg,
                "m5_bos_confirmed": m5_bos,
                "ustec_agrees": smt["agrees"],
                "ustec_details": smt,
                "rr_valid": rr_valid,
                "entry_ready": entry_ready,
                "entry_confidence": entry_confidence,
                "sl_price": rr_info["sl_price"],
                "tp_price": rr_info["tp_price"],
                "rr": rr_info["rr"],
                "sl_points": rr_info["sl_points"],
                "tp_points": rr_info["tp_points"],
                "atr": rr_info["atr_value"],
            }

        except Exception as exc:
            logger.error("get_entry_result failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "entry_check", str(exc))
            return self._empty_entry()

    # ─── fallbacks ───────────────────────────────────────────

    def _empty_fvg(self) -> Dict[str, Any]:
        return {"present": False, "top": None, "bottom": None, "midpoint": None, "age_bars": 0, "direction": None}

    def _empty_entry(self) -> Dict[str, Any]:
        return {
            "fvg_present": False, "fvg_details": self._empty_fvg(),
            "m5_bos_confirmed": False, "ustec_agrees": False,
            "ustec_details": {"agrees": False, "us500_swing": "unknown", "ustec_swing": "unknown", "divergence_type": None},
            "rr_valid": False, "entry_ready": False, "entry_confidence": None,
            "sl_price": 0, "tp_price": 0, "rr": 0, "sl_points": 0, "tp_points": 0, "atr": 0,
        }

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
