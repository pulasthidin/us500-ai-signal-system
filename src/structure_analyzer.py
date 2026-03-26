"""
Layer 2 — H4 Structure Analysis.
EMA 200/50 position, BOS/ChoCH via smartmoneyconcepts, simplified Wyckoff.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import os

import numpy as np
import pandas as pd
import pandas_ta as ta
from smartmoneyconcepts import smc

from config import (
    ADX_PERIOD,
    ADX_RANGE_THRESHOLD,
    ATR_COMPRESSION_LOOKBACK,
    ATR_COMPRESSION_RATIO,
    H4_BOS_SWING_LENGTH,
    RANGE_LOOKBACK_BARS,
    US500_SYMBOL,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("structure")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/structure.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class StructureAnalyzer:
    """Evaluates H4 market structure for Layer 2 scoring."""

    def __init__(self, ctrader_connection, us500_id: int, alert_bot=None) -> None:
        self._ctrader = ctrader_connection
        self._us500_id = us500_id
        self._alert_bot = alert_bot

    # ─── data fetching ───────────────────────────────────────

    def get_h4_bars(self) -> Optional[pd.DataFrame]:
        """Fetch 200 H4 bars from cTrader for US500."""
        try:
            df = self._ctrader.fetch_bars(self._us500_id, "H4", 200)
            if df is None or df.empty:
                logger.warning("No H4 bars returned")
                return None
            return df
        except Exception as exc:
            logger.error("get_h4_bars failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "h4_bars", str(exc))
            return None

    def get_h1_bars(self, count: int = 48) -> Optional[pd.DataFrame]:
        """Fetch H1 bars for range detection."""
        try:
            df = self._ctrader.fetch_bars(self._us500_id, "H1", count)
            if df is None or df.empty:
                logger.warning("No H1 bars returned for range detection")
                return None
            return df
        except Exception as exc:
            logger.error("get_h1_bars failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "h1_bars", str(exc))
            return None

    # ─── range / consolidation detection ─────────────────────

    def detect_range_condition(self) -> Dict[str, Any]:
        """
        Detect whether the market is in a consolidation / ranging state.

        Uses H1 bars with two complementary methods:
          1. ADX < threshold → directional strength is weak
          2. ATR compression → current ATR shrinking vs rolling average

        When both confirm, the market is definitively ranging.
        When only one confirms, it's a "mild range" (caution, not hard block).

        Also computes the range boundaries (H1 high/low of recent bars) so the
        entry checker can place SL beyond the range when appropriate.
        """
        try:
            df = self.get_h1_bars(max(ATR_COMPRESSION_LOOKBACK * 2, 48))
            if df is None or len(df) < ADX_PERIOD + 5:
                return self._empty_range()

            adx_series = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)
            adx_val = None
            if adx_series is not None and not adx_series.empty:
                adx_col = [c for c in adx_series.columns if "ADX" in c.upper() and "DM" not in c.upper()]
                if adx_col:
                    adx_raw = adx_series[adx_col[0]].dropna()
                    if not adx_raw.empty:
                        adx_val = float(adx_raw.iloc[-1])

            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
            atr_compressing = False
            atr_ratio = 1.0
            if atr_series is not None and not atr_series.dropna().empty:
                atr_values = atr_series.dropna()
                current_atr = float(atr_values.iloc[-1])
                lookback = min(ATR_COMPRESSION_LOOKBACK, len(atr_values))
                avg_atr = float(atr_values.iloc[-lookback:].mean())
                if avg_atr > 0:
                    atr_ratio = current_atr / avg_atr
                    atr_compressing = atr_ratio < ATR_COMPRESSION_RATIO

            adx_ranging = adx_val is not None and adx_val < ADX_RANGE_THRESHOLD

            if adx_ranging and atr_compressing:
                is_ranging = True
                range_strength = "strong"
            elif adx_ranging or atr_compressing:
                is_ranging = True
                range_strength = "mild"
            else:
                is_ranging = False
                range_strength = "none"

            recent = df.tail(RANGE_LOOKBACK_BARS)
            range_high = float(recent["high"].max())
            range_low = float(recent["low"].min())

            logger.info(
                "Range check: ADX=%.1f(%s) ATR_ratio=%.2f(%s) -> %s (%s) range=[%.1f, %.1f]",
                adx_val or 0, "ranging" if adx_ranging else "trending",
                atr_ratio, "compressing" if atr_compressing else "normal",
                "RANGING" if is_ranging else "TRENDING", range_strength,
                range_low, range_high,
            )

            return {
                "is_ranging": is_ranging,
                "range_strength": range_strength,
                "adx_value": round(adx_val, 2) if adx_val is not None else None,
                "adx_ranging": adx_ranging,
                "atr_compressing": atr_compressing,
                "atr_ratio": round(atr_ratio, 3),
                "range_high": round(range_high, 2),
                "range_low": round(range_low, 2),
            }

        except Exception as exc:
            logger.error("detect_range_condition failed: %s", exc, exc_info=True)
            return self._empty_range()

    @staticmethod
    def _empty_range() -> Dict[str, Any]:
        return {
            "is_ranging": False, "range_strength": "none",
            "adx_value": None, "adx_ranging": False,
            "atr_compressing": False, "atr_ratio": 1.0,
            "range_high": None, "range_low": None,
        }

    # ─── EMAs ────────────────────────────────────────────────

    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA 50 and EMA 200 columns to the DataFrame."""
        try:
            df = df.copy()
            df["ema50"] = ta.ema(df["close"], length=50)
            df["ema200"] = ta.ema(df["close"], length=200)
            return df
        except Exception as exc:
            logger.error("calculate_emas failed: %s", exc, exc_info=True)
            clean = df.copy()
            for col in ("ema50", "ema200"):
                if col in clean.columns:
                    clean.drop(columns=[col], inplace=True)
            return clean

    def get_price_vs_emas(self, current_price: float, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine price position relative to EMA 200 and EMA 50.
        Bullish = above both, Bearish = below both, otherwise unclear.
        """
        try:
            ema200 = float(df["ema200"].dropna().iloc[-1]) if "ema200" in df.columns and not df["ema200"].dropna().empty else None
            ema50 = float(df["ema50"].dropna().iloc[-1]) if "ema50" in df.columns and not df["ema50"].dropna().empty else None

            above_200 = current_price > ema200 if ema200 is not None else None
            above_50 = current_price > ema50 if ema50 is not None else None

            if ema200 is not None:
                if above_200 is True and above_50 is True:
                    ema_bias = "bullish"
                elif above_200 is False and above_50 is False:
                    ema_bias = "bearish"
                else:
                    ema_bias = "unclear"
            elif ema50 is not None:
                ema_bias = "bullish" if above_50 else "bearish"
            else:
                ema_bias = "unclear"

            return {
                "above_ema200": above_200,
                "above_ema50": above_50,
                "ema200_value": round(ema200, 2) if ema200 is not None else None,
                "ema50_value": round(ema50, 2) if ema50 is not None else None,
                "ema_bias": ema_bias,
            }
        except Exception as exc:
            logger.error("get_price_vs_emas failed: %s", exc, exc_info=True)
            return {"above_ema200": None, "above_ema50": None, "ema200_value": None, "ema50_value": None, "ema_bias": "unclear"}

    # ─── shared SMC compute ──────────────────────────────────

    def _compute_bos_choch(self, df: pd.DataFrame):
        """
        Run smc.swing_highs_lows + smc.bos_choch once and return the result.
        Both detect_bos and detect_choch read from the same DataFrame so this
        avoids computing the same 200-bar SMC result twice per cycle.
        """
        swing_hl = smc.swing_highs_lows(df, swing_length=H4_BOS_SWING_LENGTH)
        return smc.bos_choch(df, swing_hl, close_break=True)

    @staticmethod
    def _extract_last_signal(series, df: pd.DataFrame) -> tuple:
        """Return (direction, iloc_pos, bars_ago) for the last non-null entry in *series*."""
        if series.empty:
            return None, None, 0
        last_pos = len(series) - 1
        last_idx = series.index[last_pos]
        last_val = series.iloc[last_pos]
        try:
            iloc_pos = df.index.get_loc(last_idx) if last_idx in df.index else last_pos
        except Exception:
            iloc_pos = last_pos
        bars_ago = max(0, len(df) - 1 - int(iloc_pos))
        direction = None
        if isinstance(last_val, str):
            low = last_val.lower()
            if "bull" in low:
                direction = "bullish"
            elif "bear" in low:
                direction = "bearish"
        elif isinstance(last_val, (int, float)):
            if last_val > 0:
                direction = "bullish"
            elif last_val < 0:
                direction = "bearish"
        return direction, last_idx, int(bars_ago)

    # ─── BOS ─────────────────────────────────────────────────

    def detect_bos(self, df: pd.DataFrame, bos_choch=None) -> Dict[str, Any]:
        """
        Find the most recent Break of Structure.
        Pass *bos_choch* (from _compute_bos_choch) to avoid recomputing.
        """
        try:
            if bos_choch is None:
                bos_choch = self._compute_bos_choch(df)

            bos_col = "BOS" if "BOS" in bos_choch.columns else None
            if bos_col is None:
                for col in bos_choch.columns:
                    if "bos" in col.lower():
                        bos_col = col
                        break

            if bos_col is None:
                return {"direction": None, "level": None, "bars_ago": 0}

            bos_series = bos_choch[bos_col].dropna()
            direction, last_idx, bars_ago = self._extract_last_signal(bos_series, df)
            if direction is None:
                return {"direction": None, "level": None, "bars_ago": 0}

            level_col = "Level" if "Level" in bos_choch.columns else None
            level = None
            if level_col and last_idx in bos_choch.index:
                raw_level = bos_choch.loc[last_idx, level_col]
                if pd.notna(raw_level):
                    level = float(raw_level)

            return {"direction": direction, "level": round(level, 2) if level is not None else None, "bars_ago": bars_ago}

        except Exception as exc:
            logger.error("detect_bos failed: %s", exc, exc_info=True)
            return {"direction": None, "level": None, "bars_ago": 0}

    # ─── ChoCH ───────────────────────────────────────────────

    def detect_choch(self, df: pd.DataFrame, bos_choch=None) -> Dict[str, Any]:
        """
        Find the most recent Change of Character.
        Pass *bos_choch* (from _compute_bos_choch) to avoid recomputing.
        """
        try:
            if bos_choch is None:
                bos_choch = self._compute_bos_choch(df)

            choch_col = None
            for col in bos_choch.columns:
                if "choch" in col.lower() or "CHoCH" in col:
                    choch_col = col
                    break

            if choch_col is None:
                return {"direction": None, "bars_ago": 0}

            choch_series = bos_choch[choch_col].dropna()
            direction, _, bars_ago = self._extract_last_signal(choch_series, df)
            return {"direction": direction, "bars_ago": bars_ago}

        except Exception as exc:
            logger.error("detect_choch failed: %s", exc, exc_info=True)
            return {"direction": None, "bars_ago": 0}

    # ─── Wyckoff (simplified) ────────────────────────────────

    def detect_wyckoff(self, df: pd.DataFrame) -> str:
        """
        Simplified Wyckoff phase detection on the last 20 H4 bars.
        Markup = trending up + increasing volume, Markdown = opposite.
        """
        try:
            recent = df.tail(20).copy()
            if len(recent) < 10:
                return "unclear"

            price_change = recent["close"].iloc[-1] - recent["close"].iloc[0]
            vol_first_half = recent["volume"].iloc[:10].mean()
            vol_second_half = recent["volume"].iloc[10:].mean()
            volume_increasing = vol_second_half > vol_first_half * 1.05

            if price_change > 0 and volume_increasing:
                return "markup"
            elif price_change < 0 and volume_increasing:
                return "markdown"
            return "unclear"

        except Exception as exc:
            logger.error("detect_wyckoff failed: %s", exc, exc_info=True)
            return "unclear"

    # ─── Order Blocks (Pine Script match) ──────────────────

    def detect_order_blocks(self, df: pd.DataFrame, bos_result: Dict, current_price: float) -> Dict[str, Any]:
        """
        Detect Order Blocks at the origin of BOS/ChoCH moves.
        Bullish OB = last bearish candle before a bullish BOS.
        Bearish OB = last bullish candle before a bearish BOS.
        Filtered by ATR: OB height must be >= 0.5 * ATR(200).
        Mitigated when price closes through the OB.
        Returns up to 5 nearest unmitigated OBs per direction.
        """
        try:
            import pandas_ta as pta
            atr_series = pta.atr(df["high"], df["low"], df["close"], length=200)
            atr_val = float(atr_series.dropna().iloc[-1]) if atr_series is not None and not atr_series.dropna().empty else 0.0
            min_ob_height = 0.5 * atr_val

            swing_hl = smc.swing_highs_lows(df, swing_length=H4_BOS_SWING_LENGTH)
            bos_choch = smc.bos_choch(df, swing_hl, close_break=True)

            bullish_obs = []
            bearish_obs = []

            signal_cols = [c for c in bos_choch.columns if "bos" in c.lower() or "choch" in c.lower()]
            if not signal_cols:
                return self._empty_obs(current_price)

            for sig_col in signal_cols:
                for idx in bos_choch[sig_col].dropna().index:
                    val = bos_choch.loc[idx, sig_col]
                    is_bullish_break = (isinstance(val, str) and "bull" in val.lower()) or (isinstance(val, (int, float)) and val > 0)

                    try:
                        iloc_idx = df.index.get_loc(idx) if idx in df.index else int(idx)
                    except Exception:
                        try:
                            iloc_idx = int(idx)
                        except Exception:
                            continue
                    search_start = max(0, iloc_idx - 20)
                    search_range = df.iloc[search_start:iloc_idx]
                    if search_range.empty:
                        continue

                    subsequent = df.iloc[iloc_idx + 1:]

                    if is_bullish_break:
                        bearish_candles = search_range[search_range["close"] < search_range["open"]]
                        if bearish_candles.empty:
                            continue
                        ob_bar = bearish_candles.iloc[-1]
                        ob_high = float(ob_bar["high"])
                        ob_low = float(ob_bar["low"])
                        if (ob_high - ob_low) < min_ob_height and min_ob_height > 0:
                            continue
                        if current_price < ob_low:
                            continue
                        if not subsequent.empty and (subsequent["close"] < ob_low).any():
                            continue
                        bullish_obs.append({"top": round(ob_high, 2), "bottom": round(ob_low, 2), "type": "bullish"})
                    else:
                        bullish_candles = search_range[search_range["close"] > search_range["open"]]
                        if bullish_candles.empty:
                            continue
                        ob_bar = bullish_candles.iloc[-1]
                        ob_high = float(ob_bar["high"])
                        ob_low = float(ob_bar["low"])
                        if (ob_high - ob_low) < min_ob_height and min_ob_height > 0:
                            continue
                        if current_price > ob_high:
                            continue
                        if not subsequent.empty and (subsequent["close"] > ob_high).any():
                            continue
                        bearish_obs.append({"top": round(ob_high, 2), "bottom": round(ob_low, 2), "type": "bearish"})

            bullish_obs = sorted(bullish_obs, key=lambda x: abs(current_price - x["top"]))[:5]
            bearish_obs = sorted(bearish_obs, key=lambda x: abs(current_price - x["bottom"]))[:5]

            from config import ZONE_THRESHOLD_POINTS
            ob_bull_nearby = any(abs(current_price - ob["top"]) <= ZONE_THRESHOLD_POINTS * 2 for ob in bullish_obs)
            ob_bear_nearby = any(abs(current_price - ob["bottom"]) <= ZONE_THRESHOLD_POINTS * 2 for ob in bearish_obs)

            return {
                "bullish_obs": bullish_obs,
                "bearish_obs": bearish_obs,
                "ob_bullish_nearby": ob_bull_nearby,
                "ob_bearish_nearby": ob_bear_nearby,
            }

        except Exception as exc:
            logger.error("detect_order_blocks failed: %s", exc, exc_info=True)
            return self._empty_obs(current_price)

    @staticmethod
    def _empty_obs(current_price: float = 0) -> Dict[str, Any]:
        return {"bullish_obs": [], "bearish_obs": [], "ob_bullish_nearby": False, "ob_bearish_nearby": False}

    # ─── Layer 2 composite ───────────────────────────────────

    def get_layer2_result(self, current_price: float) -> Dict[str, Any]:
        """
        Run full H4 structure analysis + H1 range detection.
        Layer passes when EMA bias + BOS direction agree.
        """
        try:
            df = self.get_h4_bars()
            if df is None or df.empty:
                return self._fallback()

            df = self.calculate_emas(df)
            ema_result = self.get_price_vs_emas(current_price, df)
            bos_choch_data = self._compute_bos_choch(df)
            bos_result = self.detect_bos(df, bos_choch=bos_choch_data)
            choch_result = self.detect_choch(df, bos_choch=bos_choch_data)
            wyckoff = self.detect_wyckoff(df)
            ob_result = self.detect_order_blocks(df, bos_result, current_price)
            range_condition = self.detect_range_condition()

            ema_bias = ema_result["ema_bias"]
            bos_dir = bos_result["direction"]
            choch_dir = choch_result["direction"]
            choch_recent = choch_result["bars_ago"] <= 10 if choch_dir else False

            if ema_bias == "bullish" and bos_dir == "bullish":
                structure_bias = "long"
                score = 1
            elif ema_bias == "bearish" and bos_dir == "bearish":
                structure_bias = "short"
                score = 1
            elif ema_bias == "bearish" and choch_dir == "bearish" and choch_recent:
                structure_bias = "short"
                score = 1
            elif ema_bias == "bullish" and choch_dir == "bullish" and choch_recent:
                structure_bias = "long"
                score = 1
            elif ema_bias == "bearish" and wyckoff == "markdown":
                structure_bias = "short"
                score = 1
            elif ema_bias == "bullish" and wyckoff == "markup":
                structure_bias = "long"
                score = 1
            else:
                structure_bias = "unclear"
                score = 0

            return {
                "above_ema200": ema_result["above_ema200"],
                "above_ema50": ema_result["above_ema50"],
                "ema200_value": ema_result["ema200_value"],
                "ema50_value": ema_result["ema50_value"],
                "ema_bias": ema_bias,
                "bos_direction": bos_dir,
                "bos_level": bos_result["level"],
                "bos_bars_ago": bos_result["bars_ago"],
                "choch_direction": choch_result["direction"],
                "choch_recent": choch_recent,
                "wyckoff": wyckoff,
                "structure_bias": structure_bias,
                "score_contribution": score,
                "ob_bullish_nearby": ob_result["ob_bullish_nearby"],
                "ob_bearish_nearby": ob_result["ob_bearish_nearby"],
                "bullish_obs": ob_result["bullish_obs"],
                "bearish_obs": ob_result["bearish_obs"],
                "range_condition": range_condition,
            }

        except Exception as exc:
            logger.error("get_layer2_result failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "structure_layer2", str(exc))
            return self._fallback()

    # ─── fallback ────────────────────────────────────────────

    def _fallback(self) -> Dict[str, Any]:
        return {
            "above_ema200": None, "above_ema50": None, "ema200_value": None,
            "ema50_value": None, "ema_bias": "unclear", "bos_direction": None,
            "bos_level": None, "bos_bars_ago": 0, "choch_direction": None,
            "choch_recent": False, "wyckoff": "unclear", "structure_bias": "unclear",
            "score_contribution": 0,
            "ob_bullish_nearby": False, "ob_bearish_nearby": False,
            "bullish_obs": [], "bearish_obs": [],
            "range_condition": self._empty_range(),
        }

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
