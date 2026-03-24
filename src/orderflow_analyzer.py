"""
Layer 4 — M15 Order-Flow Analysis.

Intrabar volume delta (matches LuxAlgo Volume Delta Candles Pine Script):
  For each M15 bar, fetch M1 sub-bars from cTrader.
  Buy volume  = sum of volume where M1 close > M1 open
  Sell volume = sum of volume where M1 close < M1 open
  Delta = buy_volume - sell_volume

Falls back to bar-color method if M1 bars unavailable.
Never returns NaN/missing delta. Never crashes. Never blocks signals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import os

import numpy as np
import pandas as pd

from config import VIX_SPIKE_THRESHOLD_PCT

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("orderflow")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/orderflow.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class OrderFlowAnalyzer:
    """Calculates intrabar volume delta, detects divergences, checks VIX spikes for Layer 4."""

    def __init__(self, ctrader_connection, us500_id: int, alert_bot=None) -> None:
        self._ctrader = ctrader_connection
        self._us500_id = us500_id
        self._alert_bot = alert_bot
        self._intrabar_available: Optional[bool] = None

    # ─── data fetching ───────────────────────────────────────

    def get_m15_bars(self, count: int = 50) -> Optional[pd.DataFrame]:
        """Fetch M15 bars for US500."""
        try:
            df = self._ctrader.fetch_bars(self._us500_id, "M15", count)
            if df is None or df.empty:
                logger.warning("No M15 bars returned")
                return None
            return df
        except Exception as exc:
            logger.error("get_m15_bars failed: %s", exc, exc_info=True)
            return None

    def _get_m1_bars_for_window(self, m15_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fetch all M1 bars covering the M15 DataFrame's time range in a single call."""
        try:
            bar_count = len(m15_df) * 15 + 15
            m1_df = self._ctrader.fetch_bars(self._us500_id, "M1", bar_count)
            if m1_df is None or m1_df.empty:
                return None
            m1_df["timestamp"] = pd.to_datetime(m1_df["timestamp"], utc=True)
            return m1_df
        except Exception as exc:
            logger.warning("M1 bars fetch failed: %s — will use bar-color fallback", exc)
            return None

    # ─── intrabar volume delta (Pine Script match) ───────────

    def calculate_intrabar_delta(self, m15_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each M15 bar, decompose into M1 sub-bars and compute true intrabar delta.
        Mirrors LuxAlgo Volume Delta Candles: request.security_lower_tf() approach.

        Falls back to bar-color method if M1 data unavailable.
        """
        m15_df = m15_df.copy()

        m1_df = self._get_m1_bars_for_window(m15_df)

        if m1_df is not None and len(m1_df) > 0:
            try:
                m15_df["timestamp"] = pd.to_datetime(m15_df["timestamp"], utc=True)
                buy_vols = []
                sell_vols = []
                deltas = []

                for i, row in m15_df.iterrows():
                    bar_start = row["timestamp"]
                    bar_end = bar_start + pd.Timedelta(minutes=15)

                    sub_bars = m1_df[(m1_df["timestamp"] >= bar_start) & (m1_df["timestamp"] < bar_end)]

                    if sub_bars.empty:
                        vol = row["volume"]
                        if row["close"] > row["open"]:
                            buy_vols.append(vol)
                            sell_vols.append(0)
                        elif row["close"] < row["open"]:
                            buy_vols.append(0)
                            sell_vols.append(vol)
                        else:
                            buy_vols.append(vol / 2.0)
                            sell_vols.append(vol / 2.0)
                    else:
                        bv = float(sub_bars.loc[sub_bars["close"] > sub_bars["open"], "volume"].sum())
                        sv = float(sub_bars.loc[sub_bars["close"] < sub_bars["open"], "volume"].sum())
                        doji_vol = float(sub_bars.loc[sub_bars["close"] == sub_bars["open"], "volume"].sum())
                        buy_vols.append(bv + doji_vol / 2.0)
                        sell_vols.append(sv + doji_vol / 2.0)

                    deltas.append(buy_vols[-1] - sell_vols[-1])

                m15_df["buy_volume"] = buy_vols
                m15_df["sell_volume"] = sell_vols
                m15_df["delta"] = deltas
                m15_df["cumulative_delta"] = m15_df["delta"].cumsum()

                if self._intrabar_available is not True:
                    self._intrabar_available = True
                    logger.info("Intrabar delta active — using M1 sub-bars within M15 candles")

                return m15_df

            except Exception as exc:
                logger.warning("Intrabar delta calculation failed: %s — falling back to bar-color", exc)

        return self._calculate_barcolor_delta(m15_df)

    def _calculate_barcolor_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback: assign entire bar volume based on bar color.
        Less accurate but guarantees a delta value for every bar.
        """
        try:
            df = df.copy()
            conditions = [df["close"] > df["open"], df["close"] < df["open"]]
            choices = [df["volume"], -df["volume"]]
            df["delta"] = np.select(conditions, choices, default=0)
            is_doji = df["close"] == df["open"]
            df["buy_volume"] = np.where(df["close"] > df["open"], df["volume"],
                               np.where(is_doji, df["volume"] / 2.0, 0))
            df["sell_volume"] = np.where(df["close"] < df["open"], df["volume"],
                                np.where(is_doji, df["volume"] / 2.0, 0))
            df["cumulative_delta"] = df["delta"].cumsum()

            if self._intrabar_available is None:
                self._intrabar_available = False
                logger.warning("Using bar-color delta fallback — M1 bars not available")

            return df
        except Exception as exc:
            logger.error("_calculate_barcolor_delta failed: %s", exc, exc_info=True)
            return df

    # ─── delta direction ─────────────────────────────────────

    def get_delta_direction(self, df: pd.DataFrame, lookback: int = 5) -> str:
        """Sum of delta over last *lookback* bars: positive = buyers, negative = sellers."""
        try:
            if "delta" not in df.columns or len(df) < lookback:
                return "mixed"
            total = df["delta"].iloc[-lookback:].sum()
            if total > 0:
                return "buyers"
            elif total < 0:
                return "sellers"
            return "mixed"
        except Exception as exc:
            logger.error("get_delta_direction failed: %s", exc, exc_info=True)
            return "mixed"

    # ─── delta divergence ────────────────────────────────────

    def detect_delta_divergence(self, df: pd.DataFrame, lookback: int = 5) -> str:
        """
        Compare price swing highs/lows to delta swing highs/lows.
        Price HH + delta LH = bearish divergence.
        Price LL + delta HL = bullish divergence.
        """
        try:
            if "delta" not in df.columns or len(df) < lookback * 2:
                return "none"

            recent = df.iloc[-lookback * 2:]
            first_half = recent.iloc[:lookback]
            second_half = recent.iloc[lookback:]

            price_high1 = first_half["high"].max()
            price_high2 = second_half["high"].max()
            delta_high1 = first_half["delta"].max()
            delta_high2 = second_half["delta"].max()

            ph2_at_edge = second_half["high"].iloc[-2:].max() == price_high2
            if price_high2 > price_high1 and delta_high2 < delta_high1 and ph2_at_edge:
                return "bearish"

            price_low1 = first_half["low"].min()
            price_low2 = second_half["low"].min()
            delta_low1 = first_half["delta"].min()
            delta_low2 = second_half["delta"].min()

            pl2_at_edge = second_half["low"].iloc[-2:].min() == price_low2
            if price_low2 < price_low1 and delta_low2 > delta_low1 and pl2_at_edge:
                return "bullish"

            return "none"

        except Exception as exc:
            logger.error("detect_delta_divergence failed: %s", exc, exc_info=True)
            return "none"

    # ─── VIX spike ───────────────────────────────────────────

    def is_vix_spiking_now(self, macro_data: Dict[str, Any]) -> bool:
        """Check if VIX is *rising* beyond the spike threshold.

        Only a VIX spike UP (fear increasing) should kill Layer 4.
        A VIX crash DOWN is bullish for equities and must not be penalised.
        """
        try:
            vix_pct = macro_data.get("vix_pct")
            if vix_pct is None:
                vix_pct = 0.0
            if isinstance(macro_data.get("raw_data"), dict):
                vix_info = macro_data["raw_data"].get("vix", {})
                raw_pct = vix_info.get("pct_change")
                if raw_pct is not None:
                    try:
                        vix_pct = float(raw_pct)
                    except (TypeError, ValueError):
                        logger.warning("Invalid raw_pct value: %r — using vix_pct fallback", raw_pct)
            return float(vix_pct) > VIX_SPIKE_THRESHOLD_PCT
        except Exception as exc:
            logger.error("is_vix_spiking_now failed: %s", exc, exc_info=True)
            return False

    # ─── Layer 4 composite ───────────────────────────────────

    def get_layer4_result(self, direction: Optional[str], macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full order-flow analysis.
        Confirms bias if delta direction matches trade direction and VIX is not spiking.
        """
        try:
            df = self.get_m15_bars(50)
            if df is None or df.empty:
                return self._fallback()

            df = self.calculate_intrabar_delta(df)
            delta_dir = self.get_delta_direction(df, lookback=5)
            divergence = self.detect_delta_divergence(df, lookback=5)
            vix_spiking = self.is_vix_spiking_now(macro_data)

            confirms = False
            if direction and not vix_spiking:
                if direction.upper() == "SHORT" and delta_dir == "sellers":
                    confirms = True
                elif direction.upper() == "LONG" and delta_dir == "buyers":
                    confirms = True

                if divergence == "bearish" and direction.upper() == "SHORT":
                    confirms = True
                elif divergence == "bullish" and direction.upper() == "LONG":
                    confirms = True

            score = 1 if confirms and not vix_spiking else 0

            return {
                "delta_direction": delta_dir,
                "divergence": divergence,
                "vix_spiking_now": vix_spiking,
                "confirms_bias": confirms,
                "score_contribution": score,
                "intrabar_active": self._intrabar_available or False,
            }

        except Exception as exc:
            logger.error("get_layer4_result failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "orderflow_layer4", str(exc))
            return self._fallback()

    # ─── fallback ────────────────────────────────────────────

    def _fallback(self) -> Dict[str, Any]:
        return {
            "delta_direction": "mixed",
            "divergence": "none",
            "vix_spiking_now": False,
            "confirms_bias": False,
            "score_contribution": 0,
            "intrabar_active": False,
        }

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
