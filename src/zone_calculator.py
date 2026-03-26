"""
Layer 3 — H1 Key Zone Detection.
Previous-day highs/lows, round-number levels, volume POC, and proximity scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import os

import numpy as np
import pandas as pd

from config import (
    ASIAN_SESSION_END_UTC,
    ASIAN_SESSION_START_UTC,
    ROUND_LEVEL_INTERVAL,
    SWEEP_LOOKBACK_BARS,
    SWEEP_WICK_MIN_POINTS,
    ZONE_DEDUP_DISTANCE,
    ZONE_THRESHOLD_POINTS,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("zones")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/zones.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class ZoneCalculator:
    """Computes key S/R levels and determines whether price is at a tradeable zone."""

    def __init__(self, ctrader_connection, us500_id: int, alert_bot=None) -> None:
        self._ctrader = ctrader_connection
        self._us500_id = us500_id
        self._alert_bot = alert_bot

    # ─── round levels ────────────────────────────────────────

    def calculate_round_levels(self, current_price: float) -> List[float]:
        """Generate round-number levels every ROUND_LEVEL_INTERVAL points +/- 300 from price."""
        try:
            if ROUND_LEVEL_INTERVAL <= 0:
                return []
            base = int(current_price / ROUND_LEVEL_INTERVAL) * ROUND_LEVEL_INTERVAL
            start = base - 300
            end = base + 300
            levels = []
            price = start
            while price <= end:
                levels.append(float(price))
                price += ROUND_LEVEL_INTERVAL
            return sorted(levels)
        except Exception as exc:
            logger.error("calculate_round_levels failed: %s", exc, exc_info=True)
            return []

    # ─── previous day levels ─────────────────────────────────

    def get_previous_day_levels(self) -> Dict[str, Any]:
        """
        Fetch H1 bars for the last 2 trading days and compute
        previous-day high (PDH) and previous-day low (PDL).
        """
        try:
            df = self._ctrader.fetch_bars(self._us500_id, "H1", 48)
            if df is None or df.empty:
                logger.warning("No H1 bars for previous day levels")
                return {"pdh": None, "pdl": None, "date": None}

            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            dates = sorted(df["date"].unique())
            if len(dates) < 2:
                return {"pdh": None, "pdl": None, "date": None}
            else:
                yesterday = dates[-2]

            yesterday_bars = df[df["date"] == yesterday]
            pdh = float(yesterday_bars["high"].max())
            pdl = float(yesterday_bars["low"].min())

            logger.info("PDH=%.2f  PDL=%.2f  date=%s", pdh, pdl, yesterday)
            return {"pdh": round(pdh, 2), "pdl": round(pdl, 2), "date": str(yesterday)}

        except Exception as exc:
            logger.error("get_previous_day_levels failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "pd_levels", str(exc))
            return {"pdh": None, "pdl": None, "date": None}

    # ─── volume POC ──────────────────────────────────────────

    def calculate_poc(self, df: Optional[pd.DataFrame] = None) -> Optional[float]:
        """
        Approximate the volume Point of Control from H1 bars.
        Bins the price range into 50 buckets and accumulates volume per bin;
        the bin with the highest total volume is the POC.
        """
        try:
            if df is None:
                df = self._ctrader.fetch_bars(self._us500_id, "H1", 48)
            if df is None or df.empty:
                return None

            price_min = float(df["low"].min())
            price_max = float(df["high"].max())
            if price_max == price_min:
                return round(price_min, 2)

            num_bins = 50
            bin_edges = np.linspace(price_min, price_max, num_bins + 1)
            bin_volumes = np.zeros(num_bins)

            for _, bar in df.iterrows():
                mid = (bar["high"] + bar["low"]) / 2.0
                idx = int((mid - price_min) / (price_max - price_min) * num_bins)
                idx = max(0, min(idx, num_bins - 1))
                bin_volumes[idx] += bar["volume"]

            if bin_volumes.sum() == 0:
                poc_price = (price_min + price_max) / 2.0
            else:
                poc_idx = int(np.argmax(bin_volumes))
                poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0

            logger.info("POC calculated at %.2f", poc_price)
            return round(poc_price, 2)

        except Exception as exc:
            logger.error("calculate_poc failed: %s", exc, exc_info=True)
            return None

    # ─── equal highs / lows (Pine Script match) ────────────

    def detect_equal_highs_lows(self) -> List[Dict[str, Any]]:
        """
        Detect Equal Highs and Equal Lows from H1 swing points.
        Two swing highs within ATR threshold of each other = EQH (liquidity above).
        Two swing lows within ATR threshold of each other = EQL (liquidity below).
        Matches LuxAlgo Pine Script logic.
        """
        try:
            df = self._ctrader.fetch_bars(self._us500_id, "H1", 100)
            if df is None or df.empty:
                return []

            import pandas_ta as pta
            atr_series = pta.atr(df["high"], df["low"], df["close"], length=14)
            atr_val = float(atr_series.dropna().iloc[-1]) if atr_series is not None and not atr_series.dropna().empty else 0.0
            threshold = 0.1 * atr_val
            if threshold <= 0:
                return []

            swing_len = 3
            levels: List[Dict[str, Any]] = []

            swing_highs = []
            swing_lows = []
            for i in range(swing_len, len(df) - swing_len):
                is_sh = all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, swing_len + 1)) and \
                        all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, swing_len + 1))
                is_sl = all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, swing_len + 1)) and \
                        all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, swing_len + 1))
                if is_sh:
                    swing_highs.append(float(df["high"].iloc[i]))
                if is_sl:
                    swing_lows.append(float(df["low"].iloc[i]))

            for i in range(len(swing_highs)):
                for j in range(i + 1, len(swing_highs)):
                    if abs(swing_highs[i] - swing_highs[j]) < threshold:
                        avg_price = round((swing_highs[i] + swing_highs[j]) / 2, 2)
                        if not any(abs(avg_price - l["price"]) < threshold for l in levels):
                            levels.append({"price": avg_price, "type": "eqh"})

            for i in range(len(swing_lows)):
                for j in range(i + 1, len(swing_lows)):
                    if abs(swing_lows[i] - swing_lows[j]) < threshold:
                        avg_price = round((swing_lows[i] + swing_lows[j]) / 2, 2)
                        if not any(abs(avg_price - l["price"]) < threshold for l in levels):
                            levels.append({"price": avg_price, "type": "eql"})

            return levels

        except Exception as exc:
            logger.error("detect_equal_highs_lows failed: %s", exc, exc_info=True)
            return []

    # ─── Asian session range (AMD model) ────────────────────

    def get_asian_session_range(self) -> Dict[str, Any]:
        """
        Compute today's Asian session high and low (00:00–09:00 UTC).

        The Asian range is the accumulation phase of the AMD model.
        London/NY sweeps of this range signal the manipulation phase,
        which precedes the distribution (real) move.
        """
        try:
            df = self._ctrader.fetch_bars(self._us500_id, "M15", 100)
            if df is None or df.empty:
                return {"asian_high": None, "asian_low": None, "valid": False}

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            today = df["timestamp"].iloc[-1].normalize()

            start_h, start_m = map(int, ASIAN_SESSION_START_UTC.split(":"))
            end_h, end_m = map(int, ASIAN_SESSION_END_UTC.split(":"))
            session_start = today.replace(hour=start_h, minute=start_m)
            session_end = today.replace(hour=end_h, minute=end_m)

            asian_bars = df[(df["timestamp"] >= session_start) & (df["timestamp"] < session_end)]
            if asian_bars.empty:
                yesterday = today - pd.Timedelta(days=1)
                session_start = yesterday.replace(hour=start_h, minute=start_m)
                session_end = yesterday.replace(hour=end_h, minute=end_m)
                asian_bars = df[(df["timestamp"] >= session_start) & (df["timestamp"] < session_end)]

            if asian_bars.empty:
                return {"asian_high": None, "asian_low": None, "valid": False}

            asian_high = float(asian_bars["high"].max())
            asian_low = float(asian_bars["low"].min())

            logger.info("Asian range: %.2f – %.2f (%d bars)", asian_low, asian_high, len(asian_bars))
            return {
                "asian_high": round(asian_high, 2),
                "asian_low": round(asian_low, 2),
                "valid": True,
            }

        except Exception as exc:
            logger.error("get_asian_session_range failed: %s", exc, exc_info=True)
            return {"asian_high": None, "asian_low": None, "valid": False}

    # ─── liquidity sweep detection ────────────────────────────

    def detect_liquidity_sweeps(
        self, key_levels: List[Dict[str, Any]], m5_df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect liquidity sweeps of key levels in recent M5 bars.

        A sweep occurs when price wicks past a key level but closes back
        inside — smart money grabbed the resting liquidity (stop-loss orders)
        beyond that level.

        Buy-side sweep (above resistance): high > level, close < level
          → collected stop-losses from short sellers
          → bearish reversal signal

        Sell-side sweep (below support): low < level, close > level
          → collected stop-losses from long holders
          → bullish reversal signal

        Returns a list of swept levels with sweep details.
        """
        try:
            if m5_df is None:
                m5_df = self._ctrader.fetch_bars(self._us500_id, "M5", SWEEP_LOOKBACK_BARS + 5)
            if m5_df is None or m5_df.empty or not key_levels:
                return []

            recent = m5_df.tail(SWEEP_LOOKBACK_BARS)
            if recent.empty:
                return []

            sweeps: List[Dict[str, Any]] = []
            highs = recent["high"].values
            lows = recent["low"].values
            closes = recent["close"].values

            for lvl in key_levels:
                price = lvl.get("price")
                lvl_type = lvl.get("type", "unknown")
                if price is None:
                    continue

                buy_side_types = {"pdh", "eqh", "asian_high"}
                sell_side_types = {"pdl", "eql", "asian_low"}

                check_buy = lvl_type in buy_side_types or (lvl_type not in sell_side_types and price > closes[-1])
                check_sell = lvl_type in sell_side_types or (lvl_type not in buy_side_types and price < closes[-1])

                if check_buy:
                    for i in range(len(recent) - 1, -1, -1):
                        if highs[i] > price + SWEEP_WICK_MIN_POINTS and closes[i] < price:
                            bars_ago = len(recent) - 1 - i
                            sweeps.append({
                                "level_price": round(price, 2),
                                "level_type": lvl_type,
                                "sweep_side": "buy_side",
                                "sweep_high": round(float(highs[i]), 2),
                                "bars_ago": bars_ago,
                                "favors_direction": "SHORT",
                            })
                            break

                if check_sell:
                    for i in range(len(recent) - 1, -1, -1):
                        if lows[i] < price - SWEEP_WICK_MIN_POINTS and closes[i] > price:
                            bars_ago = len(recent) - 1 - i
                            sweeps.append({
                                "level_price": round(price, 2),
                                "level_type": lvl_type,
                                "sweep_side": "sell_side",
                                "sweep_low": round(float(lows[i]), 2),
                                "bars_ago": bars_ago,
                                "favors_direction": "LONG",
                            })
                            break

            if sweeps:
                logger.info(
                    "Liquidity sweeps detected: %s",
                    [(s["level_type"], s["sweep_side"], s["bars_ago"]) for s in sweeps],
                )

            return sweeps

        except Exception as exc:
            logger.error("detect_liquidity_sweeps failed: %s", exc, exc_info=True)
            return []

    # ─── aggregate levels ────────────────────────────────────

    def get_all_levels(
        self, current_price: float, include_asian: bool = True,
        _cached_asian: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Combine round levels, PDH, PDL, POC, EQH/EQL, and Asian session range
        into a single deduplicated list.
        """
        try:
            levels: List[Dict[str, Any]] = []

            for rl in self.calculate_round_levels(current_price):
                levels.append({"price": rl, "type": "round"})

            pd_levels = self.get_previous_day_levels()
            if pd_levels["pdh"] is not None:
                levels.append({"price": pd_levels["pdh"], "type": "pdh"})
            if pd_levels["pdl"] is not None:
                levels.append({"price": pd_levels["pdl"], "type": "pdl"})

            poc = self.calculate_poc()
            if poc is not None:
                levels.append({"price": poc, "type": "poc"})

            for eqhl in self.detect_equal_highs_lows():
                levels.append(eqhl)

            if include_asian:
                asian = _cached_asian if _cached_asian is not None else self.get_asian_session_range()
                if asian.get("valid"):
                    if asian["asian_high"] is not None:
                        levels.append({"price": asian["asian_high"], "type": "asian_high"})
                    if asian["asian_low"] is not None:
                        levels.append({"price": asian["asian_low"], "type": "asian_low"})

            levels.sort(key=lambda x: x["price"])

            _LEVEL_PRIORITY = {
                "pdh": 5, "pdl": 5, "eqh": 4, "eql": 4,
                "asian_high": 4, "asian_low": 4, "poc": 3, "round": 1,
            }

            deduped: List[Dict[str, Any]] = []
            for lvl in levels:
                merged = False
                for i, prev in enumerate(deduped):
                    if abs(lvl["price"] - prev["price"]) <= ZONE_DEDUP_DISTANCE:
                        if _LEVEL_PRIORITY.get(lvl["type"], 0) > _LEVEL_PRIORITY.get(prev["type"], 0):
                            deduped[i] = lvl
                        merged = True
                        break
                if not merged:
                    deduped.append(lvl)

            return deduped

        except Exception as exc:
            logger.error("get_all_levels failed: %s", exc, exc_info=True)
            return []

    # ─── nearest zone ────────────────────────────────────────

    def find_nearest_zone(
        self, current_price: float, all_levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find the closest key level within ZONE_THRESHOLD_POINTS.
        Also reports zone direction (support vs resistance) and all nearby levels.
        """
        try:
            if not all_levels:
                return {
                    "at_zone": False, "nearest_level": None, "zone_type": None,
                    "distance": 999.0, "zone_direction": None, "all_nearby": [],
                }

            nearest = None
            min_dist = float("inf")

            for lvl in all_levels:
                dist = abs(current_price - lvl["price"])
                if dist < min_dist:
                    min_dist = dist
                    nearest = lvl

            at_zone = min_dist <= ZONE_THRESHOLD_POINTS

            zone_direction = None
            if nearest:
                if current_price > nearest["price"]:
                    zone_direction = "support"
                elif current_price < nearest["price"]:
                    zone_direction = "resistance"
                else:
                    zone_direction = "at_level"

            all_nearby = [
                lvl for lvl in all_levels if abs(current_price - lvl["price"]) <= 20
            ]

            return {
                "at_zone": at_zone,
                "nearest_level": nearest["price"] if nearest else None,
                "zone_type": nearest["type"] if nearest else None,
                "distance": round(min_dist, 2),
                "zone_direction": zone_direction,
                "all_nearby": all_nearby,
            }

        except Exception as exc:
            logger.error("find_nearest_zone failed: %s", exc, exc_info=True)
            return {
                "at_zone": False, "nearest_level": None, "zone_type": None,
                "distance": 999.0, "zone_direction": None, "all_nearby": [],
            }

    # ─── Layer 3 composite ───────────────────────────────────

    def get_layer3_result(self, current_price: float) -> Dict[str, Any]:
        """Run full zone analysis including Asian range and liquidity sweeps."""
        try:
            asian_range = self.get_asian_session_range()
            all_levels = self.get_all_levels(current_price, include_asian=True, _cached_asian=asian_range)
            zone_info = self.find_nearest_zone(current_price, all_levels)

            liquidity_sweeps = self.detect_liquidity_sweeps(all_levels)

            has_buy_side_sweep = any(s["sweep_side"] == "buy_side" for s in liquidity_sweeps)
            has_sell_side_sweep = any(s["sweep_side"] == "sell_side" for s in liquidity_sweeps)

            return {
                "at_zone": zone_info["at_zone"],
                "zone_level": zone_info["nearest_level"],
                "zone_type": zone_info["zone_type"],
                "distance": zone_info["distance"],
                "zone_direction": zone_info["zone_direction"],
                "score_contribution": 1 if zone_info["at_zone"] else 0,
                "all_levels": all_levels,
                "all_nearby": zone_info["all_nearby"],
                "asian_range": asian_range,
                "liquidity_sweeps": liquidity_sweeps,
                "has_buy_side_sweep": has_buy_side_sweep,
                "has_sell_side_sweep": has_sell_side_sweep,
            }

        except Exception as exc:
            logger.error("get_layer3_result failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "zone_layer3", str(exc))
            return {
                "at_zone": False, "zone_level": None, "zone_type": None,
                "distance": 999.0, "zone_direction": None, "score_contribution": 0,
                "all_levels": [], "all_nearby": [],
                "asian_range": {"asian_high": None, "asian_low": None, "valid": False},
                "liquidity_sweeps": [],
                "has_buy_side_sweep": False, "has_sell_side_sweep": False,
            }

    # ─── helpers ─────────────────────────────────────────────

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
