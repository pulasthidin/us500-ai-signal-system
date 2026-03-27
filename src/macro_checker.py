"""
Layer 1 — Macro Watchlist.
Pulls VIX, DXY, US10Y, Oil, RUT via yfinance (cached), computes macro bias,
VIX bucket / sizing, and optional Groq sentiment.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

from config import (
    MACRO_CACHE_SECONDS,
    MACRO_THRESHOLDS,
    VIX_BUCKETS,
    VIX_DIRECTION_THRESHOLD,
    YF_TICKERS,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("macro")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/macro.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class MacroChecker:
    """Fetches macro data, classifies directions, and produces Layer 1 score."""

    def __init__(self, alert_bot=None) -> None:
        self._alert_bot = alert_bot
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0.0
        self._cache_lock = threading.Lock()
        self._groq_cache: Optional[str] = None
        self._groq_cache_time: float = 0.0

    # ─── data fetching ───────────────────────────────────────

    def fetch_macro_data(self) -> Dict[str, Any]:
        """
        Pull latest intraday data for VIX, DXY, US10Y, Oil, RUT.
        Uses 1-minute interval, 1-day period. Retries 3x on failure.
        Returns cached result if within MACRO_CACHE_SECONDS window.

        Thread-safe: a lock prevents concurrent yfinance calls from racing
        on cache writes (one partial failure could overwrite another's
        complete result).
        """
        with self._cache_lock:
            if self._cache and (time.time() - self._cache_time) < MACRO_CACHE_SECONDS:
                return self._cache

            tickers_str = " ".join(v for k, v in YF_TICKERS.items() if k != "sp500")
            last_exc: Optional[Exception] = None

            for attempt in range(1, 4):
                try:
                    data = yf.download(
                        tickers_str,
                        interval="1m",
                        period="1d",
                        group_by="ticker",
                        progress=False,
                        threads=True,
                    )

                    result: Dict[str, Any] = {"fetch_time": datetime.now(timezone.utc).isoformat()}

                    for key, ticker in YF_TICKERS.items():
                        if key == "sp500":
                            continue
                        try:
                            is_multi = isinstance(data.columns, pd.MultiIndex)
                            if is_multi and ticker in data.columns.get_level_values(0):
                                ticker_df = data[ticker]
                            elif is_multi:
                                result[key] = {"value": None, "pct_change": 0.0, "direction": "flat", "prev_close": None}
                                continue
                            else:
                                ticker_df = data
                            close_series = ticker_df["Close"].dropna() if "Close" in ticker_df.columns else pd.Series(dtype=float)
                            if close_series.empty:
                                result[key] = {"value": None, "pct_change": 0.0, "direction": "flat", "prev_close": None}
                                continue
                            current = float(close_series.iloc[-1])
                            prev_close = float(close_series.iloc[0])
                            pct = ((current - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
                            direction = self.classify_direction(pct, key)
                            result[key] = {
                                "value": round(current, 4),
                                "pct_change": round(pct, 2),
                                "direction": direction,
                                "prev_close": round(prev_close, 4),
                            }
                        except Exception as inner_exc:
                            logger.warning("Failed to parse %s: %s", key, inner_exc)
                            result[key] = {"value": None, "pct_change": 0.0, "direction": "flat", "prev_close": None}

                    self._cache = result
                    self._cache_time = time.time()
                    logger.info("Macro data fetched successfully")
                    return result

                except Exception as exc:
                    last_exc = exc
                    logger.warning("yfinance attempt %d failed: %s", attempt, exc)
                    time.sleep(2)

            logger.error("All 3 yfinance attempts failed: %s", last_exc)
            self._send_system_alert("WARNING", "yfinance", f"All retries failed: {last_exc}")

            if self._cache:
                logger.info("Returning stale cached macro data")
                self._cache_time = time.time()
                return self._cache

            return self._empty_macro()

    # ─── classification ──────────────────────────────────────

    def classify_direction(self, pct_change: float, instrument: str) -> str:
        """Map a % change to a directional label based on per-instrument thresholds."""
        thresholds = MACRO_THRESHOLDS.get(instrument)
        if thresholds is None:
            return "flat"

        default = "flat" if instrument in ("us10y", "dxy") else ("stable" if instrument == "oil" else "neutral")

        for label, thresh in thresholds.items():
            if thresh > 0 and pct_change >= thresh:
                return label
            if thresh < 0 and pct_change <= thresh:
                return label
        return default

    # ─── VIX helpers ─────────────────────────────────────────

    def get_vix_bucket(self, vix_value: float) -> Dict[str, Any]:
        """Return the matching VIX_BUCKETS entry for *vix_value*."""
        for name, info in VIX_BUCKETS.items():
            lo, hi = info["range"]
            if lo <= vix_value < hi:
                return {"name": name, **info}
        return {"name": "extreme", **VIX_BUCKETS["extreme"]}

    def get_vix_direction_bias(self, vix_pct: float) -> str:
        """
        VIX green (falling) -> SELL bias on VIX means market going up -> BUY_BIAS
        VIX red (rising) -> market selling off -> SELL_BIAS
        """
        if vix_pct > VIX_DIRECTION_THRESHOLD:
            return "SELL_BIAS"
        elif vix_pct < -VIX_DIRECTION_THRESHOLD:
            return "BUY_BIAS"
        return "NEUTRAL"

    # ─── macro bias ──────────────────────────────────────────

    def calculate_macro_bias(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count bullish / bearish macro conditions:
          bullish: us10y falling + oil falling + dxy falling + rut green
          bearish: us10y rising  + oil spiking + dxy rising  + rut red
        Simple majority wins (e.g. 2-1 = directional, 2-2 = MIXED).
        """
        bullish = 0
        bearish = 0

        us10y_dir = macro_data.get("us10y", {}).get("direction", "flat")
        if us10y_dir == "falling":
            bullish += 1
        elif us10y_dir == "rising":
            bearish += 1

        oil_dir = macro_data.get("oil", {}).get("direction", "stable")
        if oil_dir == "falling":
            bullish += 1
        elif oil_dir == "spiking":
            bearish += 1

        dxy_dir = macro_data.get("dxy", {}).get("direction", "flat")
        if dxy_dir == "falling":
            bullish += 1
        elif dxy_dir == "rising":
            bearish += 1

        rut_dir = macro_data.get("rut", {}).get("direction", "neutral")
        if rut_dir == "green":
            bullish += 1
        elif rut_dir == "red":
            bearish += 1

        if bearish > bullish:
            bias = "SHORT"
        elif bullish > bearish:
            bias = "LONG"
        else:
            bias = "MIXED"

        return {"bias": bias, "bullish_count": bullish, "bearish_count": bearish}

    # ─── Groq sentiment ─────────────────────────────────────

    def fetch_groq_sentiment(self) -> str:
        """
        Ask Groq for one-word sentiment on today's US stock market headlines.
        Cached for MACRO_CACHE_SECONDS to avoid burning API quota.
        Non-critical: returns NEUTRAL on any failure.
        """
        try:
            if self._groq_cache and (time.time() - self._groq_cache_time) < MACRO_CACHE_SECONDS:
                return self._groq_cache

            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                return "NEUTRAL"

            from groq import Groq
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment classifier. "
                            "Given headlines, reply with exactly one word: BULLISH, BEARISH, or NEUTRAL."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Based on today's top US financial headlines, "
                            "rate the sentiment for the US stock market. "
                            "Reply with one word only: BULLISH, BEARISH, or NEUTRAL."
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=5,
            )
            raw = completion.choices[0].message.content.strip().upper()
            if raw in ("BULLISH", "BEARISH", "NEUTRAL"):
                logger.info("Groq sentiment: %s", raw)
                self._groq_cache = raw
                self._groq_cache_time = time.time()
                return raw
            logger.warning("Groq returned unexpected: %s — defaulting NEUTRAL", raw)
            return "NEUTRAL"
        except Exception as exc:
            logger.warning("Groq sentiment failed: %s — defaulting NEUTRAL", exc)
            return "NEUTRAL"

    # ─── Layer 1 composite ───────────────────────────────────

    def get_layer1_result(self) -> Dict[str, Any]:
        """Run the full Layer 1 macro analysis and return a scored result dict."""
        try:
            macro_data = self.fetch_macro_data()

            vix_info = macro_data.get("vix", {})
            vix_raw = vix_info.get("value")
            vix_pct = vix_info.get("pct_change") if vix_info.get("pct_change") is not None else 0.0

            if vix_raw is None:
                logger.warning("VIX value unavailable — blocking signals for safety")
                result = self._fallback_layer1()
                result["trade_allowed"] = False
                result["block_reason"] = "VIX data unavailable"
                return result

            vix_value = float(vix_raw)
            bucket = self.get_vix_bucket(vix_value)
            vix_dir_bias = self.get_vix_direction_bias(vix_pct)
            bias_result = self.calculate_macro_bias(macro_data)
            groq = self.fetch_groq_sentiment()

            score = 1 if bias_result["bias"] in ("LONG", "SHORT") else 0

            return {
                "bias": bias_result["bias"],
                "vix_bucket": bucket["name"],
                "vix_value": round(vix_value, 2),
                "vix_pct": round(vix_pct, 2),
                "size_label": bucket["size"],
                "trade_allowed": bucket["allowed"],
                "short_only": bucket["short_only"],
                "vix_direction_bias": vix_dir_bias,
                "us10y_direction": macro_data.get("us10y", {}).get("direction", "flat"),
                "oil_direction": macro_data.get("oil", {}).get("direction", "stable"),
                "dxy_direction": macro_data.get("dxy", {}).get("direction", "flat"),
                "rut_direction": macro_data.get("rut", {}).get("direction", "neutral"),
                "groq_sentiment": groq,
                "bullish_count": bias_result["bullish_count"],
                "bearish_count": bias_result["bearish_count"],
                "score_contribution": score,
                "raw_data": macro_data,
            }
        except Exception as exc:
            logger.error("get_layer1_result failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "macro_layer1", str(exc))
            return self._fallback_layer1()

    # ─── fallbacks ───────────────────────────────────────────

    def _empty_macro(self) -> Dict[str, Any]:
        return {
            "vix": {"value": None, "pct_change": 0.0, "direction": "flat", "prev_close": None},
            "dxy": {"value": None, "pct_change": 0.0, "direction": "flat", "prev_close": None},
            "us10y": {"value": None, "pct_change": 0.0, "direction": "flat", "prev_close": None},
            "oil": {"value": None, "pct_change": 0.0, "direction": "stable", "prev_close": None},
            "rut": {"value": None, "pct_change": 0.0, "direction": "neutral", "prev_close": None},
            "fetch_time": datetime.now(timezone.utc).isoformat(),
        }

    def _fallback_layer1(self) -> Dict[str, Any]:
        return {
            "bias": "MIXED", "vix_bucket": "normal", "vix_value": 0.0, "vix_pct": 0.0,
            "size_label": "normal", "trade_allowed": True, "short_only": False,
            "vix_direction_bias": "NEUTRAL", "us10y_direction": "flat",
            "oil_direction": "stable", "dxy_direction": "flat", "rut_direction": "neutral",
            "groq_sentiment": "NEUTRAL", "bullish_count": 0, "bearish_count": 0,
            "score_contribution": 0, "raw_data": self._empty_macro(),
        }

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
