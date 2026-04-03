"""Tests for MacroChecker — direction classification, VIX buckets, bias logic."""

import pytest
from src.macro_checker import MacroChecker


@pytest.fixture
def checker(mock_alert_bot):
    return MacroChecker(alert_bot=mock_alert_bot)


class TestClassifyDirection:
    def test_us10y_rising(self, checker):
        assert checker.classify_direction(0.1, "us10y") == "rising"

    def test_us10y_falling(self, checker):
        assert checker.classify_direction(-0.1, "us10y") == "falling"

    def test_us10y_flat(self, checker):
        assert checker.classify_direction(0.02, "us10y") == "flat"

    def test_oil_spiking(self, checker):
        assert checker.classify_direction(1.0, "oil") == "spiking"

    def test_oil_falling(self, checker):
        assert checker.classify_direction(-0.8, "oil") == "falling"

    def test_oil_stable(self, checker):
        assert checker.classify_direction(0.1, "oil") == "stable"

    def test_dxy_rising(self, checker):
        assert checker.classify_direction(0.3, "dxy") == "rising"

    def test_dxy_falling(self, checker):
        assert checker.classify_direction(-0.3, "dxy") == "falling"

    def test_rut_green(self, checker):
        assert checker.classify_direction(0.5, "rut") == "green"

    def test_rut_red(self, checker):
        assert checker.classify_direction(-0.5, "rut") == "red"

    def test_unknown_instrument(self, checker):
        assert checker.classify_direction(1.0, "unknown") == "flat"


class TestVixBucket:
    def test_low(self, checker):
        result = checker.get_vix_bucket(12.0)
        assert result["name"] == "low"
        assert result["size"] == "full"
        assert result["allowed"] is True

    def test_normal(self, checker):
        result = checker.get_vix_bucket(17.5)
        assert result["name"] == "normal"

    def test_elevated(self, checker):
        result = checker.get_vix_bucket(22.0)
        assert result["name"] == "elevated"
        assert result["size"] == "half"

    def test_high_short_only(self, checker):
        result = checker.get_vix_bucket(27.0)
        assert result["name"] == "high"
        assert result["short_only"] is True

    def test_extreme_short_only(self, checker):
        result = checker.get_vix_bucket(32.0)
        assert result["name"] == "extreme"
        assert result["allowed"] is True
        assert result["short_only"] is True

    def test_panic(self, checker):
        result = checker.get_vix_bucket(36.0)
        assert result["name"] == "panic"
        assert result["allowed"] is False


class TestVixDirectionBias:
    def test_sell_bias(self, checker):
        assert checker.get_vix_direction_bias(3.0) == "SELL_BIAS"

    def test_buy_bias(self, checker):
        assert checker.get_vix_direction_bias(-2.5) == "BUY_BIAS"

    def test_neutral(self, checker):
        assert checker.get_vix_direction_bias(0.5) == "NEUTRAL"


class TestMacroBias:
    def test_strong_bullish(self, checker):
        data = {
            "us10y": {"direction": "falling"},
            "oil": {"direction": "falling"},
            "dxy": {"direction": "falling"},
            "rut": {"direction": "green"},
        }
        result = checker.calculate_macro_bias(data)
        assert result["bias"] == "LONG"
        assert result["bullish_count"] == 4

    def test_strong_bearish(self, checker):
        data = {
            "us10y": {"direction": "rising"},
            "oil": {"direction": "spiking"},
            "dxy": {"direction": "rising"},
            "rut": {"direction": "red"},
        }
        result = checker.calculate_macro_bias(data)
        assert result["bias"] == "SHORT"
        assert result["bearish_count"] == 4

    def test_mixed_bias(self, checker):
        data = {
            "us10y": {"direction": "falling"},
            "oil": {"direction": "spiking"},
            "dxy": {"direction": "flat"},
            "rut": {"direction": "neutral"},
        }
        result = checker.calculate_macro_bias(data)
        assert result["bias"] == "MIXED"

    def test_empty_data_returns_mixed(self, checker):
        result = checker.calculate_macro_bias({})
        assert result["bias"] == "MIXED"


# ══════════════════════════════════════════════════════════════
# NEW TEST CLASSES — caching, Groq sentiment, Layer 1 integration
# ══════════════════════════════════════════════════════════════

import time
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from config import MACRO_CACHE_SECONDS


def _make_yf_dataframe():
    """Build a realistic MultiIndex DataFrame that yf.download would return."""
    tickers = ["^VIX", "DX-Y.NYB", "^TNX", "CL=F", "^RUT"]
    dates = pd.date_range("2025-03-28 09:30", periods=5, freq="1min", tz="UTC")
    arrays = {}
    np.random.seed(99)
    for t in tickers:
        base = {"^VIX": 17.0, "DX-Y.NYB": 99.5, "^TNX": 4.3, "CL=F": 72.0, "^RUT": 2100.0}[t]
        close = base + np.cumsum(np.random.randn(5) * 0.1)
        arrays[(t, "Open")] = close + 0.05
        arrays[(t, "High")] = close + 0.2
        arrays[(t, "Low")] = close - 0.2
        arrays[(t, "Close")] = close
        arrays[(t, "Volume")] = np.random.randint(100, 1000, 5).astype(float)
    df = pd.DataFrame(arrays, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestFetchMacroDataCache:
    """fetch_macro_data caches results and re-fetches on expiry."""

    def test_fresh_fetch_populates_cache(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = _make_yf_dataframe()
            result = checker.fetch_macro_data()

        assert result is not None
        assert checker._cache is not None
        assert "vix" in result or "fetch_time" in result

    def test_cache_hit_within_window(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = _make_yf_dataframe()
            checker.fetch_macro_data()
            checker.fetch_macro_data()
            assert mock_yf.download.call_count == 1

    def test_stale_cache_refetches(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = _make_yf_dataframe()
            checker.fetch_macro_data()

            checker._cache_time -= MACRO_CACHE_SECONDS + 1

            checker.fetch_macro_data()
            assert mock_yf.download.call_count == 2


class TestVixTickerFallback:
    """When yf.download returns no VIX data, Ticker fallback recovers it."""

    def test_vix_recovered_when_download_missing(self, mock_alert_bot):
        """VIX missing from download -> Ticker fallback provides value."""
        checker = MacroChecker(alert_bot=mock_alert_bot)
        df = _make_yf_dataframe()
        df[("^VIX", "Close")] = np.nan

        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 23.5
        mock_fast_info.previous_close = 22.0
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.fast_info = mock_fast_info

        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = df
            mock_yf.Ticker.return_value = mock_ticker_instance
            result = checker.fetch_macro_data()

        assert result["vix"]["value"] is not None
        assert result["vix"]["value"] == round(23.5, 4)
        expected_pct = round((23.5 - 22.0) / 22.0 * 100, 2)
        assert result["vix"]["pct_change"] == expected_pct

    def test_vix_stays_none_when_both_fail(self, mock_alert_bot):
        """VIX missing from download AND Ticker fails -> stays None."""
        checker = MacroChecker(alert_bot=mock_alert_bot)
        df = _make_yf_dataframe()
        df[("^VIX", "Close")] = np.nan

        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = df
            mock_yf.Ticker.side_effect = Exception("Yahoo API down")
            result = checker.fetch_macro_data()

        assert result["vix"]["value"] is None

    def test_fallback_not_called_when_vix_present(self, mock_alert_bot):
        """When VIX downloads fine, Ticker is never called."""
        checker = MacroChecker(alert_bot=mock_alert_bot)

        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = _make_yf_dataframe()
            result = checker.fetch_macro_data()

        mock_yf.Ticker.assert_not_called()
        assert result["vix"]["value"] is not None

    def test_fallback_uses_last_price_when_no_previous_close(self, mock_alert_bot):
        """When previous_close is unavailable, pct_change defaults to 0."""
        checker = MacroChecker(alert_bot=mock_alert_bot)
        df = _make_yf_dataframe()
        df[("^VIX", "Close")] = np.nan

        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 25.0
        mock_fast_info.previous_close = None
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.fast_info = mock_fast_info

        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = df
            mock_yf.Ticker.return_value = mock_ticker_instance
            result = checker.fetch_macro_data()

        assert result["vix"]["value"] == 25.0
        assert result["vix"]["pct_change"] == 0.0


class TestFetchGroqSentiment:
    """fetch_groq_sentiment returns parsed sentiment or NEUTRAL on failure."""

    def test_bullish_sentiment_parsed(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "BULLISH"

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_completion

        import sys
        groq_mod = sys.modules["groq"]
        original_groq = getattr(groq_mod, "Groq", None)
        groq_mod.Groq = MagicMock(return_value=mock_client_instance)
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
                result = checker.fetch_groq_sentiment()
            assert result == "BULLISH"
        finally:
            groq_mod.Groq = original_groq

    def test_api_error_returns_neutral(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API down")

        import sys
        groq_mod = sys.modules["groq"]
        original_groq = getattr(groq_mod, "Groq", None)
        groq_mod.Groq = MagicMock(return_value=mock_client_instance)
        try:
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
                result = checker.fetch_groq_sentiment()
            assert result == "NEUTRAL"
        finally:
            groq_mod.Groq = original_groq

    def test_no_api_key_returns_neutral(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            result = checker.fetch_groq_sentiment()
        assert result == "NEUTRAL"


class TestGetLayer1FullPath:
    """get_layer1_result combines macro data, VIX, and Groq into a scored dict."""

    def test_all_required_keys_present(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = _make_yf_dataframe()
            with patch.object(checker, "fetch_groq_sentiment", return_value="NEUTRAL"):
                result = checker.get_layer1_result()

        required_keys = [
            "bias", "vix_bucket", "vix_value", "vix_pct", "trade_allowed",
            "us10y_direction", "oil_direction", "dxy_direction", "rut_direction",
            "size_label", "short_only", "vix_direction_bias", "groq_sentiment",
            "bullish_count", "bearish_count", "score_contribution", "raw_data",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_partial_data_graceful(self, mock_alert_bot):
        checker = MacroChecker(alert_bot=mock_alert_bot)
        sparse_df = _make_yf_dataframe()
        sparse_df[("^TNX", "Close")] = np.nan
        sparse_df[("CL=F", "Close")] = np.nan

        with patch("src.macro_checker.yf") as mock_yf:
            mock_yf.download.return_value = sparse_df
            with patch.object(checker, "fetch_groq_sentiment", return_value="NEUTRAL"):
                result = checker.get_layer1_result()

        assert "bias" in result
        assert "vix_bucket" in result
