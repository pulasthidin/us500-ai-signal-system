"""Tests for StructureAnalyzer — EMA, BOS/ChoCH, Wyckoff, Order Blocks, Range Detection."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.structure_analyzer import StructureAnalyzer


@pytest.fixture
def analyzer(mock_ctrader, mock_alert_bot):
    return StructureAnalyzer(mock_ctrader, us500_id=1, alert_bot=mock_alert_bot)


@pytest.fixture
def ranging_h1_df():
    """H1 bars that simulate a consolidation: tight range, low directional movement."""
    np.random.seed(99)
    n = 48
    base = 6500.0
    closes = base + np.random.randn(n) * 3
    opens = closes + np.random.randn(n) * 1
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 2
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 2
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-03-01", periods=n, freq="1h", tz="UTC"),
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": np.random.randint(5000, 50000, n),
    })


@pytest.fixture
def trending_h1_df():
    """H1 bars that simulate a strong uptrend: consistent higher closes."""
    np.random.seed(77)
    n = 48
    base = 6400.0
    trend = np.linspace(0, 150, n)
    closes = base + trend + np.random.randn(n) * 3
    opens = closes - np.abs(np.random.randn(n)) * 3
    highs = closes + np.abs(np.random.randn(n)) * 5
    lows = opens - np.abs(np.random.randn(n)) * 2
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-03-01", periods=n, freq="1h", tz="UTC"),
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": np.random.randint(5000, 50000, n),
    })


# ─── H1 bar fetching ─────────────────────────────────────

class TestGetH1Bars:
    def test_returns_dataframe(self, analyzer):
        result = analyzer.get_h1_bars(48)
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_returns_none_on_empty(self, analyzer):
        analyzer._ctrader.fetch_bars = lambda sid, tf, cnt: pd.DataFrame()
        result = analyzer.get_h1_bars()
        assert result is None


# ─── EMA calculation ──────────────────────────────────────

class TestEMAs:
    def test_calculate_emas_adds_columns(self, analyzer, sample_h4_df):
        import pandas_ta as ta
        ta.ema = MagicMock(side_effect=lambda series, length: pd.Series(np.full(len(series), 6500.0)))
        df = analyzer.calculate_emas(sample_h4_df)
        assert "ema50" in df.columns
        assert "ema200" in df.columns

    def test_price_vs_emas_bullish(self, analyzer):
        df = pd.DataFrame({"ema50": [6450.0], "ema200": [6400.0]})
        result = analyzer.get_price_vs_emas(6500.0, df)
        assert result["ema_bias"] == "bullish"
        assert result["above_ema200"] is True
        assert result["above_ema50"] is True

    def test_price_vs_emas_bearish(self, analyzer):
        df = pd.DataFrame({"ema50": [6550.0], "ema200": [6600.0]})
        result = analyzer.get_price_vs_emas(6500.0, df)
        assert result["ema_bias"] == "bearish"


# ─── Range detection ──────────────────────────────────────

class TestDetectRangeCondition:
    def test_returns_all_required_keys(self, analyzer):
        result = analyzer.detect_range_condition()
        required = ["is_ranging", "range_strength", "adx_value", "adx_ranging",
                     "atr_compressing", "atr_ratio", "range_high", "range_low"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_ranging_detected_with_flat_data(self, analyzer, ranging_h1_df):
        import pandas_ta as ta
        adx_df = pd.DataFrame({"ADX_14": [15.0] * len(ranging_h1_df)})
        ta.adx = MagicMock(return_value=adx_df)
        atr_series = pd.Series([5.0] * len(ranging_h1_df))
        ta.atr = MagicMock(return_value=atr_series)
        analyzer._ctrader.fetch_bars = MagicMock(return_value=ranging_h1_df)

        result = analyzer.detect_range_condition()
        assert result["is_ranging"] is True
        assert result["adx_ranging"] is True

    def test_trending_not_flagged_as_ranging(self, analyzer, trending_h1_df):
        import pandas_ta as ta
        adx_df = pd.DataFrame({"ADX_14": [35.0] * len(trending_h1_df)})
        ta.adx = MagicMock(return_value=adx_df)
        atr_vals = np.linspace(5, 10, len(trending_h1_df))
        ta.atr = MagicMock(return_value=pd.Series(atr_vals))
        analyzer._ctrader.fetch_bars = MagicMock(return_value=trending_h1_df)

        result = analyzer.detect_range_condition()
        assert result["is_ranging"] is False
        assert result["range_strength"] == "none"

    def test_range_boundaries_computed(self, analyzer, ranging_h1_df):
        import pandas_ta as ta
        adx_df = pd.DataFrame({"ADX_14": [15.0] * len(ranging_h1_df)})
        ta.adx = MagicMock(return_value=adx_df)
        ta.atr = MagicMock(return_value=pd.Series([5.0] * len(ranging_h1_df)))
        analyzer._ctrader.fetch_bars = MagicMock(return_value=ranging_h1_df)

        result = analyzer.detect_range_condition()
        assert result["range_high"] is not None
        assert result["range_low"] is not None
        assert result["range_high"] > result["range_low"]

    def test_empty_range_on_failure(self, analyzer):
        analyzer._ctrader.fetch_bars = MagicMock(return_value=None)
        result = analyzer.detect_range_condition()
        assert result["is_ranging"] is False
        assert result["range_strength"] == "none"


class TestEmptyRange:
    def test_empty_range_structure(self):
        result = StructureAnalyzer._empty_range()
        assert result["is_ranging"] is False
        assert result["adx_value"] is None
        assert result["range_high"] is None


# ─── Layer 2 integration ─────────────────────────────────

class TestGetLayer2Result:
    def test_result_includes_range_condition(self, analyzer):
        import pandas_ta as ta
        ta.ema = MagicMock(return_value=pd.Series([6500.0] * 200))
        ta.adx = MagicMock(return_value=pd.DataFrame({"ADX_14": [25.0] * 48}))
        ta.atr = MagicMock(return_value=pd.Series([10.0] * 200))
        from smartmoneyconcepts import smc
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame())
        smc.bos_choch = MagicMock(return_value=pd.DataFrame())

        result = analyzer.get_layer2_result(6500.0)
        assert "range_condition" in result
        assert isinstance(result["range_condition"], dict)

    def test_fallback_includes_range_condition(self, analyzer):
        analyzer._ctrader.fetch_bars = MagicMock(return_value=None)
        result = analyzer.get_layer2_result(6500.0)
        assert "range_condition" in result
        assert result["range_condition"]["is_ranging"] is False
