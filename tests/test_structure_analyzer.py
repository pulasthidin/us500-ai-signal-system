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

    def test_price_vs_emas_bullish(self, analyzer):
        df = pd.DataFrame({"ema50": [6450.0]})
        result = analyzer.get_price_vs_emas(6500.0, df)
        assert result["ema_bias"] == "bullish"
        assert result["above_ema50"] is True

    def test_price_vs_emas_bearish(self, analyzer):
        df = pd.DataFrame({"ema50": [6550.0]})
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


# ─── BOS detection ───────────────────────────────────────

class TestDetectBOS:
    def test_bullish_bos(self, analyzer, sample_h4_df):
        bos_choch_df = pd.DataFrame(
            {"BOS": [np.nan] * 198 + ["bullish", np.nan],
             "Level": [np.nan] * 198 + [6550.0, np.nan]},
            index=sample_h4_df.index,
        )
        result = analyzer.detect_bos(sample_h4_df, bos_choch=bos_choch_df)
        assert result["direction"] == "bullish"
        assert result["level"] == 6550.0
        assert result["bars_ago"] == 1

    def test_bearish_bos(self, analyzer, sample_h4_df):
        bos_choch_df = pd.DataFrame(
            {"BOS": [np.nan] * 195 + ["bearish"] + [np.nan] * 4,
             "Level": [np.nan] * 195 + [6400.0] + [np.nan] * 4},
            index=sample_h4_df.index,
        )
        result = analyzer.detect_bos(sample_h4_df, bos_choch=bos_choch_df)
        assert result["direction"] == "bearish"
        assert result["level"] == 6400.0
        assert result["bars_ago"] == 4

    def test_no_bos_in_flat_data(self, analyzer):
        n = 200
        flat_df = pd.DataFrame({
            "open": [6500.0] * n, "high": [6505.0] * n,
            "low": [6495.0] * n, "close": [6500.0] * n,
            "volume": [10000] * n,
        })
        bos_choch_df = pd.DataFrame({"BOS": [np.nan] * n})
        result = analyzer.detect_bos(flat_df, bos_choch=bos_choch_df)
        assert result["direction"] is None
        assert result["level"] is None
        assert result["bars_ago"] == 0


# ─── ChoCH detection ────────────────────────────────────

class TestDetectChoCH:
    def test_choch_recency_check(self, analyzer, sample_h4_df):
        bos_choch_df = pd.DataFrame(
            {"CHoCH": [np.nan] * 195 + ["bearish"] + [np.nan] * 4},
            index=sample_h4_df.index,
        )
        result = analyzer.detect_choch(sample_h4_df, bos_choch=bos_choch_df)
        assert result["direction"] == "bearish"
        assert isinstance(result["bars_ago"], int)
        assert result["bars_ago"] == 4
        assert result["bars_ago"] <= 10

    def test_no_choch_returns_none_direction(self, analyzer):
        n = 200
        flat_df = pd.DataFrame({
            "open": [6500.0] * n, "high": [6505.0] * n,
            "low": [6495.0] * n, "close": [6500.0] * n,
            "volume": [10000] * n,
        })
        bos_choch_df = pd.DataFrame({"CHoCH": [np.nan] * n})
        result = analyzer.detect_choch(flat_df, bos_choch=bos_choch_df)
        assert result["direction"] is None
        assert result["bars_ago"] == 0


# ─── Wyckoff detection ──────────────────────────────────

class TestDetectWyckoff:
    def test_markdown_in_downtrend(self, analyzer, sample_h4_df):
        n = 30
        closes = np.linspace(6600, 6400, n)
        opens = closes + 5
        highs = opens + 3
        lows = closes - 3
        volumes = np.concatenate([np.full(15, 5000), np.full(15, 15000)])
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        result = analyzer.detect_wyckoff(df)
        assert result == "markdown"

    def test_markup_in_uptrend(self, analyzer):
        n = 30
        closes = np.linspace(6400, 6600, n)
        opens = closes - 5
        highs = closes + 3
        lows = opens - 3
        volumes = np.concatenate([np.full(15, 5000), np.full(15, 15000)])
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        result = analyzer.detect_wyckoff(df)
        assert result == "markup"

    def test_unclear_in_sideways(self, analyzer):
        n = 30
        closes = np.full(n, 6500.0)
        opens = np.full(n, 6500.0)
        highs = np.full(n, 6502.0)
        lows = np.full(n, 6498.0)
        volumes = np.full(n, 10000)
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        result = analyzer.detect_wyckoff(df)
        assert result == "unclear"


# ─── Order Block detection ───────────────────────────────

class TestDetectOrderBlocks:
    def test_ob_detection_returns_dict(self, analyzer, sample_h4_df):
        from smartmoneyconcepts import smc
        import pandas_ta as ta

        ta.atr = MagicMock(return_value=pd.Series([10.0] * len(sample_h4_df)))
        swing_df = pd.DataFrame(index=sample_h4_df.index)
        bos_df = pd.DataFrame(
            {"BOS": [np.nan] * 198 + ["bullish", np.nan]},
            index=sample_h4_df.index,
        )
        smc.swing_highs_lows = MagicMock(return_value=swing_df)
        smc.bos_choch = MagicMock(return_value=bos_df)

        bos_result = {"direction": "bullish", "level": 6550.0, "bars_ago": 1}
        result = analyzer.detect_order_blocks(sample_h4_df, bos_result, 6500.0)

        assert isinstance(result, dict)
        assert "ob_bullish_nearby" in result
        assert "ob_bearish_nearby" in result
        assert "bullish_obs" in result
        assert "bearish_obs" in result

    def test_no_ob_when_no_bos(self, analyzer, sample_h4_df):
        from smartmoneyconcepts import smc
        import pandas_ta as ta

        ta.atr = MagicMock(return_value=pd.Series([10.0] * len(sample_h4_df)))
        bos_df = pd.DataFrame(
            {"BOS": [np.nan] * len(sample_h4_df)},
            index=sample_h4_df.index,
        )
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame(index=sample_h4_df.index))
        smc.bos_choch = MagicMock(return_value=bos_df)

        bos_result = {"direction": None, "level": None, "bars_ago": 0}
        result = analyzer.detect_order_blocks(sample_h4_df, bos_result, 6500.0)

        assert result["ob_bullish_nearby"] is False
        assert result["ob_bearish_nearby"] is False
        assert result["bullish_obs"] == []
        assert result["bearish_obs"] == []


# ─── Layer 2 integration (extended) ─────────────────────

class TestLayer2Integration:
    def test_bearish_ema_plus_bos_gives_short_bias(self, analyzer):
        import pandas_ta as ta
        from smartmoneyconcepts import smc

        n = 200
        np.random.seed(42)
        closes = np.full(n, 6400.0) + np.random.randn(n) * 2
        opens = closes + np.random.randn(n) * 1
        highs = np.maximum(opens, closes) + 2
        lows = np.minimum(opens, closes) - 2
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC"),
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": np.full(n, 10000),
        })
        analyzer._ctrader.fetch_bars = MagicMock(side_effect=lambda sid, tf, cnt: df.copy())

        ta.ema = MagicMock(return_value=pd.Series([6500.0] * n))
        ta.adx = MagicMock(return_value=pd.DataFrame({"ADX_14": [30.0] * n}))
        ta.atr = MagicMock(return_value=pd.Series([10.0] * n))

        bos_df = pd.DataFrame({"BOS": [np.nan] * (n - 2) + ["bearish", np.nan]})
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame())
        smc.bos_choch = MagicMock(return_value=bos_df)

        result = analyzer.get_layer2_result(6400.0)
        assert result["structure_bias"] == "short"
        assert result["score_contribution"] == 1

    def test_all_layer2_keys_present(self, analyzer):
        import pandas_ta as ta
        from smartmoneyconcepts import smc

        ta.ema = MagicMock(return_value=pd.Series([6500.0] * 200))
        ta.adx = MagicMock(return_value=pd.DataFrame({"ADX_14": [25.0] * 200}))
        ta.atr = MagicMock(return_value=pd.Series([10.0] * 200))
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame())
        smc.bos_choch = MagicMock(return_value=pd.DataFrame())

        result = analyzer.get_layer2_result(6500.0)
        expected_keys = [
            "above_ema50", "ema50_value", "ema_bias",
            "bos_direction", "bos_level", "bos_bars_ago",
            "choch_direction", "choch_recent", "wyckoff",
            "structure_bias", "score_contribution",
            "ob_bullish_nearby", "ob_bearish_nearby",
            "bullish_obs", "bearish_obs", "range_condition",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
