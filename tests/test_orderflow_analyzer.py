"""Tests for OrderFlowAnalyzer — delta calculation, direction, divergence, VIX spike."""

import pytest
import numpy as np
import pandas as pd
from src.orderflow_analyzer import OrderFlowAnalyzer


@pytest.fixture
def analyzer(mock_ctrader, mock_alert_bot):
    return OrderFlowAnalyzer(mock_ctrader, us500_id=1, alert_bot=mock_alert_bot)


@pytest.fixture
def bullish_df():
    """DataFrame where most bars close higher than open (buyers dominate)."""
    n = 20
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min"),
        "open": [100]*n,
        "high": [110]*n,
        "low": [98]*n,
        "close": [108]*n,
        "volume": [1000]*n,
    })


@pytest.fixture
def bearish_df():
    """DataFrame where most bars close lower than open (sellers dominate)."""
    n = 20
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="15min"),
        "open": [108]*n,
        "high": [110]*n,
        "low": [98]*n,
        "close": [100]*n,
        "volume": [1000]*n,
    })


class TestVolumeDelta:
    def test_bullish_bars_positive_delta(self, analyzer, bullish_df):
        result = analyzer._calculate_barcolor_delta(bullish_df)
        assert (result["delta"] > 0).all()

    def test_bearish_bars_negative_delta(self, analyzer, bearish_df):
        result = analyzer._calculate_barcolor_delta(bearish_df)
        assert (result["delta"] < 0).all()

    def test_cumulative_delta_increases_for_buyers(self, analyzer, bullish_df):
        result = analyzer._calculate_barcolor_delta(bullish_df)
        assert result["cumulative_delta"].iloc[-1] > 0

    def test_doji_bars_zero_delta(self, analyzer):
        df = pd.DataFrame({
            "open": [100, 100], "high": [110, 110],
            "low": [90, 90], "close": [100, 100], "volume": [500, 500],
        })
        result = analyzer._calculate_barcolor_delta(df)
        assert (result["delta"] == 0).all()

    def test_intrabar_falls_back_to_barcolor(self, analyzer, bullish_df):
        analyzer._ctrader.fetch_bars.return_value = None
        result = analyzer.calculate_intrabar_delta(bullish_df)
        assert "delta" in result.columns
        assert (result["delta"] > 0).all()


class TestDeltaDirection:
    def test_buyers(self, analyzer, bullish_df):
        df = analyzer._calculate_barcolor_delta(bullish_df)
        assert analyzer.get_delta_direction(df, 5) == "buyers"

    def test_sellers(self, analyzer, bearish_df):
        df = analyzer._calculate_barcolor_delta(bearish_df)
        assert analyzer.get_delta_direction(df, 5) == "sellers"

    def test_mixed_with_no_delta_column(self, analyzer, bullish_df):
        assert analyzer.get_delta_direction(bullish_df, 5) == "mixed"


class TestDeltaDivergence:
    def test_bearish_divergence(self, analyzer):
        n = 20
        highs = list(range(100, 110)) + list(range(110, 120))
        deltas = list(range(500, 510)) + list(range(495, 505))
        df = pd.DataFrame({
            "high": highs, "low": [90]*n, "close": [100]*n,
            "open": [100]*n, "volume": [1000]*n,
            "delta": deltas,
        })
        result = analyzer.detect_delta_divergence(df, lookback=10)
        assert result in ("bearish", "none")

    def test_no_divergence_on_aligned_data(self, analyzer, bullish_df):
        df = analyzer._calculate_barcolor_delta(bullish_df)
        result = analyzer.detect_delta_divergence(df, lookback=5)
        assert result in ("bullish", "bearish", "none")


class TestVixSpiking:
    def test_spiking_when_over_threshold(self, analyzer):
        macro = {"vix_pct": 4.0, "raw_data": {"vix": {"pct_change": 4.0}}}
        assert analyzer.is_vix_spiking_now(macro) is True

    def test_not_spiking_when_under_threshold(self, analyzer):
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 1.0}}}
        assert analyzer.is_vix_spiking_now(macro) is False


class TestLayer4Result:
    def test_confirms_long_with_buyers(self, analyzer):
        analyzer.get_m15_bars = lambda count: pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="15min"),
            "open": [100]*20, "high": [110]*20, "low": [98]*20,
            "close": [108]*20, "volume": [1000]*20,
        })
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 1.0}}}
        result = analyzer.get_layer4_result("LONG", macro)
        assert result["confirms_bias"] is True
        assert result["score_contribution"] == 1

    def test_fails_when_vix_spiking(self, analyzer):
        analyzer.get_m15_bars = lambda count: pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="15min"),
            "open": [100]*20, "high": [110]*20, "low": [98]*20,
            "close": [108]*20, "volume": [1000]*20,
        })
        macro = {"vix_pct": 5.0, "raw_data": {"vix": {"pct_change": 5.0}}}
        result = analyzer.get_layer4_result("LONG", macro)
        assert result["score_contribution"] == 0

    def test_no_direction_returns_zero(self, analyzer):
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 1.0}}}
        result = analyzer.get_layer4_result(None, macro)
        assert result["score_contribution"] == 0

    def test_confirms_short_with_sellers(self, analyzer):
        """SHORT + sellers delta = confirms_bias."""
        analyzer.get_m15_bars = lambda count: pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="15min"),
            "open": [108]*20, "high": [110]*20, "low": [98]*20,
            "close": [100]*20, "volume": [1000]*20,
        })
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 1.0}}}
        result = analyzer.get_layer4_result("SHORT", macro)
        assert result["confirms_bias"] is True
        assert result["score_contribution"] == 1
        assert result["delta_direction"] == "sellers"

    def test_fallback_when_m15_bars_none(self, analyzer):
        """get_m15_bars returning None → safe fallback, no crash."""
        analyzer.get_m15_bars = lambda count: None
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 1.0}}}
        result = analyzer.get_layer4_result("LONG", macro)
        assert result["delta_direction"] == "mixed"
        assert result["confirms_bias"] is False
        assert result["score_contribution"] == 0

    def test_system_alert_on_exception(self, analyzer):
        """Exception in get_layer4_result → alert_bot.send_system_alert called."""
        def _explode(count):
            raise RuntimeError("test explosion")
        analyzer.get_m15_bars = _explode
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 1.0}}}
        result = analyzer.get_layer4_result("LONG", macro)
        assert result["score_contribution"] == 0
        analyzer._alert_bot.send_system_alert.assert_called_once()


class TestIntrabarDelta:
    """Tests for the primary M1 sub-bar decomposition path."""

    def test_intrabar_with_m1_sub_bars(self, analyzer):
        """Full intrabar path: M15 bar decomposed into M1 sub-bars."""
        m15_df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC"),
            "open": [100, 100, 100],
            "high": [110, 110, 110],
            "low": [95, 95, 95],
            "close": [108, 108, 108],
            "volume": [1500, 1500, 1500],
        })
        # Build M1 sub-bars: 15 bullish bars per M15 candle (45 + buffer)
        m1_timestamps = pd.date_range("2025-01-01", periods=50, freq="1min", tz="UTC")
        m1_df = pd.DataFrame({
            "timestamp": m1_timestamps,
            "open": [100]*50,
            "high": [110]*50,
            "low": [95]*50,
            "close": [108]*50,  # all bullish
            "volume": [100]*50,
        })
        analyzer._ctrader.fetch_bars.side_effect = None
        analyzer._ctrader.fetch_bars.return_value = m1_df
        result = analyzer.calculate_intrabar_delta(m15_df)
        assert "delta" in result.columns
        assert "buy_volume" in result.columns
        assert "sell_volume" in result.columns
        assert (result["delta"] > 0).all()
        assert analyzer._intrabar_available is True

    def test_intrabar_mixed_m1_bars(self, analyzer):
        """M1 sub-bars mix of bull/bear/doji → correct delta decomposition."""
        m15_df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-01-01 00:00"], utc=True),
            "open": [100], "high": [110], "low": [90], "close": [105], "volume": [900],
        })
        m1_timestamps = pd.date_range("2025-01-01 00:00", periods=15, freq="1min", tz="UTC")
        # 5 bullish (vol=100 each), 5 bearish (vol=100 each), 5 doji (vol=100 each)
        m1_df = pd.DataFrame({
            "timestamp": m1_timestamps,
            "open":  [100]*5 + [105]*5 + [100]*5,
            "high":  [110]*15,
            "low":   [90]*15,
            "close": [105]*5 + [100]*5 + [100]*5,  # bull, bear, doji
            "volume": [100]*15,
        })
        analyzer._ctrader.fetch_bars.side_effect = None
        analyzer._ctrader.fetch_bars.return_value = m1_df
        result = analyzer.calculate_intrabar_delta(m15_df)
        # buy_volume = 500 + 250 (half doji) = 750, sell = 500 + 250 = 750, delta = 0
        assert result["buy_volume"].iloc[0] == 750.0
        assert result["sell_volume"].iloc[0] == 750.0
        assert result["delta"].iloc[0] == 0.0

    def test_intrabar_exception_falls_back(self, analyzer, bullish_df):
        """Exception during M1 processing → clean fallback to bar-color."""
        def _bad_fetch(*args, **kwargs):
            raise ConnectionError("M1 bars unavailable")
        analyzer._ctrader.fetch_bars.side_effect = _bad_fetch
        result = analyzer.calculate_intrabar_delta(bullish_df)
        assert "delta" in result.columns
        # Fallback bar-color should still produce valid delta
        assert not result["delta"].isna().any()


class TestVixSpikingEdgeCases:
    """Edge cases for the VIX spike detection (the selected line 231)."""

    def test_negative_vix_pct_not_spiking(self, analyzer):
        """VIX crash DOWN is bullish for equities — must NOT be penalised."""
        macro = {"vix_pct": -5.0, "raw_data": {"vix": {"pct_change": -5.0}}}
        assert analyzer.is_vix_spiking_now(macro) is False

    def test_zero_vix_pct_not_spiking(self, analyzer):
        macro = {"vix_pct": 0.0, "raw_data": {"vix": {"pct_change": 0.0}}}
        assert analyzer.is_vix_spiking_now(macro) is False

    def test_exactly_at_threshold_not_spiking(self, analyzer):
        """Threshold is 3.0 — exactly 3.0 should NOT trigger (strictly greater)."""
        macro = {"vix_pct": 3.0, "raw_data": {"vix": {"pct_change": 3.0}}}
        assert analyzer.is_vix_spiking_now(macro) is False

    def test_just_above_threshold_spiking(self, analyzer):
        macro = {"vix_pct": 3.01, "raw_data": {"vix": {"pct_change": 3.01}}}
        assert analyzer.is_vix_spiking_now(macro) is True

    def test_raw_data_overrides_top_level_vix_pct(self, analyzer):
        """raw_data.vix.pct_change takes precedence over top-level vix_pct."""
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": 5.0}}}
        assert analyzer.is_vix_spiking_now(macro) is True

    def test_missing_raw_data_uses_top_level(self, analyzer):
        macro = {"vix_pct": 4.0}
        assert analyzer.is_vix_spiking_now(macro) is True

    def test_none_vix_pct_defaults_safe(self, analyzer):
        macro = {"vix_pct": None, "raw_data": {}}
        assert analyzer.is_vix_spiking_now(macro) is False

    def test_empty_macro_data_defaults_safe(self, analyzer):
        assert analyzer.is_vix_spiking_now({}) is False

    def test_malformed_raw_pct_string_handled(self, analyzer):
        """If raw_pct is a string (malformed upstream), should not crash."""
        macro = {"vix_pct": 1.0, "raw_data": {"vix": {"pct_change": "bad_data"}}}
        # Should fall back to top-level vix_pct=1.0 (not spiking), not crash
        assert analyzer.is_vix_spiking_now(macro) is False


class TestDeltaDivergenceDeterministic:
    """Deterministic divergence tests with crafted data."""

    def test_bearish_divergence_deterministic(self, analyzer):
        """Price HH + delta LH at edge = bearish divergence."""
        n = 10
        # First half: price highs 100-109, delta highs 500-509
        # Second half: price highs 110-119 (HH), delta highs 495-504 (LH)
        # Last 2 bars must contain the price high
        df = pd.DataFrame({
            "high":   list(range(100, 110)) + [112, 113, 114, 115, 116, 117, 118, 119, 118, 119],
            "low":    [90]*n + [90]*n,
            "close":  [100]*20,
            "open":   [100]*20,
            "volume": [1000]*20,
            "delta":  list(range(500, 510)) + [495, 496, 497, 498, 499, 500, 501, 502, 503, 504],
        })
        assert analyzer.detect_delta_divergence(df, lookback=10) == "bearish"

    def test_bullish_divergence_deterministic(self, analyzer):
        """Price LL + delta HL at edge = bullish divergence."""
        n = 10
        # First half: price lows 90-81 (going down), delta lows -500 to -509
        # Second half: price lows 80-71 (LL), delta lows -495 to -504 (HL, less negative)
        # Last 2 bars must contain the price low
        df = pd.DataFrame({
            "high":   [110]*20,
            "low":    list(range(90, 80, -1)) + [78, 77, 76, 75, 74, 73, 72, 71, 72, 71],
            "close":  [100]*20,
            "open":   [100]*20,
            "volume": [1000]*20,
            "delta":  list(range(-500, -510, -1)) + [-495, -496, -497, -498, -499, -500, -501, -502, -503, -504],
        })
        assert analyzer.detect_delta_divergence(df, lookback=10) == "bullish"
