"""Tests for ZoneCalculator — round levels, PDH/PDL, POC, zone proximity."""

import pytest
import numpy as np
import pandas as pd
from src.zone_calculator import ZoneCalculator
from config import ROUND_LEVEL_INTERVAL, ZONE_THRESHOLD_POINTS


@pytest.fixture
def zone_calc(mock_ctrader, mock_alert_bot):
    return ZoneCalculator(mock_ctrader, us500_id=1, alert_bot=mock_alert_bot)


class TestRoundLevels:
    def test_generates_levels(self, zone_calc):
        levels = zone_calc.calculate_round_levels(6550.0)
        assert len(levels) > 0
        assert all(lvl % ROUND_LEVEL_INTERVAL == 0 for lvl in levels)

    def test_levels_are_sorted(self, zone_calc):
        levels = zone_calc.calculate_round_levels(6550.0)
        assert levels == sorted(levels)

    def test_current_price_near_a_level(self, zone_calc):
        levels = zone_calc.calculate_round_levels(6500.0)
        assert 6500.0 in levels

    def test_range_spans_600_points(self, zone_calc):
        levels = zone_calc.calculate_round_levels(6500.0)
        assert max(levels) - min(levels) >= 600


class TestPreviousDayLevels:
    def test_returns_pdh_pdl(self, zone_calc):
        result = zone_calc.get_previous_day_levels()
        assert "pdh" in result
        assert "pdl" in result
        if result["pdh"] is not None:
            assert result["pdh"] >= result["pdl"]


class TestCalculatePOC:
    def test_poc_within_price_range(self, zone_calc, sample_ohlcv_df):
        poc = zone_calc.calculate_poc(sample_ohlcv_df)
        if poc is not None:
            assert sample_ohlcv_df["low"].min() <= poc <= sample_ohlcv_df["high"].max()

    def test_poc_with_zero_volume(self, zone_calc):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h"),
            "open": [100]*10, "high": [110]*10,
            "low": [90]*10, "close": [105]*10, "volume": [0]*10,
        })
        poc = zone_calc.calculate_poc(df)
        assert poc is not None
        assert 90 <= poc <= 110


class TestFindNearestZone:
    def test_at_zone_when_close(self, zone_calc):
        levels = [{"price": 6500.0, "type": "round"}, {"price": 6550.0, "type": "round"}]
        result = zone_calc.find_nearest_zone(6505.0, levels)
        assert result["at_zone"] is True
        assert result["nearest_level"] == 6500.0

    def test_not_at_zone_when_far(self, zone_calc):
        levels = [{"price": 6500.0, "type": "round"}, {"price": 6550.0, "type": "round"}]
        result = zone_calc.find_nearest_zone(6525.0, levels)
        assert result["at_zone"] is False
        assert result["distance"] == 25.0

    def test_empty_levels(self, zone_calc):
        result = zone_calc.find_nearest_zone(6500.0, [])
        assert result["at_zone"] is False
        assert result["nearest_level"] is None

    def test_zone_direction_support(self, zone_calc):
        levels = [{"price": 6490.0, "type": "pdl"}]
        result = zone_calc.find_nearest_zone(6495.0, levels)
        assert result["zone_direction"] == "support"

    def test_zone_direction_resistance(self, zone_calc):
        levels = [{"price": 6510.0, "type": "pdh"}]
        result = zone_calc.find_nearest_zone(6505.0, levels)
        assert result["zone_direction"] == "resistance"


# ─── Asian session range ──────────────────────────────────

class TestAsianSessionRange:
    def test_returns_required_keys(self, zone_calc):
        result = zone_calc.get_asian_session_range()
        assert "asian_high" in result
        assert "asian_low" in result
        assert "valid" in result

    def test_valid_range_when_data_available(self, zone_calc):
        now = pd.Timestamp.now("UTC").normalize()
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now, periods=60, freq="15min", tz="UTC"),
            "open": [6500] * 60, "high": [6520] * 60,
            "low": [6480] * 60, "close": [6510] * 60,
            "volume": [10000] * 60,
        })
        zone_calc._ctrader.fetch_bars = lambda sid, tf, cnt: bars
        result = zone_calc.get_asian_session_range()
        if result["valid"]:
            assert result["asian_high"] >= result["asian_low"]

    def test_empty_when_no_bars(self, zone_calc):
        zone_calc._ctrader.fetch_bars = lambda sid, tf, cnt: pd.DataFrame()
        result = zone_calc.get_asian_session_range()
        assert result["valid"] is False


# ─── Liquidity sweep detection ────────────────────────────

class TestDetectLiquiditySweeps:
    def test_detects_buy_side_sweep(self, zone_calc):
        """Price wicks above PDH but closes below = buy-side sweep."""
        n = 35
        m5 = pd.DataFrame({
            "high": [6500] * (n - 1) + [6522],
            "low": [6490] * n,
            "close": [6500] * (n - 1) + [6498],
            "open": [6495] * n,
        })
        levels = [{"price": 6520.0, "type": "pdh"}]
        sweeps = zone_calc.detect_liquidity_sweeps(levels, m5_df=m5)
        buy_side = [s for s in sweeps if s["sweep_side"] == "buy_side"]
        assert len(buy_side) >= 1
        assert buy_side[0]["level_type"] == "pdh"

    def test_detects_sell_side_sweep(self, zone_calc):
        """Price wicks below PDL but closes above = sell-side sweep."""
        n = 35
        m5 = pd.DataFrame({
            "high": [6510] * n,
            "low": [6500] * (n - 1) + [6478],
            "close": [6505] * (n - 1) + [6502],
            "open": [6505] * n,
        })
        levels = [{"price": 6480.0, "type": "pdl"}]
        sweeps = zone_calc.detect_liquidity_sweeps(levels, m5_df=m5)
        sell_side = [s for s in sweeps if s["sweep_side"] == "sell_side"]
        assert len(sell_side) >= 1

    def test_no_sweep_when_close_beyond_level(self, zone_calc):
        """If price closes beyond the level, it's a breakout not a sweep."""
        n = 35
        m5 = pd.DataFrame({
            "high": [6500] * (n - 1) + [6525],
            "low": [6490] * n,
            "close": [6500] * (n - 1) + [6523],
            "open": [6495] * n,
        })
        levels = [{"price": 6520.0, "type": "pdh"}]
        sweeps = zone_calc.detect_liquidity_sweeps(levels, m5_df=m5)
        buy_side = [s for s in sweeps if s["sweep_side"] == "buy_side"]
        assert len(buy_side) == 0

    def test_empty_levels_returns_empty(self, zone_calc):
        m5 = pd.DataFrame({"high": [6510], "low": [6490], "close": [6500], "open": [6495]})
        assert zone_calc.detect_liquidity_sweeps([], m5_df=m5) == []

    def test_sweep_includes_bars_ago(self, zone_calc):
        n = 35
        m5 = pd.DataFrame({
            "high": [6500] * (n - 5) + [6522] + [6500] * 4,
            "low": [6490] * n,
            "close": [6500] * (n - 5) + [6498] + [6500] * 4,
            "open": [6495] * n,
        })
        levels = [{"price": 6520.0, "type": "pdh"}]
        sweeps = zone_calc.detect_liquidity_sweeps(levels, m5_df=m5)
        if sweeps:
            assert "bars_ago" in sweeps[0]
            assert sweeps[0]["bars_ago"] >= 0


# ─── get_all_levels with Asian levels ─────────────────────

class TestGetAllLevelsWithAsian:
    def test_include_asian_true_adds_asian_levels(self, zone_calc):
        cached_asian = {"asian_high": 6520.0, "asian_low": 6480.0, "valid": True}
        zone_calc.detect_equal_highs_lows = lambda: []
        levels = zone_calc.get_all_levels(6500.0, include_asian=True, _cached_asian=cached_asian)
        types = [l["type"] for l in levels]
        assert "asian_high" in types or "asian_low" in types

    def test_include_asian_false_skips_asian(self, zone_calc):
        zone_calc.detect_equal_highs_lows = lambda: []
        levels = zone_calc.get_all_levels(6500.0, include_asian=False)
        types = [l["type"] for l in levels]
        assert "asian_high" not in types
        assert "asian_low" not in types


# ─── Layer 3 result with new fields ───────────────────────

class TestGetLayer3Result:
    def test_returns_score_1_at_zone(self, zone_calc):
        zone_calc.get_all_levels = lambda p, include_asian=True, _cached_asian=None: [{"price": p, "type": "round"}]
        zone_calc.get_asian_session_range = lambda: {"asian_high": None, "asian_low": None, "valid": False}
        zone_calc.detect_liquidity_sweeps = lambda levels, m5_df=None: []
        result = zone_calc.get_layer3_result(6500.0)
        assert result["score_contribution"] == 1
        assert result["at_zone"] is True

    def test_returns_score_0_between_zones(self, zone_calc):
        zone_calc.get_all_levels = lambda p, include_asian=True, _cached_asian=None: [{"price": 6500.0, "type": "round"}, {"price": 6550.0, "type": "round"}]
        zone_calc.get_asian_session_range = lambda: {"asian_high": None, "asian_low": None, "valid": False}
        zone_calc.detect_liquidity_sweeps = lambda levels, m5_df=None: []
        result = zone_calc.get_layer3_result(6525.0)
        assert result["score_contribution"] == 0

    def test_layer3_includes_sweep_fields(self, zone_calc):
        zone_calc.get_all_levels = lambda p, include_asian=True, _cached_asian=None: [{"price": p, "type": "round"}]
        zone_calc.get_asian_session_range = lambda: {"asian_high": 6520.0, "asian_low": 6480.0, "valid": True}
        zone_calc.detect_liquidity_sweeps = lambda levels, m5_df=None: [
            {"sweep_side": "buy_side", "level_type": "pdh", "bars_ago": 3}
        ]
        result = zone_calc.get_layer3_result(6500.0)
        assert "liquidity_sweeps" in result
        assert "has_buy_side_sweep" in result
        assert "has_sell_side_sweep" in result
        assert "asian_range" in result
        assert result["has_buy_side_sweep"] is True

    def test_layer3_fallback_has_new_fields(self, zone_calc):
        zone_calc.get_all_levels = lambda p, include_asian=True: (_ for _ in ()).throw(Exception("boom"))
        result = zone_calc.get_layer3_result(6500.0)
        assert "liquidity_sweeps" in result
        assert result["liquidity_sweeps"] == []
        assert result["has_buy_side_sweep"] is False
