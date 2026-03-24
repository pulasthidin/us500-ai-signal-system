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


class TestGetLayer3Result:
    def test_returns_score_1_at_zone(self, zone_calc):
        zone_calc.get_all_levels = lambda p: [{"price": p, "type": "round"}]
        result = zone_calc.get_layer3_result(6500.0)
        assert result["score_contribution"] == 1
        assert result["at_zone"] is True

    def test_returns_score_0_between_zones(self, zone_calc):
        zone_calc.get_all_levels = lambda p: [{"price": 6500.0, "type": "round"}, {"price": 6550.0, "type": "round"}]
        result = zone_calc.get_layer3_result(6525.0)
        assert result["score_contribution"] == 0
