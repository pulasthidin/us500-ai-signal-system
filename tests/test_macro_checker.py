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

    def test_extreme(self, checker):
        result = checker.get_vix_bucket(35.0)
        assert result["name"] == "extreme"
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
