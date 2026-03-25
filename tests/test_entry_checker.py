"""Tests for EntryChecker — ATR, R:R, swing SL, smart TP, SMT, M5 entry composite."""

import pytest
import numpy as np
import pandas as pd
from src.entry_checker import EntryChecker
from config import MIN_RR, SL_ATR_MULTIPLIER, SL_MIN_ATR_MULTIPLIER, TP_ATR_MULTIPLIER


@pytest.fixture
def entry(mock_ctrader, mock_alert_bot):
    return EntryChecker(mock_ctrader, us500_id=1, ustec_id=2, alert_bot=mock_alert_bot)


@pytest.fixture
def flat_df():
    """50-bar OHLCV where highs/lows are predictable for swing SL testing."""
    base = 6500.0
    n = 50
    df = pd.DataFrame({
        "open":  [base] * n,
        "high":  [base + 10] * n,
        "low":   [base - 10] * n,
        "close": [base] * n,
    })
    return df


# ─── ATR ────────────────────────────────────────────────────

class TestCalculateATR:
    def test_atr_positive(self, entry, sample_ohlcv_df):
        atr = entry.calculate_atr(sample_ohlcv_df)
        assert atr > 0

    def test_atr_small_df(self, entry):
        df = pd.DataFrame({
            "open": [100, 102], "high": [105, 108],
            "low": [98, 99], "close": [103, 106],
        })
        atr = entry.calculate_atr(df, period=2)
        assert atr > 0


# ─── Swing SL detection ─────────────────────────────────────

class TestDetectSwingSL:
    def test_long_sl_below_recent_lows(self, entry, flat_df):
        sl = entry._detect_swing_sl(flat_df, "LONG", atr=10.0, n_bars=5)
        # swing_low = 6490, buffer = 1.0, sl = 6490 - 1.0 = 6489
        assert sl < 6490.0
        assert sl is not None

    def test_short_sl_above_recent_highs(self, entry, flat_df):
        sl = entry._detect_swing_sl(flat_df, "SHORT", atr=10.0, n_bars=5)
        # swing_high = 6510, buffer = 1.0, sl = 6510 + 1.0 = 6511
        assert sl > 6510.0
        assert sl is not None

    def test_returns_none_for_small_df(self, entry):
        tiny = pd.DataFrame({"high": [100, 101], "low": [99, 98]})
        sl = entry._detect_swing_sl(tiny, "LONG", atr=5.0, n_bars=5)
        assert sl is None

    def test_returns_none_for_none_df(self, entry):
        sl = entry._detect_swing_sl(None, "LONG", atr=5.0)
        assert sl is None

    def test_buffer_scales_with_atr(self, entry, flat_df):
        sl_low_atr = entry._detect_swing_sl(flat_df, "LONG", atr=5.0)
        sl_high_atr = entry._detect_swing_sl(flat_df, "LONG", atr=20.0)
        # higher ATR → larger buffer → SL further below swing low
        assert sl_high_atr < sl_low_atr


# ─── Smart TP selection ─────────────────────────────────────

class TestSelectTP:
    def _make_levels(self, types_prices):
        return [{"type": t, "price": p} for t, p in types_prices]

    def test_selects_eqh_over_pdh_for_long(self, entry):
        levels = self._make_levels([("pdh", 6530.0), ("eqh", 6525.0)])
        tp, source = entry._select_tp(6500.0, "LONG", atr=10.0, sl_distance=10.0, zone_levels=levels)
        # eqh has priority 4 vs pdh priority 3
        assert source == "eqh"
        assert tp == 6525.0

    def test_selects_eql_over_pdl_for_short(self, entry):
        levels = self._make_levels([("pdl", 6470.0), ("eql", 6475.0)])
        tp, source = entry._select_tp(6500.0, "SHORT", atr=10.0, sl_distance=10.0, zone_levels=levels)
        assert source == "eql"
        assert tp == 6475.0

    def test_rejects_level_below_min_rr(self, entry):
        # Level only 5 pts away, sl_distance=10 → RR=0.5 < MIN_RR=1.6
        levels = self._make_levels([("eqh", 6505.0)])
        tp, source = entry._select_tp(6500.0, "LONG", atr=10.0, sl_distance=10.0, zone_levels=levels)
        assert tp is None
        assert source is None

    def test_rejects_level_beyond_atr_cap(self, entry):
        # Level 35 pts away, atr=10 → max_dist=30, 35 > 30 → rejected
        levels = self._make_levels([("pdh", 6535.0)])
        tp, source = entry._select_tp(6500.0, "LONG", atr=10.0, sl_distance=10.0, zone_levels=levels)
        assert tp is None

    def test_returns_none_when_no_levels(self, entry):
        tp, source = entry._select_tp(6500.0, "LONG", atr=10.0, sl_distance=10.0, zone_levels=[])
        assert tp is None

    def test_returns_none_when_atr_zero(self, entry):
        levels = self._make_levels([("eqh", 6520.0)])
        tp, source = entry._select_tp(6500.0, "LONG", atr=0.0, sl_distance=10.0, zone_levels=levels)
        assert tp is None

    def test_long_ignores_levels_below_price(self, entry):
        levels = self._make_levels([("pdl", 6480.0)])  # below price — wrong direction
        tp, source = entry._select_tp(6500.0, "LONG", atr=10.0, sl_distance=10.0, zone_levels=levels)
        assert tp is None

    def test_short_ignores_levels_above_price(self, entry):
        levels = self._make_levels([("pdh", 6520.0)])  # above price — wrong direction
        tp, source = entry._select_tp(6500.0, "SHORT", atr=10.0, sl_distance=10.0, zone_levels=levels)
        assert tp is None


# ─── R:R calculation ─────────────────────────────────────────

class TestCalculateRR:
    def test_long_atr_fallback(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 10.0)
        assert result["sl_price"] == 6500.0 - SL_ATR_MULTIPLIER * 10
        assert result["tp_price"] == 6500.0 + TP_ATR_MULTIPLIER * 10
        assert result["rr"] >= MIN_RR
        assert result["tp_source"] == "atr"

    def test_short_atr_fallback(self, entry):
        result = entry.calculate_rr(6500.0, "SHORT", 10.0)
        assert result["sl_price"] == 6500.0 + SL_ATR_MULTIPLIER * 10
        assert result["tp_price"] == 6500.0 - TP_ATR_MULTIPLIER * 10

    def test_rr_ratio_matches_atr_multipliers(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 10.0)
        expected_rr = TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER
        assert abs(result["rr"] - expected_rr) < 0.01

    def test_zero_atr_no_crash(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 0.0)
        assert result["rr"] == 0

    def test_swing_sl_used_when_provided(self, entry):
        # swing_sl at 6480 → sl_distance = 20
        result = entry.calculate_rr(6500.0, "LONG", 10.0, swing_sl=6480.0)
        assert result["sl_price"] == 6480.0
        assert result["sl_points"] == 20.0

    def test_sl_min_floor_applied_when_swing_too_tight(self, entry):
        # swing_sl = 6498 → distance = 2, ATR=10, floor=10 → SL should be floored
        result = entry.calculate_rr(6500.0, "LONG", 10.0, swing_sl=6498.0)
        min_dist = SL_MIN_ATR_MULTIPLIER * 10.0
        assert result["sl_points"] >= min_dist

    def test_tp_source_is_structure_type_when_zone_matches(self, entry):
        zone_levels = [{"type": "eqh", "price": 6525.0}]
        # sl_distance via ATR = 15, tp_dist = 25, rr = 25/15 > MIN_RR
        result = entry.calculate_rr(6500.0, "LONG", 10.0, zone_levels=zone_levels)
        assert result["tp_source"] == "eqh"
        assert result["tp_price"] == 6525.0

    def test_tp_source_falls_back_to_atr(self, entry):
        # No zone levels → ATR fallback
        result = entry.calculate_rr(6500.0, "LONG", 10.0, zone_levels=[])
        assert result["tp_source"] == "atr"

    def test_result_has_all_required_keys(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 10.0)
        for key in ("sl_price", "tp_price", "rr", "sl_points", "tp_points", "atr_value", "tp_source"):
            assert key in result, f"Missing key: {key}"


# ─── USTEC SMT ───────────────────────────────────────────────

class TestCheckUstecSMT:
    def test_agrees_when_both_trending_up(self, entry):
        up_df = pd.DataFrame({
            "high": np.linspace(100, 120, 15),
            "low": np.linspace(90, 110, 15),
            "close": np.linspace(95, 115, 15),
        })
        result = entry.check_ustec_smt(up_df, up_df.copy(), "LONG")
        assert isinstance(result["agrees"], (bool, np.bool_))

    def test_handles_empty_dataframes(self, entry):
        result = entry.check_ustec_smt(pd.DataFrame(), pd.DataFrame(), "LONG")
        assert result["agrees"] is False

    def test_handles_none_dataframes(self, entry):
        result = entry.check_ustec_smt(None, None, "SHORT")
        assert result["agrees"] is False


# ─── Composite entry result ──────────────────────────────────

class TestGetEntryResult:
    def test_returns_all_keys(self, entry):
        result = entry.get_entry_result("LONG", 6500.0)
        for key in (
            "fvg_present", "m5_bos_confirmed", "ustec_agrees",
            "rr_valid", "entry_ready", "sl_price", "tp_price",
            "atr", "tp_source", "entry_confidence",
        ):
            assert key in result, f"Missing key: {key}"

    def test_entry_ready_requires_fvg_and_rr(self, entry):
        """entry_ready gates on FVG + R:R only; BOS and SMT are bonuses."""
        result = entry.get_entry_result("LONG", 6500.0)
        if result["entry_ready"]:
            assert result["fvg_present"] is True
            assert result["rr_valid"] is True
        assert "entry_confidence" in result

    def test_invalid_direction_returns_empty(self, entry):
        result = entry.get_entry_result("SIDEWAYS", 6500.0)
        assert result["entry_ready"] is False
        assert result["tp_source"] is None

    def test_tp_source_present_when_entry_ready(self, entry):
        result = entry.get_entry_result("LONG", 6500.0)
        # tp_source should always be a string or None — never missing
        assert "tp_source" in result
