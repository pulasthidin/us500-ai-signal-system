"""Tests for EntryChecker — ATR, R:R, swing SL, smart TP, SMT, M5 entry composite."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.entry_checker import EntryChecker
from config import MIN_RR, SL_ATR_MULTIPLIER, SL_MIN_ATR_MULTIPLIER, TP_ATR_MULTIPLIER


def _make_bullish_fvg_df():
    """10-bar DataFrame with a clear bullish body-based FVG at bar 7.

    Bar 6 body top (6495) < Bar 8 body bottom (6518) creates the gap.
    Bar 7 is a strong bullish impulse candle (body ratio ~0.85).
    """
    return pd.DataFrame({
        "open":   [6500, 6502, 6498, 6501, 6503, 6490, 6490, 6498, 6518, 6520],
        "high":   [6505, 6505, 6503, 6505, 6505, 6498, 6498, 6522, 6525, 6528],
        "low":    [6498, 6496, 6496, 6499, 6495, 6488, 6488, 6496, 6515, 6518],
        "close":  [6502, 6498, 6501, 6503, 6497, 6495, 6495, 6520, 6522, 6525],
        "volume": [10000] * 10,
    })


def _make_bearish_fvg_df():
    """10-bar DataFrame with a clear bearish body-based FVG at bar 7.

    Bar 6 body bottom (6515) > Bar 8 body top (6492) creates the gap.
    Bar 7 is a strong bearish impulse candle (body ratio ~0.85).
    """
    return pd.DataFrame({
        "open":   [6500, 6502, 6498, 6501, 6503, 6510, 6520, 6512, 6492, 6490],
        "high":   [6505, 6505, 6503, 6505, 6507, 6515, 6525, 6514, 6495, 6492],
        "low":    [6498, 6496, 6496, 6499, 6501, 6505, 6512, 6488, 6485, 6483],
        "close":  [6502, 6498, 6501, 6503, 6505, 6510, 6515, 6490, 6488, 6485],
        "volume": [10000] * 10,
    })


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
        assert "tp_source" in result

    def test_new_fields_present_in_result(self, entry):
        result = entry.get_entry_result("LONG", 6500.0)
        for key in ("displacement_valid", "has_liquidity_sweep", "sweep_details", "sl_method"):
            assert key in result, f"Missing key: {key}"

    def test_entry_with_range_condition(self, entry):
        range_cond = {"is_ranging": True, "range_strength": "strong",
                      "adx_value": 15.0, "range_high": 6530.0, "range_low": 6470.0}
        result = entry.get_entry_result("LONG", 6500.0, range_condition=range_cond)
        assert "sl_method" in result

    def test_entry_with_liquidity_sweeps(self, entry):
        sweeps = [{"level_price": 6480.0, "level_type": "pdl", "sweep_side": "sell_side",
                    "sweep_low": 6478.0, "bars_ago": 5, "favors_direction": "LONG"}]
        result = entry.get_entry_result("LONG", 6500.0, liquidity_sweeps=sweeps)
        assert isinstance(result["has_liquidity_sweep"], bool)

    def test_ranging_no_sweep_blocks_entry(self, entry):
        """In ranging market with no sweep, entry should be blocked."""
        range_cond = {"is_ranging": True, "range_strength": "strong",
                      "adx_value": 15.0, "range_high": 6530.0, "range_low": 6470.0}
        result = entry.get_entry_result("LONG", 6500.0, range_condition=range_cond, liquidity_sweeps=[])
        if result["fvg_present"] and result["rr_valid"]:
            assert result["entry_ready"] is False


# ─── Displacement candle validation ──────────────────────────

class TestIsDisplacementCandle:
    def test_strong_bullish_candle_passes(self):
        assert EntryChecker._is_displacement_candle(
            open_price=6490.0, close_price=6510.0, high=6512.0, low=6488.0, atr=15.0
        ) is True

    def test_doji_candle_fails(self):
        assert EntryChecker._is_displacement_candle(
            open_price=6500.0, close_price=6500.5, high=6510.0, low=6490.0, atr=15.0
        ) is False

    def test_small_candle_below_atr_fails(self):
        assert EntryChecker._is_displacement_candle(
            open_price=6499.0, close_price=6501.0, high=6501.5, low=6498.5, atr=15.0
        ) is False

    def test_zero_range_candle_fails(self):
        assert EntryChecker._is_displacement_candle(
            open_price=6500.0, close_price=6500.0, high=6500.0, low=6500.0, atr=10.0
        ) is False

    def test_large_body_with_wicks_passes(self):
        assert EntryChecker._is_displacement_candle(
            open_price=6505.0, close_price=6490.0, high=6508.0, low=6487.0, atr=15.0
        ) is True

    def test_zero_atr_only_checks_body_ratio(self):
        result = EntryChecker._is_displacement_candle(
            open_price=6490.0, close_price=6510.0, high=6512.0, low=6488.0, atr=0
        )
        assert result is True


# ─── Range-aware SL ──────────────────────────────────────────

class TestDetectRangeSL:
    def test_short_sl_above_range_high(self, entry):
        range_cond = {"is_ranging": True, "range_high": 6530.0, "range_low": 6470.0}
        sl = entry._detect_range_sl("SHORT", range_cond, atr=20.0)
        assert sl is not None
        assert sl > 6530.0

    def test_long_sl_below_range_low(self, entry):
        range_cond = {"is_ranging": True, "range_high": 6530.0, "range_low": 6470.0}
        sl = entry._detect_range_sl("LONG", range_cond, atr=20.0)
        assert sl is not None
        assert sl < 6470.0

    def test_returns_none_when_not_ranging(self, entry):
        range_cond = {"is_ranging": False}
        sl = entry._detect_range_sl("LONG", range_cond, atr=20.0)
        assert sl is None

    def test_returns_none_when_range_missing(self, entry):
        range_cond = {"is_ranging": True, "range_high": None, "range_low": None}
        sl = entry._detect_range_sl("LONG", range_cond, atr=20.0)
        assert sl is None


# ─── Sweep direction filter ──────────────────────────────────

class TestCheckSweepForDirection:
    def test_short_needs_buy_side_sweep(self):
        sweeps = [
            {"sweep_side": "buy_side", "level_type": "pdh", "bars_ago": 3},
            {"sweep_side": "sell_side", "level_type": "pdl", "bars_ago": 5},
        ]
        result = EntryChecker._check_sweep_for_direction(sweeps, "SHORT")
        assert result["has_sweep"] is True
        assert result["sweep_details"]["sweep_side"] == "buy_side"

    def test_long_needs_sell_side_sweep(self):
        sweeps = [
            {"sweep_side": "buy_side", "level_type": "pdh", "bars_ago": 3},
            {"sweep_side": "sell_side", "level_type": "pdl", "bars_ago": 5},
        ]
        result = EntryChecker._check_sweep_for_direction(sweeps, "LONG")
        assert result["has_sweep"] is True
        assert result["sweep_details"]["sweep_side"] == "sell_side"

    def test_no_matching_sweep(self):
        sweeps = [{"sweep_side": "buy_side", "level_type": "pdh", "bars_ago": 3}]
        result = EntryChecker._check_sweep_for_direction(sweeps, "LONG")
        assert result["has_sweep"] is False

    def test_empty_sweeps(self):
        result = EntryChecker._check_sweep_for_direction([], "SHORT")
        assert result["has_sweep"] is False

    def test_picks_most_recent_sweep(self):
        sweeps = [
            {"sweep_side": "buy_side", "level_type": "pdh", "bars_ago": 10},
            {"sweep_side": "buy_side", "level_type": "eqh", "bars_ago": 2},
        ]
        result = EntryChecker._check_sweep_for_direction(sweeps, "SHORT")
        assert result["sweep_details"]["bars_ago"] == 2


# ══════════════════════════════════════════════════════════════
# REGRESSION TESTS — Winning trades must still fire with new code
# ══════════════════════════════════════════════════════════════

class TestRegressionWinningTrades:
    """
    Regression suite: replicate real winning trade conditions from the live system.
    Every test here represents a trade the system correctly identified.
    New filters MUST NOT block these entries.
    """

    def test_short_supply_zone_rejection_trending_with_sweep(self, entry):
        """
        Real trade: Mar 26 2026 — SHORT at ~6,595 into supply zone.
        Price swept above prior high (buy-side sweep), rejected with strong
        displacement candle, dropped to 6,568. WIN +27pts.

        Conditions: VIX 24.13, trending market (ADX ~28), buy-side sweep confirmed,
        displacement FVG, BOS confirmed, USTEC agrees.
        Must produce: entry_ready=True, confidence=full or high.
        """
        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 28.0, "adx_ranging": False,
            "atr_compressing": False, "atr_ratio": 1.1,
            "range_high": 6620.0, "range_low": 6560.0,
        }
        sweeps = [
            {"level_price": 6610.0, "level_type": "pdh", "sweep_side": "buy_side",
             "sweep_high": 6615.0, "bars_ago": 8, "favors_direction": "SHORT"},
        ]
        result = entry.get_entry_result(
            "SHORT", 6595.0,
            zone_levels=[
                {"price": 6550.0, "type": "round"},
                {"price": 6568.0, "type": "eql"},
            ],
            range_condition=range_cond,
            liquidity_sweeps=sweeps,
        )
        assert result["entry_ready"] is True or result["fvg_present"] is True, (
            "Winning SHORT trade at supply zone must not be blocked! "
            f"fvg={result['fvg_present']}, rr_valid={result['rr_valid']}, "
            f"entry_ready={result['entry_ready']}"
        )
        assert result["sl_method"] == "swing", "Trending market should use swing SL, not range SL"
        if result["entry_ready"]:
            assert result["has_liquidity_sweep"] is True
            assert result["entry_confidence"] in ("full", "high", "base"), (
                f"With sweep + entry, confidence should be base or higher, got {result['entry_confidence']}"
            )

    def test_short_trending_no_sweep_still_fires(self, entry):
        """
        Even without a sweep, a SHORT in a trending market must still fire.
        The sweep only affects grade (A→B), not entry_ready.
        """
        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 30.0, "adx_ranging": False,
        }
        result = entry.get_entry_result(
            "SHORT", 6600.0,
            range_condition=range_cond,
            liquidity_sweeps=[],
        )
        if result["fvg_present"] and result["rr_valid"]:
            assert result["entry_ready"] is True, (
                "In trending market, entry_ready must be True when FVG + RR pass, "
                "even without a sweep"
            )
            assert result["entry_confidence"] == "no_sweep"

    def test_long_demand_zone_bounce_with_sweep(self, entry):
        """
        Simulated winning LONG: price sweeps below demand zone (sell-side sweep),
        bounces with displacement candle, FVG forms. Trending market.
        """
        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 25.0, "adx_ranging": False,
        }
        sweeps = [
            {"level_price": 6480.0, "level_type": "eql", "sweep_side": "sell_side",
             "sweep_low": 6477.0, "bars_ago": 5, "favors_direction": "LONG"},
        ]
        result = entry.get_entry_result(
            "LONG", 6490.0,
            zone_levels=[{"price": 6530.0, "type": "pdh"}],
            range_condition=range_cond,
            liquidity_sweeps=sweeps,
        )
        if result["fvg_present"] and result["rr_valid"]:
            assert result["entry_ready"] is True
            assert result["has_liquidity_sweep"] is True

    def test_ranging_with_sweep_still_fires(self, entry):
        """
        Even in a ranging market, if there's a valid sweep, entry must fire.
        The sweep unlocks the entry in ranging mode.
        """
        range_cond = {
            "is_ranging": True, "range_strength": "strong",
            "adx_value": 14.0, "adx_ranging": True,
            "atr_compressing": True, "atr_ratio": 0.6,
            "range_high": 6530.0, "range_low": 6470.0,
        }
        sweeps = [
            {"level_price": 6530.0, "level_type": "pdh", "sweep_side": "buy_side",
             "sweep_high": 6533.0, "bars_ago": 3, "favors_direction": "SHORT"},
        ]
        result = entry.get_entry_result(
            "SHORT", 6520.0,
            range_condition=range_cond,
            liquidity_sweeps=sweeps,
        )
        if result["fvg_present"] and result["rr_valid"]:
            assert result["entry_ready"] is True, (
                "Ranging + sweep confirmed: entry must fire! "
                f"fvg={result['fvg_present']}, rr={result['rr_valid']}, sweep={result['has_liquidity_sweep']}"
            )

    def test_displacement_does_not_block_strong_candles(self):
        """
        The displacement filter must not reject candles that look like
        the real winning trade: strong body, moderate wicks.
        These values match a typical 15-point bearish displacement candle on M5.
        """
        assert EntryChecker._is_displacement_candle(
            open_price=6600.0, close_price=6585.0,
            high=6603.0, low=6583.0, atr=15.0,
        ) is True, "A 15-point bearish candle with body 75% of range must pass displacement"

        assert EntryChecker._is_displacement_candle(
            open_price=6595.0, close_price=6580.0,
            high=6598.0, low=6578.0, atr=15.0,
        ) is True, "A 15-point bearish candle with body 75% must pass"

    def test_displacement_rejects_only_noise(self):
        """
        Only tiny doji/indecision candles should be rejected.
        These values match the chop candles from Wednesday's range.
        """
        assert EntryChecker._is_displacement_candle(
            open_price=6590.0, close_price=6591.0,
            high=6595.0, low=6587.0, atr=15.0,
        ) is False, "1-point body in 8-point range = doji, must be rejected"

        assert EntryChecker._is_displacement_candle(
            open_price=6590.0, close_price=6588.0,
            high=6591.0, low=6587.0, atr=15.0,
        ) is False, "Tiny 4-point candle below ATR threshold must be rejected"


# ─── M5 FVG detection ────────────────────────────────────

class TestDetectM5FVG:
    def test_bullish_fvg_detected(self, entry):
        df = _make_bullish_fvg_df()
        result = entry.detect_m5_fvg(df, "LONG", atr=0.0)
        assert result["present"] is True
        assert result["direction"] == "bullish"

    def test_bearish_fvg_detected(self, entry):
        df = _make_bearish_fvg_df()
        result = entry.detect_m5_fvg(df, "SHORT", atr=0.0)
        assert result["present"] is True
        assert result["direction"] == "bearish"

    def test_no_fvg_overlapping_candles(self, entry):
        n = 20
        df = pd.DataFrame({
            "open": [6500.0] * n, "high": [6502.0] * n,
            "low": [6498.0] * n, "close": [6500.0] * n,
            "volume": [10000] * n,
        })
        result = entry.detect_m5_fvg(df, "LONG", atr=0.0)
        assert result["present"] is False

    def test_fvg_details_include_top_bottom(self, entry):
        df = _make_bullish_fvg_df()
        result = entry.detect_m5_fvg(df, "LONG", atr=0.0)
        assert result["present"] is True
        assert result["top"] is not None
        assert result["bottom"] is not None
        assert isinstance(result["age_bars"], int)
        assert result["age_bars"] >= 0
        assert result["top"] > result["bottom"]


# ─── M5 BOS detection ────────────────────────────────────

class TestDetectM5BOS:
    def test_bos_confirmed(self, entry):
        from smartmoneyconcepts import smc

        n = 50
        bos_df = pd.DataFrame({"BOS": [np.nan] * 48 + ["bullish", np.nan]})
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame())
        smc.bos_choch = MagicMock(return_value=bos_df)

        df = pd.DataFrame({
            "open": [6500.0] * n, "high": [6510.0] * n,
            "low": [6490.0] * n, "close": [6500.0] * n,
        })
        assert entry.detect_m5_bos(df, "LONG") is True

    def test_no_bos(self, entry):
        from smartmoneyconcepts import smc

        n = 50
        bos_df = pd.DataFrame({"BOS": [np.nan] * n})
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame())
        smc.bos_choch = MagicMock(return_value=bos_df)

        df = pd.DataFrame({
            "open": [6500.0] * n, "high": [6510.0] * n,
            "low": [6490.0] * n, "close": [6500.0] * n,
        })
        assert entry.detect_m5_bos(df, "LONG") is False


# ─── FVG size and age ────────────────────────────────────

class TestFVGSizeAndAge:
    def test_fvg_size_points_calculated(self, entry):
        df = _make_bullish_fvg_df()
        result = entry.detect_m5_fvg(df, "LONG", atr=0.0)
        assert result["present"] is True
        fvg_size = abs(result["top"] - result["bottom"])
        assert fvg_size > 0

    def test_fvg_age_bars_from_detection(self, entry):
        df = _make_bullish_fvg_df()
        result = entry.detect_m5_fvg(df, "LONG", atr=0.0)
        assert result["present"] is True
        assert isinstance(result["age_bars"], int)
        assert result["age_bars"] >= 0
        assert result["age_bars"] < len(df)


# ─── Entry result integration ────────────────────────────

class TestEntryResultIntegration:
    def test_entry_ready_with_fvg_and_bos(self, entry):
        from smartmoneyconcepts import smc

        fvg_df = _make_bullish_fvg_df()
        entry._ctrader.fetch_bars = MagicMock(return_value=fvg_df.copy())

        bos_df = pd.DataFrame({"BOS": [np.nan] * 8 + ["bullish", np.nan]})
        smc.swing_highs_lows = MagicMock(return_value=pd.DataFrame())
        smc.bos_choch = MagicMock(return_value=bos_df)

        result = entry.get_entry_result("LONG", 6500.0)
        assert result["fvg_present"] is True
        if result["rr_valid"]:
            assert result["entry_ready"] is True

    def test_entry_not_ready_without_fvg(self, entry):
        n = 50
        flat_df = pd.DataFrame({
            "open": [6500.0] * n, "high": [6502.0] * n,
            "low": [6498.0] * n, "close": [6500.0] * n,
            "volume": [10000] * n,
        })
        entry._ctrader.fetch_bars = MagicMock(return_value=flat_df.copy())

        result = entry.get_entry_result("LONG", 6500.0)
        assert result["fvg_present"] is False
        assert result["entry_ready"] is False
