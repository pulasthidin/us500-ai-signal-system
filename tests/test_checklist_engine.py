"""Tests for ChecklistEngine — session detection, scoring, filters, decisions."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from src.checklist_engine import ChecklistEngine
from config import GRADE_A_SCORE, GRADE_B_SCORE


@pytest.fixture
def engine(mock_alert_bot):
    macro = MagicMock()
    struct = MagicMock()
    zones = MagicMock()
    flow = MagicMock()
    entry = MagicMock()
    sig_log = MagicMock()
    sig_log.is_duplicate.return_value = False

    eng = ChecklistEngine(
        macro_checker=macro,
        structure_analyzer=struct,
        zone_calculator=zones,
        orderflow_analyzer=flow,
        entry_checker=entry,
        signal_logger=sig_log,
        alert_bot=mock_alert_bot,
    )
    return eng


class TestSessionDetection:
    def test_is_trading_session_within_hours(self, engine):
        sl_tz = timezone(timedelta(hours=5.5))
        with patch("src.checklist_engine.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 3, 20, 14, 0, tzinfo=sl_tz)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = engine.is_trading_session()
            assert isinstance(result, bool)

    def test_get_current_session_returns_string(self, engine):
        session = engine.get_current_session()
        assert isinstance(session, str)

    def test_get_sl_time_format(self, engine):
        sl_time = engine.get_sl_time()
        assert "SL" in sl_time


class TestDayFlags:
    def test_day_flags_has_required_keys(self, engine):
        flags = engine.get_day_flags()
        assert "day_name" in flags
        assert "is_monday" in flags
        assert "is_friday" in flags
        assert "monday_caution" in flags
        assert "friday_caution" in flags


class TestScoring:
    def test_full_send_on_4_4_with_entry(self, engine, sample_macro_data, sample_news_data):
        engine._macro.get_layer1_result.return_value = sample_macro_data
        engine._structure.get_layer2_result.return_value = {
            "structure_bias": "long", "score_contribution": 1,
            "above_ema200": True, "above_ema50": True, "ema_bias": "bullish",
            "bos_direction": "bullish", "bos_level": 6500, "bos_bars_ago": 2,
            "choch_direction": None, "choch_recent": False, "wyckoff": "markup",
            "ema200_value": 6400, "ema50_value": 6450,
        }
        engine._zones.get_layer3_result.return_value = {
            "at_zone": True, "zone_level": 6500, "zone_type": "round",
            "distance": 3.0, "zone_direction": "support",
            "score_contribution": 1, "all_levels": [], "all_nearby": [],
        }
        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "buyers", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": True, "score_contribution": 1,
        }
        engine._entry.get_entry_result.return_value = {
            "fvg_present": True, "fvg_details": {},
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr_valid": True, "entry_ready": True, "entry_confidence": "full",
            "sl_price": 6485.0, "tp_price": 6525.0, "rr": 2.67,
            "sl_points": 15.0, "tp_points": 25.0, "atr": 10.0,
            "ustec_details": {},
        }

        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert result["score"] == 4
        assert result["decision"] == "FULL_SEND"
        assert result["grade"] == "A"

    def test_no_trade_on_low_score(self, engine, sample_macro_data, sample_news_data):
        sample_macro_data["score_contribution"] = 0
        sample_macro_data["bias"] = "MIXED"
        engine._structure.get_layer2_result.return_value = {
            "structure_bias": "unclear", "score_contribution": 0,
            "above_ema200": None, "above_ema50": None, "ema_bias": "unclear",
            "bos_direction": None, "bos_level": None, "bos_bars_ago": 0,
            "choch_direction": None, "choch_recent": False, "wyckoff": "unclear",
            "ema200_value": None, "ema50_value": None,
        }
        engine._zones.get_layer3_result.return_value = {
            "at_zone": False, "zone_level": None, "zone_type": None,
            "distance": 50, "zone_direction": None,
            "score_contribution": 0, "all_levels": [], "all_nearby": [],
        }
        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "mixed", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": False, "score_contribution": 0,
        }

        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert result["score"] <= 2
        assert result["decision"] == "NO_TRADE"


class TestDirectionDerivation:
    """_derive_direction returns (direction, confidence) tuple."""

    def test_macro_and_structure_agree_short(self, engine):
        layer1 = {"bias": "SHORT", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "short", "ema_bias": "bearish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "SHORT"
        assert confidence == "high"

    def test_macro_and_structure_agree_long(self, engine):
        layer1 = {"bias": "LONG", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "long", "ema_bias": "bullish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "LONG"
        assert confidence == "high"

    def test_vix_tiebreak_short(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "SELL_BIAS"}
        layer2 = {"structure_bias": "short", "ema_bias": "bearish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "SHORT"
        assert confidence == "high"

    def test_vix_tiebreak_long(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "BUY_BIAS"}
        layer2 = {"structure_bias": "long", "ema_bias": "bullish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "LONG"
        assert confidence == "high"

    def test_fallback_ema_bearish_plus_vix_sell(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "SELL_BIAS"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bearish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "SHORT"
        assert confidence == "medium"

    def test_fallback_ema_bullish_plus_macro_long(self, engine):
        layer1 = {"bias": "LONG", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bullish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "LONG"
        assert confidence == "medium"

    def test_fallback_ema_bearish_plus_macro_short(self, engine):
        layer1 = {"bias": "SHORT", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bearish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "SHORT"
        assert confidence == "medium"

    def test_fallback_ema_bullish_plus_vix_buy(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "BUY_BIAS"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bullish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction == "LONG"
        assert confidence == "medium"

    def test_no_direction_when_all_unclear(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "unclear"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction is None
        assert confidence is None

    def test_no_direction_when_ema_conflicts_with_macro(self, engine):
        layer1 = {"bias": "LONG", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bearish"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction is None
        assert confidence is None

    def test_no_direction_when_ema_missing(self, engine):
        layer1 = {"bias": "SHORT", "vix_direction_bias": "SELL_BIAS"}
        layer2 = {"structure_bias": "unclear"}
        direction, confidence = engine._derive_direction(layer1, layer2)
        assert direction is None
        assert confidence is None


class TestFilters:
    def test_hard_stop_on_vix_over_30(self, engine, sample_news_data):
        result = {
            "layer1": {"vix_value": 35.0}, "direction": "SHORT",
            "score": 4, "grade": "A", "decision": "FULL_SEND",
            "caution_flags": [], "block_reason": None,
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        assert filtered["decision"] == "HARD_STOP"
        assert filtered["block_reason"] == "VIX >= 30"

    def test_pre_news_blocks(self, engine):
        news = {"pre_news_blocked": True, "pre_news_event": "CPI", "is_news_day": True,
                "post_news_caution": False}
        result = {
            "layer1": {"vix_value": 18.0, "short_only": False},
            "direction": "LONG", "score": 4, "grade": "A",
            "decision": "FULL_SEND", "caution_flags": [], "block_reason": None,
        }
        filtered = engine.apply_all_filters(result, news)
        assert filtered["decision"] == "NO_TRADE"
        assert "Pre-news" in filtered["block_reason"]

    def test_short_only_blocks_long(self, engine, sample_news_data):
        result = {
            "layer1": {"vix_value": 27.0, "short_only": True},
            "direction": "LONG", "score": 3, "grade": "B",
            "decision": "HALF_SIZE", "caution_flags": [], "block_reason": None,
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        assert filtered["decision"] == "NO_TRADE"
        assert "short only" in filtered["block_reason"]


class TestShouldSendAlert:
    def test_full_send_sends(self, engine):
        result = {"decision": "FULL_SEND", "entry_ready": True,
                  "block_reason": None, "direction": "LONG",
                  "current_price": 6500, "score": 4}
        assert engine.should_send_alert(result) is True

    def test_no_trade_silent(self, engine):
        result = {"decision": "NO_TRADE", "entry_ready": False,
                  "block_reason": None, "score": 1}
        assert engine.should_send_alert(result) is False

    def test_hard_stop_alerts_once_then_suppressed(self, engine):
        result = {"decision": "HARD_STOP", "entry_ready": False,
                  "block_reason": "VIX > 30", "score": 0}
        assert engine.should_send_alert(result) is True
        assert engine.should_send_alert(result) is False

    def test_hard_stop_resets_when_vix_drops(self, engine):
        hard = {"decision": "HARD_STOP", "entry_ready": False,
                "block_reason": "VIX > 30", "score": 0}
        assert engine.should_send_alert(hard) is True
        assert engine.should_send_alert(hard) is False
        normal = {"decision": "NO_TRADE", "entry_ready": False,
                  "block_reason": None, "score": 1}
        engine.should_send_alert(normal)
        assert engine.should_send_alert(hard) is True

    def test_wait_with_score_3_and_direction_alerts(self, engine):
        result = {"decision": "WAIT", "entry_ready": False,
                  "block_reason": None, "score": 3,
                  "direction": "SHORT", "current_price": 6500}
        assert engine.should_send_alert(result) is True

    def test_wait_without_direction_silent(self, engine):
        result = {"decision": "WAIT", "entry_ready": False,
                  "block_reason": None, "score": 3}
        assert engine.should_send_alert(result) is False

    def test_blocked_signal_does_not_send(self, engine):
        result = {"decision": "FULL_SEND", "entry_ready": True,
                  "block_reason": "Pre-news: CPI", "score": 4}
        assert engine.should_send_alert(result) is False


# ─── Range / sweep filter paths ──────────────────────────────

class TestRangeFilters:
    def test_apply_filters_adds_strong_range_caution(self, engine, sample_news_data):
        result = {
            "layer1": {"vix_value": 17.0, "short_only": False},
            "direction": "SHORT", "score": 4, "grade": "A",
            "decision": "FULL_SEND", "caution_flags": [], "block_reason": None,
            "entry_ready": True, "has_liquidity_sweep": True,
            "direction_confidence": "high",
            "range_condition": {
                "is_ranging": True, "range_strength": "strong", "adx_value": 14.0,
            },
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        range_flags = [f for f in filtered["caution_flags"] if "RANGE" in f.upper()]
        assert len(range_flags) >= 1
        assert "STRONG RANGE" in range_flags[0]

    def test_apply_filters_adds_mild_range_caution(self, engine, sample_news_data):
        result = {
            "layer1": {"vix_value": 17.0, "short_only": False},
            "direction": "LONG", "score": 3, "grade": "B",
            "decision": "HALF_SIZE", "caution_flags": [], "block_reason": None,
            "entry_ready": True, "has_liquidity_sweep": False,
            "direction_confidence": "medium",
            "range_condition": {
                "is_ranging": True, "range_strength": "mild", "adx_value": 19.0,
            },
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        range_flags = [f for f in filtered["caution_flags"] if "RANGE" in f.upper()]
        assert len(range_flags) >= 1
        assert "MILD RANGE" in range_flags[0]

    def test_no_sweep_caution_flag(self, engine, sample_news_data):
        result = {
            "layer1": {"vix_value": 17.0, "short_only": False},
            "direction": "LONG", "score": 4, "grade": "A",
            "decision": "FULL_SEND", "caution_flags": [], "block_reason": None,
            "entry_ready": True, "has_liquidity_sweep": False,
            "direction_confidence": "high",
            "range_condition": {"is_ranging": False},
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        sweep_flags = [f for f in filtered["caution_flags"] if "SWEEP" in f.upper()]
        assert len(sweep_flags) >= 1

    def test_no_range_flag_when_trending(self, engine, sample_news_data):
        result = {
            "layer1": {"vix_value": 17.0, "short_only": False},
            "direction": "LONG", "score": 4, "grade": "A",
            "decision": "FULL_SEND", "caution_flags": [], "block_reason": None,
            "entry_ready": True, "has_liquidity_sweep": True,
            "direction_confidence": "high",
            "range_condition": {"is_ranging": False, "range_strength": "none"},
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        range_flags = [f for f in filtered["caution_flags"] if "RANGE" in f.upper()]
        assert len(range_flags) == 0

    def test_fallback_result_has_range_fields(self, engine):
        result = engine._fallback_result(6500.0)
        assert "is_ranging" in result
        assert "range_condition" in result
        assert "has_liquidity_sweep" in result
        assert result["is_ranging"] is False


class TestChecklistRangeSweepIntegration:
    def _setup_layers_with_range(self, engine, sample_macro_data, is_ranging=False):
        """Helper to set up all 4 layers with range condition for full checklist tests."""
        engine._macro.get_layer1_result.return_value = sample_macro_data
        range_cond = {
            "is_ranging": is_ranging, "range_strength": "strong" if is_ranging else "none",
            "adx_value": 15.0 if is_ranging else 30.0,
            "adx_ranging": is_ranging, "atr_compressing": is_ranging,
            "atr_ratio": 0.6 if is_ranging else 1.0,
            "range_high": 6530.0, "range_low": 6470.0,
        }
        engine._structure.get_layer2_result.return_value = {
            "structure_bias": "long", "score_contribution": 1,
            "above_ema200": True, "above_ema50": True, "ema_bias": "bullish",
            "bos_direction": "bullish", "bos_level": 6500, "bos_bars_ago": 2,
            "choch_direction": None, "choch_recent": False, "wyckoff": "markup",
            "ema200_value": 6400, "ema50_value": 6450,
            "ob_bullish_nearby": False, "ob_bearish_nearby": False,
            "bullish_obs": [], "bearish_obs": [],
            "range_condition": range_cond,
        }
        engine._zones.get_layer3_result.return_value = {
            "at_zone": True, "zone_level": 6500, "zone_type": "round",
            "distance": 3.0, "zone_direction": "support",
            "score_contribution": 1, "all_levels": [], "all_nearby": [],
            "asian_range": {"asian_high": 6510, "asian_low": 6490, "valid": True},
            "liquidity_sweeps": [],
            "has_buy_side_sweep": False, "has_sell_side_sweep": False,
        }
        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "buyers", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": True, "score_contribution": 1,
        }

    def test_checklist_result_includes_range_fields(self, engine, sample_macro_data, sample_news_data):
        self._setup_layers_with_range(engine, sample_macro_data, is_ranging=False)
        engine._entry.get_entry_result.return_value = {
            "entry_ready": True, "entry_confidence": "no_sweep",
            "fvg_present": True, "fvg_details": {}, "rr_valid": True,
            "has_liquidity_sweep": False, "displacement_valid": True,
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "sl_price": 6485, "tp_price": 6525, "rr": 2.5,
            "sl_points": 15, "tp_points": 25, "atr": 10, "tp_source": "atr",
            "sl_method": "swing", "sweep_details": None,
        }
        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert "is_ranging" in result
        assert "range_condition" in result
        assert "has_liquidity_sweep" in result

    def test_no_sweep_trending_downgrades_a_to_b(self, engine, sample_macro_data, sample_news_data):
        self._setup_layers_with_range(engine, sample_macro_data, is_ranging=False)
        engine._entry.get_entry_result.return_value = {
            "entry_ready": True, "entry_confidence": "no_sweep",
            "fvg_present": True, "fvg_details": {}, "rr_valid": True,
            "has_liquidity_sweep": False, "displacement_valid": True,
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "sl_price": 6485, "tp_price": 6525, "rr": 2.5,
            "sl_points": 15, "tp_points": 25, "atr": 10, "tp_source": "atr",
            "sl_method": "swing", "sweep_details": None,
        }
        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert result["grade"] in ("B", None)
        assert result["decision"] in ("HALF_SIZE", "WAIT", "NO_TRADE")


# ══════════════════════════════════════════════════════════════
# REGRESSION — Real winning trades must produce the same result
# ══════════════════════════════════════════════════════════════

class TestRegressionWinningTradesChecklist:
    """
    End-to-end regression tests: replicate real winning trade conditions
    through the full checklist pipeline. New filters MUST NOT change the
    outcome for trades the old system correctly identified.
    """

    def test_mar26_short_supply_rejection_full_send(self, engine, sample_news_data):
        """
        Real trade: Mar 26 2026 — SHORT at 6,595 into supply zone.
        Score 4/4, BOS bearish, sweep confirmed, FVG with displacement.
        Old system: FULL_SEND grade A.
        New system MUST: still produce FULL_SEND or HALF_SIZE (never NO_TRADE/WAIT).
        """
        macro = {
            "bias": "SHORT", "vix_bucket": "elevated", "vix_value": 24.13,
            "vix_pct": 3.5, "size_label": "half", "trade_allowed": True,
            "short_only": False, "vix_direction_bias": "SELL_BIAS",
            "us10y_direction": "rising", "oil_direction": "spiking",
            "dxy_direction": "rising", "rut_direction": "red",
            "groq_sentiment": "BEARISH", "bullish_count": 0, "bearish_count": 4,
            "score_contribution": 1,
            "raw_data": {"vix": {"value": 24.13, "pct_change": 3.5}},
        }

        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 28.0, "adx_ranging": False,
            "atr_compressing": False, "atr_ratio": 1.1,
            "range_high": 6620.0, "range_low": 6560.0,
        }

        engine._structure.get_layer2_result.return_value = {
            "structure_bias": "short", "score_contribution": 1,
            "above_ema200": True, "above_ema50": False, "ema_bias": "bearish",
            "bos_direction": "bearish", "bos_level": 6580, "bos_bars_ago": 3,
            "choch_direction": "bearish", "choch_recent": True, "wyckoff": "markdown",
            "ema200_value": 6400, "ema50_value": 6610,
            "ob_bullish_nearby": False, "ob_bearish_nearby": True,
            "bullish_obs": [], "bearish_obs": [{"top": 6610, "bottom": 6600, "type": "bearish"}],
            "range_condition": range_cond,
        }

        engine._zones.get_layer3_result.return_value = {
            "at_zone": True, "zone_level": 6600, "zone_type": "round",
            "distance": 5.0, "zone_direction": "resistance",
            "score_contribution": 1, "all_levels": [
                {"price": 6550.0, "type": "round"},
                {"price": 6568.0, "type": "eql"},
                {"price": 6600.0, "type": "round"},
            ],
            "all_nearby": [],
            "asian_range": {"asian_high": 6590, "asian_low": 6575, "valid": True},
            "liquidity_sweeps": [
                {"level_price": 6610.0, "level_type": "pdh", "sweep_side": "buy_side",
                 "sweep_high": 6615.0, "bars_ago": 8, "favors_direction": "SHORT"},
            ],
            "has_buy_side_sweep": True, "has_sell_side_sweep": False,
        }

        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "sellers", "divergence": "bearish",
            "vix_spiking_now": False, "confirms_bias": True, "score_contribution": 1,
        }

        engine._entry.get_entry_result.return_value = {
            "entry_ready": True, "entry_confidence": "full",
            "fvg_present": True,
            "fvg_details": {"top": 6605, "bottom": 6598, "age_bars": 2,
                            "displacement_valid": True, "direction": "bearish"},
            "displacement_valid": True,
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr_valid": True, "rr": 2.8,
            "has_liquidity_sweep": True,
            "sweep_details": {"level_type": "pdh", "sweep_side": "buy_side", "bars_ago": 8},
            "sl_price": 6615.0, "tp_price": 6560.0, "sl_points": 20.0, "tp_points": 35.0,
            "atr": 12.0, "tp_source": "eql", "sl_method": "swing",
            "ustec_details": {"agrees": True},
        }

        result = engine.run_full_checklist(6595.0, macro, sample_news_data)

        assert result["decision"] in ("FULL_SEND", "HALF_SIZE"), (
            f"Winning SHORT must fire! Got decision={result['decision']}, "
            f"grade={result['grade']}, block={result.get('block_reason')}"
        )
        assert result["direction"] == "SHORT"
        assert result["entry_ready"] is True
        assert result["score"] == 4
        assert result["has_liquidity_sweep"] is True
        assert result["is_ranging"] is False

    def test_trending_short_with_sweep_stays_grade_a(self, engine, sample_news_data):
        """
        Score 4/4 + entry_confidence=full + sweep confirmed + trending
        = FULL_SEND grade A. The new code must preserve this.
        """
        macro = {
            "bias": "SHORT", "vix_bucket": "normal", "vix_value": 18.0,
            "vix_pct": 1.0, "size_label": "normal", "short_only": False,
            "vix_direction_bias": "SELL_BIAS", "score_contribution": 1,
            "us10y_direction": "rising", "oil_direction": "stable",
            "dxy_direction": "rising", "rut_direction": "red",
            "groq_sentiment": "BEARISH", "bullish_count": 0, "bearish_count": 3,
            "raw_data": {"vix": {"value": 18.0, "pct_change": 1.0}},
        }

        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 32.0, "adx_ranging": False,
            "atr_compressing": False, "atr_ratio": 1.2,
            "range_high": 6550.0, "range_low": 6480.0,
        }

        engine._structure.get_layer2_result.return_value = {
            "structure_bias": "short", "score_contribution": 1,
            "above_ema200": True, "above_ema50": False, "ema_bias": "bearish",
            "bos_direction": "bearish", "bos_level": 6510, "bos_bars_ago": 2,
            "choch_direction": None, "choch_recent": False, "wyckoff": "markdown",
            "ema200_value": 6350, "ema50_value": 6530,
            "ob_bullish_nearby": False, "ob_bearish_nearby": True,
            "bullish_obs": [], "bearish_obs": [],
            "range_condition": range_cond,
        }

        engine._zones.get_layer3_result.return_value = {
            "at_zone": True, "zone_level": 6500, "zone_type": "round",
            "distance": 5.0, "zone_direction": "resistance",
            "score_contribution": 1, "all_levels": [],
            "all_nearby": [],
            "asian_range": {"asian_high": 6510, "asian_low": 6490, "valid": True},
            "liquidity_sweeps": [
                {"level_price": 6530.0, "level_type": "eqh", "sweep_side": "buy_side",
                 "sweep_high": 6533.0, "bars_ago": 5, "favors_direction": "SHORT"}
            ],
            "has_buy_side_sweep": True, "has_sell_side_sweep": False,
        }

        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "sellers", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": True, "score_contribution": 1,
        }

        engine._entry.get_entry_result.return_value = {
            "entry_ready": True, "entry_confidence": "full",
            "fvg_present": True, "fvg_details": {"top": 6515, "bottom": 6508},
            "displacement_valid": True,
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr_valid": True, "rr": 2.5,
            "has_liquidity_sweep": True,
            "sweep_details": {"level_type": "eqh", "sweep_side": "buy_side", "bars_ago": 5},
            "sl_price": 6535.0, "tp_price": 6470.0, "sl_points": 30, "tp_points": 35,
            "atr": 12.0, "tp_source": "eql", "sl_method": "swing",
            "ustec_details": {"agrees": True},
        }

        result = engine.run_full_checklist(6505.0, macro, sample_news_data)

        assert result["decision"] == "FULL_SEND", (
            f"4/4 + full confidence + sweep + trending = FULL_SEND! "
            f"Got {result['decision']}, grade={result['grade']}"
        )
        assert result["grade"] == "A"
        assert result["direction"] == "SHORT"

    def test_trending_long_without_sweep_fires_as_half_size(self, engine, sample_news_data):
        """
        Score 4/4 + no sweep + trending = the no_sweep downgrade fires.
        Must still produce a trade (HALF_SIZE grade B), never block it.
        """
        macro = {
            "bias": "LONG", "vix_bucket": "normal", "vix_value": 17.0,
            "vix_pct": -2.0, "size_label": "normal", "short_only": False,
            "vix_direction_bias": "BUY_BIAS", "score_contribution": 1,
            "us10y_direction": "falling", "oil_direction": "falling",
            "dxy_direction": "falling", "rut_direction": "green",
            "groq_sentiment": "BULLISH", "bullish_count": 4, "bearish_count": 0,
            "raw_data": {"vix": {"value": 17.0, "pct_change": -2.0}},
        }

        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 26.0, "adx_ranging": False,
            "atr_compressing": False, "atr_ratio": 1.0,
            "range_high": 6550.0, "range_low": 6480.0,
        }

        engine._structure.get_layer2_result.return_value = {
            "structure_bias": "long", "score_contribution": 1,
            "above_ema200": True, "above_ema50": True, "ema_bias": "bullish",
            "bos_direction": "bullish", "bos_level": 6500, "bos_bars_ago": 2,
            "choch_direction": None, "choch_recent": False, "wyckoff": "markup",
            "ema200_value": 6350, "ema50_value": 6480,
            "ob_bullish_nearby": True, "ob_bearish_nearby": False,
            "bullish_obs": [], "bearish_obs": [],
            "range_condition": range_cond,
        }

        engine._zones.get_layer3_result.return_value = {
            "at_zone": True, "zone_level": 6500, "zone_type": "round",
            "distance": 3.0, "zone_direction": "support",
            "score_contribution": 1, "all_levels": [],
            "all_nearby": [],
            "asian_range": {"asian_high": 6510, "asian_low": 6490, "valid": True},
            "liquidity_sweeps": [],
            "has_buy_side_sweep": False, "has_sell_side_sweep": False,
        }

        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "buyers", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": True, "score_contribution": 1,
        }

        engine._entry.get_entry_result.return_value = {
            "entry_ready": True, "entry_confidence": "no_sweep",
            "fvg_present": True, "fvg_details": {"top": 6510, "bottom": 6505},
            "displacement_valid": True,
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr_valid": True, "rr": 2.5,
            "has_liquidity_sweep": False,
            "sweep_details": None,
            "sl_price": 6485.0, "tp_price": 6540.0, "sl_points": 15, "tp_points": 40,
            "atr": 10.0, "tp_source": "pdh", "sl_method": "swing",
            "ustec_details": {"agrees": True},
        }

        result = engine.run_full_checklist(6500.0, macro, sample_news_data)

        assert result["decision"] in ("FULL_SEND", "HALF_SIZE"), (
            f"Trending LONG without sweep must still fire! "
            f"Got decision={result['decision']}, block={result.get('block_reason')}"
        )
        assert result["entry_ready"] is True
        assert result["direction"] == "LONG"
