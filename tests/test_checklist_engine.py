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
    def test_macro_and_structure_agree_short(self, engine):
        layer1 = {"bias": "SHORT", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "short", "ema_bias": "bearish"}
        assert engine._derive_direction(layer1, layer2) == "SHORT"

    def test_macro_and_structure_agree_long(self, engine):
        layer1 = {"bias": "LONG", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "long", "ema_bias": "bullish"}
        assert engine._derive_direction(layer1, layer2) == "LONG"

    def test_vix_tiebreak_short(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "SELL_BIAS"}
        layer2 = {"structure_bias": "short", "ema_bias": "bearish"}
        assert engine._derive_direction(layer1, layer2) == "SHORT"

    def test_vix_tiebreak_long(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "BUY_BIAS"}
        layer2 = {"structure_bias": "long", "ema_bias": "bullish"}
        assert engine._derive_direction(layer1, layer2) == "LONG"

    def test_fallback_ema_bearish_plus_vix_sell(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "SELL_BIAS"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bearish"}
        assert engine._derive_direction(layer1, layer2) == "SHORT"

    def test_fallback_ema_bullish_plus_macro_long(self, engine):
        layer1 = {"bias": "LONG", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bullish"}
        assert engine._derive_direction(layer1, layer2) == "LONG"

    def test_fallback_ema_bearish_plus_macro_short(self, engine):
        layer1 = {"bias": "SHORT", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bearish"}
        assert engine._derive_direction(layer1, layer2) == "SHORT"

    def test_fallback_ema_bullish_plus_vix_buy(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "BUY_BIAS"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bullish"}
        assert engine._derive_direction(layer1, layer2) == "LONG"

    def test_no_direction_when_all_unclear(self, engine):
        layer1 = {"bias": "MIXED", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "unclear"}
        assert engine._derive_direction(layer1, layer2) is None

    def test_no_direction_when_ema_conflicts_with_macro(self, engine):
        layer1 = {"bias": "LONG", "vix_direction_bias": "NEUTRAL"}
        layer2 = {"structure_bias": "unclear", "ema_bias": "bearish"}
        assert engine._derive_direction(layer1, layer2) is None

    def test_no_direction_when_ema_missing(self, engine):
        layer1 = {"bias": "SHORT", "vix_direction_bias": "SELL_BIAS"}
        layer2 = {"structure_bias": "unclear"}
        assert engine._derive_direction(layer1, layer2) is None


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
