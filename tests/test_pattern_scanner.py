"""Tests for PatternScanner — activation gate, similarity, dedup, early warnings."""

import json
import os
import time
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.pattern_scanner import (
    PatternScanner,
    STAGE_FILE,
    PATTERN_WIN_PROB_THRESHOLD,
    PATTERN_MIN_SIMILAR_WINS,
    PATTERN_DEDUP_MINUTES,
)


@pytest.fixture
def scanner(tmp_path, mock_alert_bot):
    macro = MagicMock()
    macro.get_layer1_result.return_value = {
        "bias": "LONG", "vix_value": 17, "vix_pct": -2.5, "vix_bucket": "normal",
        "vix_direction_bias": "BUY_BIAS", "us10y_direction": "falling",
        "oil_direction": "falling", "dxy_direction": "flat", "rut_direction": "green",
        "groq_sentiment": "NEUTRAL", "bullish_count": 3, "bearish_count": 0,
    }
    zones = MagicMock()
    zones.get_layer3_result.return_value = {
        "at_zone": True, "zone_type": "round", "zone_level": 6500.0,
        "distance": 3.0, "zone_direction": "support",
    }
    flow = MagicMock()
    flow.get_layer4_result.return_value = {
        "delta_direction": "buyers", "divergence": "none",
        "vix_spiking_now": False, "confirms_bias": True,
    }
    checklist = MagicMock()
    checklist.get_current_session.return_value = "London"
    checklist.get_day_flags.return_value = {
        "day_name": "Tuesday", "is_monday": False, "is_friday": False,
    }
    checklist.is_trading_session.return_value = True

    ctrader = MagicMock()
    ctrader.get_current_price.return_value = 6500.0

    sig_log = MagicMock()
    sig_log.get_training_data.return_value = pd.DataFrame()
    sig_log.log_pattern_alert.return_value = 1
    sig_log.is_duplicate.return_value = False

    model_pred = MagicMock()
    model_pred.models = {"model3": True}
    model_pred.predict_win_probability.return_value = 0.85
    model_pred.get_shap_explanation.return_value = ["VIX falling", "At round level", "London session"]

    stage_file = str(tmp_path / "stage.json")
    import src.pattern_scanner as mod
    mod.STAGE_FILE = stage_file

    ps = PatternScanner(
        macro_checker=macro, zone_calculator=zones,
        orderflow_analyzer=flow, checklist_engine=checklist,
        ctrader_connection=ctrader, us500_id=1,
        signal_logger=sig_log, model_predictor=model_pred,
        alert_bot=mock_alert_bot,
    )
    return ps, sig_log, model_pred, mock_alert_bot, stage_file


class TestIsActive:
    def test_inactive_when_no_stage_file(self, scanner):
        ps, *_ = scanner
        assert ps.is_active() is False

    def test_inactive_at_stage_0(self, scanner):
        ps, _, _, _, path = scanner
        with open(path, "w") as f:
            json.dump({"stage": 0}, f)
        assert ps.is_active() is False

    def test_inactive_at_stage_1(self, scanner):
        ps, _, _, _, path = scanner
        with open(path, "w") as f:
            json.dump({"stage": 1}, f)
        assert ps.is_active() is False

    def test_active_at_stage_2(self, scanner):
        ps, _, _, _, path = scanner
        with open(path, "w") as f:
            json.dump({"stage": 2}, f)
        assert ps.is_active() is True

    def test_active_at_stage_3(self, scanner):
        ps, _, _, _, path = scanner
        with open(path, "w") as f:
            json.dump({"stage": 3}, f)
        assert ps.is_active() is True


class TestMarketSnapshot:
    def test_returns_all_required_keys(self, scanner):
        ps, *_ = scanner
        snap = ps.get_current_market_snapshot()
        assert snap is not None
        for key in ["current_price", "session", "macro_bias", "vix_level",
                     "at_zone", "delta_direction", "direction"]:
            assert key in snap, f"Missing key: {key}"

    def test_returns_none_when_price_unavailable(self, scanner):
        ps, *_ = scanner
        ps._ctrader.get_current_price.return_value = None
        assert ps.get_current_market_snapshot() is None

    def test_direction_set_from_macro_bias(self, scanner):
        ps, *_ = scanner
        snap = ps.get_current_market_snapshot()
        assert snap["direction"] == "LONG"

    def test_direction_none_when_mixed(self, scanner):
        ps, *_ = scanner
        ps._macro.get_layer1_result.return_value["bias"] = "MIXED"
        snap = ps.get_current_market_snapshot()
        assert snap["direction"] is None


class TestPatternSimilarity:
    def test_returns_win_probability(self, scanner):
        ps, sig_log, model_pred, _, _ = scanner
        sig_log.get_training_data.return_value = pd.DataFrame({
            "outcome_label": [1, 1, 1],
            "macro_bias": ["LONG", "LONG", "SHORT"],
            "vix_bucket": ["normal", "normal", "elevated"],
            "delta_direction": ["buyers", "buyers", "sellers"],
            "at_zone": [1, 1, 0],
            "session": ["London", "London", "NY_Session"],
            "id": [1, 2, 3],
            "direction": ["LONG", "LONG", "SHORT"],
            "pnl_points": [20, 15, 25],
        })
        snap = ps.get_current_market_snapshot()
        result = ps.calculate_pattern_similarity(snap)
        assert result["win_probability"] == 0.85

    def test_returns_zero_when_model_missing(self, scanner):
        ps, _, model_pred, _, _ = scanner
        model_pred.models = {}
        snap = ps.get_current_market_snapshot()
        result = ps.calculate_pattern_similarity(snap)
        assert result["win_probability"] == 0.0
        assert result["similar_signals_count"] == 0

    def test_counts_similar_wins(self, scanner):
        ps, sig_log, _, _, _ = scanner
        sig_log.get_training_data.return_value = pd.DataFrame({
            "outcome_label": [1] * 15,
            "macro_bias": ["LONG"] * 15,
            "vix_bucket": ["normal"] * 15,
            "delta_direction": ["buyers"] * 15,
            "at_zone": [1] * 15,
            "session": ["London"] * 15,
            "id": list(range(15)),
            "direction": ["LONG"] * 15,
            "pnl_points": [20] * 15,
        })
        snap = ps.get_current_market_snapshot()
        result = ps.calculate_pattern_similarity(snap)
        assert result["similar_signals_count"] == 15


class TestShouldSendEarlyWarning:
    def _make_snapshot(self, direction="LONG"):
        return {"direction": direction}

    def test_true_when_all_conditions_met(self, scanner):
        ps, *_ = scanner
        sim = {"win_probability": 0.90, "similar_signals_count": 20}
        assert ps.should_send_early_warning(sim, self._make_snapshot()) is True

    def test_false_when_probability_too_low(self, scanner):
        ps, *_ = scanner
        sim = {"win_probability": 0.60, "similar_signals_count": 20}
        assert ps.should_send_early_warning(sim, self._make_snapshot()) is False

    def test_false_when_not_enough_similar(self, scanner):
        ps, *_ = scanner
        sim = {"win_probability": 0.90, "similar_signals_count": 5}
        assert ps.should_send_early_warning(sim, self._make_snapshot()) is False

    def test_false_outside_trading_session(self, scanner):
        ps, *_ = scanner
        ps._checklist.is_trading_session.return_value = False
        sim = {"win_probability": 0.90, "similar_signals_count": 20}
        assert ps.should_send_early_warning(sim, self._make_snapshot()) is False

    def test_dedup_blocks_same_direction(self, scanner):
        ps, *_ = scanner
        ps._last_alert_time = time.time()
        ps._last_alert_direction = "LONG"
        sim = {"win_probability": 0.90, "similar_signals_count": 20}
        assert ps.should_send_early_warning(sim, self._make_snapshot("LONG")) is False

    def test_dedup_allows_different_direction(self, scanner):
        ps, *_ = scanner
        ps._last_alert_time = time.time()
        ps._last_alert_direction = "SHORT"
        sim = {"win_probability": 0.90, "similar_signals_count": 20}
        assert ps.should_send_early_warning(sim, self._make_snapshot("LONG")) is True

    def test_dedup_allows_after_timeout(self, scanner):
        ps, *_ = scanner
        ps._last_alert_time = time.time() - (PATTERN_DEDUP_MINUTES * 60 + 1)
        ps._last_alert_direction = "LONG"
        sim = {"win_probability": 0.90, "similar_signals_count": 20}
        assert ps.should_send_early_warning(sim, self._make_snapshot("LONG")) is True


class TestSendEarlyWarning:
    def test_sends_and_logs(self, scanner):
        ps, sig_log, _, bot, _ = scanner
        snap = ps.get_current_market_snapshot()
        sim = {"win_probability": 0.85, "similar_signals_count": 20,
               "top_matching_features": ["VIX falling", "At round"], "nearest_historical_win": None}
        ps.send_early_warning_alert(snap, sim)
        sig_log.log_pattern_alert.assert_called_once()
        bot.send_trade_message.assert_called_once()
        msg = bot.send_trade_message.call_args[0][0]
        assert "PATTERN FORMING" in msg
        assert "85%" in msg

    def test_updates_dedup_state(self, scanner):
        ps, _, _, _, _ = scanner
        snap = ps.get_current_market_snapshot()
        sim = {"win_probability": 0.85, "similar_signals_count": 20,
               "top_matching_features": [], "nearest_historical_win": None}
        ps.send_early_warning_alert(snap, sim)
        assert ps._last_alert_direction == "LONG"
        assert ps._last_alert_time > 0


class TestRunPatternScan:
    def test_silent_when_inactive(self, scanner):
        ps, sig_log, _, _, _ = scanner
        ps.run_pattern_scan()
        sig_log.log_pattern_alert.assert_not_called()

    def test_silent_outside_session(self, scanner):
        ps, sig_log, _, _, path = scanner
        with open(path, "w") as f:
            json.dump({"stage": 2}, f)
        ps._checklist.is_trading_session.return_value = False
        ps.run_pattern_scan()
        sig_log.log_pattern_alert.assert_not_called()


class TestTrackOutcome:
    def test_matches_pattern_to_signal(self, scanner):
        ps, sig_log, _, _, _ = scanner
        now = datetime.now(timezone.utc)
        pa_ts = (now - timedelta(minutes=10)).isoformat()
        sig_ts = now.isoformat()

        sig_log.get_recent_pattern_alerts.return_value = [
            {"id": 1, "timestamp": pa_ts, "direction": "LONG", "matched_signal_id": None}
        ]
        sig_log.get_recent_signals.return_value = [
            {"id": 100, "timestamp": sig_ts, "direction": "LONG"}
        ]

        result = ps.track_pattern_alert_outcome()
        assert result is not None
        assert result["pattern_alert_id"] == 1
        assert result["signal_id"] == 100
        assert 9 <= result["minutes_early"] <= 11
        sig_log.update_pattern_alert_match.assert_called_with(1, 100)

    def test_no_match_if_different_direction(self, scanner):
        ps, sig_log, _, _, _ = scanner
        now = datetime.now(timezone.utc)
        pa_ts = (now - timedelta(minutes=10)).isoformat()
        sig_ts = now.isoformat()

        sig_log.get_recent_pattern_alerts.return_value = [
            {"id": 1, "timestamp": pa_ts, "direction": "LONG", "matched_signal_id": None}
        ]
        sig_log.get_recent_signals.return_value = [
            {"id": 100, "timestamp": sig_ts, "direction": "SHORT"}
        ]

        assert ps.track_pattern_alert_outcome() is None

    def test_no_match_if_too_old(self, scanner):
        ps, sig_log, _, _, _ = scanner
        now = datetime.now(timezone.utc)
        pa_ts = (now - timedelta(minutes=45)).isoformat()
        sig_ts = now.isoformat()

        sig_log.get_recent_pattern_alerts.return_value = [
            {"id": 1, "timestamp": pa_ts, "direction": "LONG", "matched_signal_id": None}
        ]
        sig_log.get_recent_signals.return_value = [
            {"id": 100, "timestamp": sig_ts, "direction": "LONG"}
        ]

        assert ps.track_pattern_alert_outcome() is None

    def test_skips_already_matched_alerts(self, scanner):
        ps, sig_log, _, _, _ = scanner
        sig_log.get_recent_pattern_alerts.return_value = [
            {"id": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
             "direction": "LONG", "matched_signal_id": 50}
        ]
        sig_log.get_recent_signals.return_value = [
            {"id": 100, "timestamp": datetime.now(timezone.utc).isoformat(), "direction": "LONG"}
        ]

        assert ps.track_pattern_alert_outcome() is None

    def test_returns_none_when_no_alerts(self, scanner):
        ps, sig_log, _, _, _ = scanner
        sig_log.get_recent_pattern_alerts.return_value = []
        assert ps.track_pattern_alert_outcome() is None
