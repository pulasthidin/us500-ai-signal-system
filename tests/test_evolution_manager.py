"""Tests for EvolutionManager — stage progression, no-downgrade, Telegram alerts."""

import json
import os
import pytest
from unittest.mock import MagicMock, call

from src.evolution_manager import EvolutionManager, STAGE_FILE, STAGE_THRESHOLDS


@pytest.fixture
def evo(tmp_path, mock_alert_bot):
    sig_log = MagicMock()
    trainer = MagicMock()
    trainer.train_meta_label_model.return_value = True
    mgr = EvolutionManager(sig_log, trainer, alert_bot=mock_alert_bot)
    stage_file = str(tmp_path / "stage.json")
    import src.evolution_manager as mod
    mod.STAGE_FILE = stage_file
    return mgr, sig_log, trainer, mock_alert_bot, stage_file


class TestStageFile:
    def test_default_stage_when_no_file(self, evo):
        mgr, *_ = evo
        assert mgr.get_current_stage() == 0

    def test_load_after_save(self, evo):
        mgr, _, _, _, path = evo
        mgr._save_stage({"stage": 2, "pattern_scanner_active": True, "daily_training_active": False})
        assert mgr.get_current_stage() == 2

    def test_get_stage_info_returns_full_dict(self, evo):
        mgr, *_ = evo
        info = mgr.get_stage_info()
        assert "stage" in info
        assert "pattern_scanner_active" in info
        assert "daily_training_active" in info


class TestStageProgression:
    def test_stays_stage_0_under_200(self, evo):
        mgr, sig_log, _, _, _ = evo
        sig_log.get_signal_count.return_value = 150
        result = mgr.check_evolution_stage()
        assert result["stage"] == 0

    def test_activates_stage_1_at_200(self, evo):
        mgr, sig_log, trainer, bot, _ = evo
        sig_log.get_signal_count.return_value = 200
        result = mgr.check_evolution_stage()
        assert result["stage"] == 1
        trainer.train_meta_label_model.assert_called_once()
        bot.send_system_alert.assert_called()
        stage1_calls = [c for c in bot.send_system_alert.call_args_list if "STAGE 1" in str(c)]
        assert len(stage1_calls) == 1

    def test_activates_stage_2_at_500(self, evo):
        mgr, sig_log, _, bot, path = evo
        mgr._save_stage({"stage": 1, "pattern_scanner_active": False, "daily_training_active": False})
        sig_log.get_signal_count.return_value = 500
        result = mgr.check_evolution_stage()
        assert result["stage"] == 2
        info = mgr.get_stage_info()
        assert info["pattern_scanner_active"] is True

    def test_activates_stage_3_at_1000(self, evo):
        mgr, sig_log, _, _, path = evo
        mgr._save_stage({"stage": 2, "pattern_scanner_active": True, "daily_training_active": False})
        sig_log.get_signal_count.return_value = 1000
        result = mgr.check_evolution_stage()
        assert result["stage"] == 3
        info = mgr.get_stage_info()
        assert info["daily_training_active"] is True
        assert info["pattern_scanner_active"] is True


class TestStageNeverDowngrades:
    def test_stage_2_stays_if_count_drops(self, evo):
        mgr, sig_log, _, _, _ = evo
        mgr._save_stage({"stage": 2, "pattern_scanner_active": True, "daily_training_active": False})
        sig_log.get_signal_count.return_value = 100
        result = mgr.check_evolution_stage()
        assert result["stage"] == 2


class TestJumpMultipleStages:
    def test_jump_0_to_3_cascades_all_activations(self, evo):
        mgr, sig_log, trainer, bot, _ = evo
        sig_log.get_signal_count.return_value = 1200
        result = mgr.check_evolution_stage()
        assert result["stage"] == 3
        trainer.train_meta_label_model.assert_called_once()
        stage_alerts = [c for c in bot.send_system_alert.call_args_list if "STAGE" in str(c)]
        assert len(stage_alerts) == 3

    def test_jump_0_to_2_cascades_stages_1_and_2(self, evo):
        mgr, sig_log, trainer, bot, _ = evo
        sig_log.get_signal_count.return_value = 600
        result = mgr.check_evolution_stage()
        assert result["stage"] == 2
        trainer.train_meta_label_model.assert_called_once()
        stage_alerts = [c for c in bot.send_system_alert.call_args_list if "STAGE" in str(c)]
        assert len(stage_alerts) == 2


class TestEvolutionReport:
    def test_run_evolution_check_sends_status(self, evo):
        mgr, sig_log, _, bot, _ = evo
        sig_log.get_signal_count.return_value = 150
        result = mgr.run_evolution_check()
        assert result["signal_count"] == 150
        status_calls = [c for c in bot.send_system_alert.call_args_list if "EVOLUTION STATUS" in str(c)]
        assert len(status_calls) == 1

    def test_report_shows_next_threshold(self, evo):
        mgr, sig_log, _, bot, _ = evo
        sig_log.get_signal_count.return_value = 50
        mgr.run_evolution_check()
        last_call_msg = str(bot.send_system_alert.call_args_list[-1])
        assert "200" in last_call_msg
        assert "150" in last_call_msg

    def test_report_shows_max_stage_reached(self, evo):
        mgr, sig_log, _, bot, _ = evo
        mgr._save_stage({"stage": 3, "pattern_scanner_active": True, "daily_training_active": True})
        sig_log.get_signal_count.return_value = 1500
        mgr.run_evolution_check()
        last_call_msg = str(bot.send_system_alert.call_args_list[-1])
        assert "Maximum stage" in last_call_msg


class TestTrainingFailureSafe:
    def test_stage_stays_0_if_training_fails(self, evo):
        mgr, sig_log, trainer, bot, _ = evo
        trainer.train_meta_label_model.side_effect = Exception("GPU OOM")
        sig_log.get_signal_count.return_value = 250
        result = mgr.check_evolution_stage()
        assert result["stage"] == 0
        warn_calls = [c for c in bot.send_system_alert.call_args_list if "training error" in str(c)]
        assert len(warn_calls) == 1
