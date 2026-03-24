"""Tests for HealthMonitor — component checks, state change detection."""

import pytest
from unittest.mock import MagicMock, patch
from src.health_monitor import HealthMonitor


@pytest.fixture
def monitor(mock_ctrader, mock_alert_bot):
    sig_log = MagicMock()
    sig_log.get_recent_signals.return_value = []
    return HealthMonitor(mock_ctrader, sig_log, alert_bot=mock_alert_bot)


class TestCheckCTrader:
    def test_connected(self, monitor):
        result = monitor.check_ctrader()
        assert result["ok"] is True
        assert result["message"] == "CONNECTED"

    def test_disconnected(self, monitor):
        monitor._ctrader.get_connection_status.return_value = "DISCONNECTED"
        result = monitor.check_ctrader()
        assert result["ok"] is False


class TestCheckDatabase:
    def test_db_check_with_valid_db(self, monitor, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS signals (id INTEGER)")
        with patch("src.health_monitor.os.path.join", return_value=db_path):
            result = monitor.check_database()
            assert isinstance(result.get("ok"), bool)


class TestCheckModelFiles:
    def test_no_models_present(self, monitor, tmp_path):
        with patch("src.health_monitor.os.path.exists", return_value=False):
            result = monitor.check_model_files()
        assert result["model1"] is False
        assert result["model2"] is False
        assert result["model3"] is False
        assert result["ok"] is False

    def test_some_models_present(self, monitor):
        def _exists(path):
            return "model1" in path
        with patch("src.health_monitor.os.path.exists", side_effect=_exists):
            result = monitor.check_model_files()
        assert result["model1"] is True
        assert result["ok"] is True


class TestStateChangeDetection:
    def test_alerts_on_state_change(self, monitor, mock_alert_bot):
        with patch.object(monitor, "check_ctrader", return_value={"ok": True, "message": "CONNECTED"}), \
             patch.object(monitor, "check_yfinance", return_value={"ok": True, "message": "OK"}), \
             patch.object(monitor, "check_calendar_api", return_value={"ok": True}), \
             patch.object(monitor, "check_database", return_value={"ok": True}), \
             patch.object(monitor, "check_model_files", return_value={"ok": True, "model1": True, "model2": True, "model3": True, "model3_age_days": 0}), \
             patch.object(monitor, "check_signal_frequency", return_value={"ok": True, "hours_since_last": 0}):
            monitor.run_health_check()

        mock_alert_bot.send_system_alert.reset_mock()

        with patch.object(monitor, "check_ctrader", return_value={"ok": True, "message": "CONNECTED"}), \
             patch.object(monitor, "check_yfinance", return_value={"ok": False, "message": "Timeout"}), \
             patch.object(monitor, "check_calendar_api", return_value={"ok": True}), \
             patch.object(monitor, "check_database", return_value={"ok": True}), \
             patch.object(monitor, "check_model_files", return_value={"ok": True, "model1": True, "model2": True, "model3": True, "model3_age_days": 0}), \
             patch.object(monitor, "check_signal_frequency", return_value={"ok": True, "hours_since_last": 0}):
            monitor.run_health_check()

        mock_alert_bot.send_system_alert.assert_called()
        call_args = mock_alert_bot.send_system_alert.call_args_list
        assert any("yfinance" in str(c) for c in call_args)

    def test_no_alert_on_same_state(self, monitor, mock_alert_bot):
        checks = {
            "check_ctrader": {"ok": True, "message": "OK"},
            "check_yfinance": {"ok": True, "message": "OK"},
            "check_calendar_api": {"ok": True},
            "check_database": {"ok": True},
            "check_model_files": {"ok": True, "model1": True, "model2": True, "model3": True, "model3_age_days": 0},
            "check_signal_frequency": {"ok": True, "hours_since_last": 0},
        }
        with patch.multiple(monitor, **{k: MagicMock(return_value=v) for k, v in checks.items()}):
            monitor.run_health_check()
            mock_alert_bot.send_system_alert.reset_mock()
            monitor.run_health_check()

        mock_alert_bot.send_system_alert.assert_not_called()
