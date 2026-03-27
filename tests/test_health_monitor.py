"""Tests for HealthMonitor — component checks, state change detection."""

import pytest
from datetime import datetime, timezone, timedelta
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


# ─── Signal frequency ────────────────────────────────────


class TestCheckSignalFrequency:
    """Tests for check_signal_frequency — trading-hours gating and staleness."""

    def _mock_now(self, fixed_utc):
        def _now(tz=None):
            return fixed_utc.astimezone(tz) if tz else fixed_utc
        return _now

    def test_alerts_when_no_signals_during_trading(self, monitor):
        fixed_utc = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        five_hours_ago = (fixed_utc - timedelta(hours=5)).isoformat()
        monitor._signal_logger.get_last_signal_timestamp.return_value = five_hours_ago

        with patch("src.health_monitor.datetime") as mock_dt:
            mock_dt.now = self._mock_now(fixed_utc)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = monitor.check_signal_frequency()

        assert result["ok"] is False
        assert result["hours_since_last"] >= 4

    def test_no_alert_outside_trading_hours(self, monitor):
        fixed_utc = datetime(2026, 3, 28, 22, 0, 0, tzinfo=timezone.utc)

        with patch("src.health_monitor.datetime") as mock_dt:
            mock_dt.now = self._mock_now(fixed_utc)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = monitor.check_signal_frequency()

        assert result["ok"] is True
        assert result["hours_since_last"] == 0

    def test_no_alert_with_recent_signals(self, monitor):
        fixed_utc = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        thirty_min_ago = (fixed_utc - timedelta(minutes=30)).isoformat()
        monitor._signal_logger.get_last_signal_timestamp.return_value = thirty_min_ago

        with patch("src.health_monitor.datetime") as mock_dt:
            mock_dt.now = self._mock_now(fixed_utc)
            mock_dt.fromisoformat = datetime.fromisoformat
            result = monitor.check_signal_frequency()

        assert result["ok"] is True
        assert result["hours_since_last"] < 1


# ─── yfinance probe ──────────────────────────────────────


class TestCheckYfinance:
    def test_healthy_when_vix_available(self, monitor):
        mock_ticker = MagicMock()
        mock_ticker.fast_info.last_price = 18.5
        with patch("src.health_monitor.yf.Ticker", return_value=mock_ticker):
            result = monitor.check_yfinance()
        assert result["ok"] is True
        assert "VIX" in result["message"]

    def test_unhealthy_when_vix_fails(self, monitor):
        with patch("src.health_monitor.yf.Ticker", side_effect=Exception("API timeout")):
            result = monitor.check_yfinance()
        assert result["ok"] is False
        assert "API timeout" in result["message"]


# ─── get_system_status ────────────────────────────────────


class TestGetSystemStatus:
    _ALL_OK = {
        "check_ctrader": {"ok": True, "message": "CONNECTED"},
        "check_yfinance": {"ok": True, "message": "VIX=18"},
        "check_calendar_api": {"ok": True},
        "check_database": {"ok": True},
        "check_model_files": {"ok": True, "model1": True, "model2": True, "model3": True, "model3_age_days": 0},
        "check_signal_frequency": {"ok": True, "hours_since_last": 0},
    }

    def test_returns_all_required_keys(self, monitor):
        with patch.multiple(monitor, **{k: MagicMock(return_value=v) for k, v in self._ALL_OK.items()}):
            status = monitor.get_system_status()
        for key in ("ctrader", "yfinance", "calendar", "database", "models", "signal_freq"):
            assert key in status, f"Missing key: {key}"

    def test_status_values_are_dicts(self, monitor):
        with patch.multiple(monitor, **{k: MagicMock(return_value=v) for k, v in self._ALL_OK.items()}):
            status = monitor.get_system_status()
        for key, val in status.items():
            assert isinstance(val, dict), f"{key} should be a dict, got {type(val)}"


# ─── monitor_loop ─────────────────────────────────────────


class TestMonitorLoop:
    _HEALTHY = {
        "check_ctrader": {"ok": True, "message": "CONNECTED"},
        "check_yfinance": {"ok": True, "message": "OK"},
        "check_calendar_api": {"ok": True},
        "check_database": {"ok": True},
        "check_model_files": {"ok": True, "model1": True, "model2": True, "model3": True, "model3_age_days": 0},
        "check_signal_frequency": {"ok": True, "hours_since_last": 0},
    }

    def test_calls_all_checks(self, monitor):
        with patch.object(monitor, "check_ctrader", return_value={"ok": True, "message": "OK"}) as m1, \
             patch.object(monitor, "check_yfinance", return_value={"ok": True, "message": "OK"}) as m2, \
             patch.object(monitor, "check_calendar_api", return_value={"ok": True}) as m3, \
             patch.object(monitor, "check_database", return_value={"ok": True}) as m4, \
             patch.object(monitor, "check_model_files", return_value={"ok": True, "model1": True, "model2": True, "model3": True, "model3_age_days": 0}) as m5, \
             patch.object(monitor, "check_signal_frequency", return_value={"ok": True, "hours_since_last": 0}) as m6:
            monitor.monitor_loop()
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()
        m4.assert_called_once()
        m5.assert_called_once()
        m6.assert_called_once()

    def test_sends_alert_on_state_change(self, monitor, mock_alert_bot):
        unhealthy = dict(self._HEALTHY)
        unhealthy["check_yfinance"] = {"ok": False, "message": "Timeout"}

        with patch.multiple(monitor, **{k: MagicMock(return_value=v) for k, v in self._HEALTHY.items()}):
            monitor.monitor_loop()
        mock_alert_bot.send_system_alert.reset_mock()

        with patch.multiple(monitor, **{k: MagicMock(return_value=v) for k, v in unhealthy.items()}):
            monitor.monitor_loop()
        mock_alert_bot.send_system_alert.assert_called()
        assert any("yfinance" in str(c) for c in mock_alert_bot.send_system_alert.call_args_list)


# ─── record_error ─────────────────────────────────────────


class TestRecordError:
    def test_increments_error_count(self, monitor):
        with patch("src.health_monitor.logger") as mock_logger:
            monitor.record_error("timeout")
            monitor.record_error("timeout")
        assert mock_logger.warning.call_count == 2
