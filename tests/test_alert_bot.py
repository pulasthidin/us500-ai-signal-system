"""Tests for AlertBot — message formatting (no real Telegram calls)."""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("TELEGRAM_TRADE_TOKEN", "")
os.environ.setdefault("TELEGRAM_TRADE_CHAT_ID", "")
os.environ.setdefault("TELEGRAM_SYSTEM_TOKEN", "")
os.environ.setdefault("TELEGRAM_SYSTEM_CHAT_ID", "")

from src.alert_bot import AlertBot


@pytest.fixture
def bot():
    with patch("src.alert_bot.telegram"):
        return AlertBot()


def _full_send_result():
    return {
        "decision": "FULL_SEND",
        "direction": "LONG",
        "current_price": 6544.50,
        "score": 4,
        "size_label": "full",
        "vix_bucket": "normal",
        "session": "London",
        "sl_time": "14:23 SL",
        "day_name": "Tuesday",
        "caution_flags": [],
        "layer1": {"bias": "LONG", "vix_pct": -3.2, "vix_value": 17.0, "vix_bucket": "normal"},
        "layer2": {"bos_direction": "bullish"},
        "layer3": {"at_zone": True, "zone_type": "pdl", "zone_level": 6544.0, "distance": 1.5},
        "layer4": {"delta_direction": "buyers", "divergence": "none"},
        "entry": {
            "fvg_present": True, "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr_valid": True, "entry_ready": True,
            "sl_price": 6532.00, "tp_price": 6565.00,
            "rr": 2.04, "sl_points": 12.5, "tp_points": 20.5,
            "fvg_details": {},
        },
    }


class TestFormatTradeAlert:
    def test_full_send_contains_direction(self, bot):
        msg = bot.format_trade_alert(_full_send_result())
        assert "LONG" in msg
        assert "FULL SEND" in msg

    def test_full_send_contains_prices(self, bot):
        msg = bot.format_trade_alert(_full_send_result())
        assert "6,544.50" in msg
        assert "6,532.00" in msg
        assert "6,565.00" in msg

    def test_full_send_contains_score(self, bot):
        msg = bot.format_trade_alert(_full_send_result())
        assert "4/4" in msg

    def test_full_send_contains_layers(self, bot):
        msg = bot.format_trade_alert(_full_send_result())
        assert "L1:" in msg
        assert "L2:" in msg
        assert "L3:" in msg
        assert "L4:" in msg

    def test_half_size_format(self, bot):
        r = _full_send_result()
        r["decision"] = "HALF_SIZE"
        msg = bot.format_trade_alert(r)
        assert "HALF SIZE" in msg

    def test_ml_enhancement_shown(self, bot):
        ml = {"win_probability": 0.74, "shap_top_features": ["VIX falling", "FVG present", "London session"]}
        msg = bot.format_trade_alert(_full_send_result(), ml)
        assert "74%" in msg
        assert "VIX falling" in msg


class TestFormatHardStop:
    def test_hard_stop(self, bot):
        r = {"decision": "HARD_STOP", "layer1": {"vix_value": 31.2}}
        msg = bot.format_trade_alert(r)
        assert "HARD STOP" in msg
        assert "31.2" in msg


class TestFormatWait:
    def test_wait_format(self, bot):
        r = {
            "decision": "WAIT", "direction": "LONG", "score": 3,
            "session": "London", "sl_time": "14:23 SL",
            "layer3": {"zone_level": 6600},
            "entry": {"fvg_present": False, "m5_bos_confirmed": True,
                      "ustec_agrees": True, "rr_valid": True},
        }
        msg = bot.format_trade_alert(r)
        assert "WAIT" in msg
        assert "FVG" in msg


class TestFormatMorningBrief:
    def test_morning_brief_contains_key_sections(self, bot):
        macro = {
            "bias": "LONG", "vix_bucket": "normal",
            "vix_value": 17.5, "vix_pct": -2.5,
            "raw_data": {
                "vix": {"value": 17.5, "pct_change": -2.5},
                "dxy": {"value": 99.5, "pct_change": 0.1},
                "us10y": {"value": 4.3, "pct_change": -0.1},
                "oil": {"value": 72.0, "pct_change": -0.8},
                "rut": {"pct_change": 0.5},
            },
        }
        news = {"events_today": [], "is_news_day": False}
        zones = [{"price": 6500, "type": "round"}, {"price": 6550, "type": "round"}]
        msg = bot.format_morning_brief(macro, news, zones)
        assert "DAY BIAS" in msg
        assert "KEY LEVELS" in msg
        assert "NEWS" in msg
        assert "SESSIONS" in msg


class TestFormatSystemAlert:
    def test_warning(self, bot):
        msg = bot.format_system_alert("WARNING", "yfinance", "All retries failed")
        assert "WARNING" in msg
        assert "yfinance" in msg

    def test_critical(self, bot):
        msg = bot.format_system_alert("CRITICAL", "ctrader", "Connection lost")
        assert "CRITICAL" in msg
        assert "Manual check needed" in msg

    def test_info(self, bot):
        msg = bot.format_system_alert("INFO", "startup", "App started")
        assert "INFO" in msg


class TestResistanceSupportOrder:
    def test_resistance_above_support(self, bot):
        zones = [
            {"price": 6400, "type": "round"},
            {"price": 6450, "type": "pdl"},
            {"price": 6500, "type": "round"},
            {"price": 6550, "type": "round"},
            {"price": 6600, "type": "pdh"},
            {"price": 6650, "type": "round"},
        ]
        macro = {"bias": "LONG", "vix_bucket": "normal", "vix_value": 17.5, "vix_pct": -2.5,
                 "raw_data": {"vix": {"value": 17.5, "pct_change": -2.5},
                              "dxy": {"value": 99, "pct_change": 0}, "us10y": {"value": 4, "pct_change": 0},
                              "oil": {"value": 70, "pct_change": 0}, "rut": {"pct_change": 0}}}
        msg = bot.format_morning_brief(macro, {"events_today": []}, zones)
        res_line = [l for l in msg.split("\n") if l.startswith("Res:")][0]
        sup_line = [l for l in msg.split("\n") if l.startswith("Sup:")][0]
        assert "6,650" in res_line or "6,600" in res_line
        assert "6,400" in sup_line or "6,450" in sup_line


class TestPatternScannerLine:
    def test_pattern_called_line_appears(self, bot):
        r = _full_send_result()
        r["pattern_called_minutes_ago"] = 23.4
        msg = bot.format_trade_alert(r)
        assert "Pattern scanner called this 23 min ago" in msg

    def test_no_pattern_line_when_absent(self, bot):
        msg = bot.format_trade_alert(_full_send_result())
        assert "Pattern scanner" not in msg


class TestWeeklyReportEvolution:
    def test_includes_evolution_section(self, bot):
        stats = {
            "win_rate": 60.0, "by_session": {}, "by_grade": {},
            "evolution": {"stage": 2, "signal_count": 550},
        }
        msg = bot.format_weekly_report(stats)
        assert "EVOLUTION:" in msg
        assert "Stage: 2" in msg
        assert "550" in msg

    def test_includes_pattern_scanner_section(self, bot):
        stats = {
            "win_rate": 60.0, "by_session": {}, "by_grade": {},
            "pattern_accuracy": {
                "total_alerts": 25, "match_rate": 40.0,
                "win_rate_after_match": 75.0,
            },
        }
        msg = bot.format_weekly_report(stats)
        assert "PATTERN SCANNER:" in msg
        assert "25" in msg
        assert "40%" in msg

    def test_omits_pattern_section_when_no_alerts(self, bot):
        stats = {
            "win_rate": 60.0, "by_session": {}, "by_grade": {},
            "pattern_accuracy": {"total_alerts": 0, "match_rate": 0, "win_rate_after_match": 0},
        }
        msg = bot.format_weekly_report(stats)
        assert "PATTERN SCANNER:" not in msg


# ══════════════════════════════════════════════════════════════
# NEW TEST CLASSES — send behaviour, dedup, error handling
# ══════════════════════════════════════════════════════════════

class TestSystemAlertDedup:
    """send_system_alert deduplicates identical alerts within the time window."""

    @pytest.fixture
    def bot_with_system(self):
        with patch("src.alert_bot.telegram"):
            b = AlertBot()
            b._system_bot = MagicMock()
            b._system_chat_id = "123"
            b._run_async = MagicMock()
            return b

    def test_same_message_suppressed_within_window(self, bot_with_system):
        bot_with_system.send_system_alert("WARNING", "yfinance", "All retries failed")
        bot_with_system.send_system_alert("WARNING", "yfinance", "All retries failed")
        assert bot_with_system._run_async.call_count == 1

    def test_different_messages_both_sent(self, bot_with_system):
        bot_with_system.send_system_alert("WARNING", "yfinance", "All retries failed")
        bot_with_system.send_system_alert("WARNING", "ctrader", "Connection lost")
        assert bot_with_system._run_async.call_count == 2

    def test_same_message_sent_after_window_expires(self, bot_with_system):
        bot_with_system.send_system_alert("WARNING", "yfinance", "All retries failed")
        assert bot_with_system._run_async.call_count == 1

        for key in bot_with_system._system_alert_cache:
            bot_with_system._system_alert_cache[key] -= 11 * 60

        bot_with_system.send_system_alert("WARNING", "yfinance", "All retries failed")
        assert bot_with_system._run_async.call_count == 2


class TestSendTradeAlert:
    """send_trade_alert formats and dispatches to the trade channel."""

    @pytest.fixture
    def bot_with_trade(self):
        with patch("src.alert_bot.telegram"):
            b = AlertBot()
            b._trade_bot = MagicMock()
            b._trade_chat_id = "456"
            b._run_async = MagicMock()
            return b

    def test_sends_with_ml_data(self, bot_with_trade):
        ml = {"win_probability": 0.74, "shap_top_features": ["VIX falling"]}
        bot_with_trade.send_trade_alert(_full_send_result(), ml)
        bot_with_trade._run_async.assert_called_once()

    def test_sends_without_ml_data(self, bot_with_trade):
        ml = {"win_probability": None, "shap_top_features": None}
        bot_with_trade.send_trade_alert(_full_send_result(), ml)
        bot_with_trade._run_async.assert_called_once()


class TestSendWeeklyReport:
    """send_weekly_report formats stats then dispatches to the system channel."""

    def test_calls_format_and_sends(self):
        with patch("src.alert_bot.telegram"):
            b = AlertBot()
            b._system_bot = MagicMock()
            b._system_chat_id = "789"
            b._run_async = MagicMock()

            stats = {"win_rate": 55.0, "by_session": {}, "by_grade": {}}
            with patch.object(b, "format_weekly_report", wraps=b.format_weekly_report) as spy:
                b.send_weekly_report(stats)
                spy.assert_called_once_with(stats)
            b._run_async.assert_called_once()


class TestFormatWeeklyReportSections:
    """format_weekly_report includes the right sections when data is present."""

    @pytest.fixture
    def bot(self):
        with patch("src.alert_bot.telegram"):
            return AlertBot()

    def test_includes_by_direction_section(self, bot):
        stats = {
            "win_rate": 60.0, "by_session": {}, "by_grade": {},
            "by_direction": {
                "LONG":  {"win_rate": 65.0, "wins": 13, "total": 20, "total_pnl": 42.5},
                "SHORT": {"win_rate": 55.0, "wins": 11, "total": 20, "total_pnl": -5.0},
            },
        }
        msg = bot.format_weekly_report(stats)
        assert "LONG vs SHORT:" in msg
        assert "LONG:" in msg
        assert "SHORT:" in msg

    def test_includes_shap_features(self, bot):
        stats = {
            "win_rate": 60.0, "by_session": {}, "by_grade": {},
            "shap_report": ["vix_pct", "delta_direction", "session_london"],
        }
        msg = bot.format_weekly_report(stats)
        assert "TOP ML FEATURES:" in msg
        assert "vix_pct" in msg

    def test_includes_pattern_accuracy(self, bot):
        stats = {
            "win_rate": 60.0, "by_session": {}, "by_grade": {},
            "pattern_accuracy": {"total_alerts": 30, "match_rate": 50.0, "win_rate_after_match": 70.0},
        }
        msg = bot.format_weekly_report(stats)
        assert "PATTERN SCANNER:" in msg
        assert "30" in msg

    def test_handles_empty_stats_gracefully(self, bot):
        msg = bot.format_weekly_report({})
        assert isinstance(msg, str)
        assert len(msg) > 0


class TestSendErrorHandling:
    """Telegram errors inside senders are caught — no propagation."""

    def test_telegram_error_caught(self):
        with patch("src.alert_bot.telegram"):
            b = AlertBot()
            b._trade_bot = MagicMock()
            b._trade_chat_id = "456"
            b._run_async = MagicMock(side_effect=Exception("Network down"))

            b.send_trade_alert(_full_send_result())  # must not raise
