"""Tests for CalendarChecker — news window detection and filtering."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from src.calendar_checker import CalendarChecker


@pytest.fixture
def checker(mock_alert_bot):
    return CalendarChecker(alert_bot=mock_alert_bot)


class TestFilterHighImpactUSD:
    def test_filters_correctly(self, checker):
        events = [
            {"country": "USD", "impact": "High", "title": "NFP Release"},
            {"country": "USD", "impact": "Low", "title": "Housing Starts"},
            {"country": "EUR", "impact": "High", "title": "ECB Rate"},
            {"country": "USD", "impact": "High", "title": "Something Else"},
        ]
        result = checker.filter_high_impact_usd(events)
        assert len(result) == 1
        assert result[0]["title"] == "NFP Release"

    def test_empty_list(self, checker):
        assert checker.filter_high_impact_usd([]) == []

    def test_matches_multiple_keywords(self, checker):
        events = [
            {"country": "USD", "impact": "High", "title": "CPI y/y"},
            {"country": "USD", "impact": "High", "title": "FOMC Statement"},
            {"country": "USD", "impact": "High", "title": "PPI m/m"},
        ]
        result = checker.filter_high_impact_usd(events)
        assert len(result) == 3


class TestGetNewsStatus:
    def test_no_news_day(self, checker):
        checker._events = []
        checker._last_fetch_date = None
        with patch.object(checker, "fetch_weekly_calendar", return_value=[]):
            status = checker.get_news_status()
        assert status["is_news_day"] is False
        assert status["pre_news_blocked"] is False

    def test_returns_all_keys(self, checker):
        checker._events = []
        with patch.object(checker, "fetch_weekly_calendar", return_value=[]):
            status = checker.get_news_status()
        expected_keys = [
            "is_news_day", "events_today", "pre_news_blocked",
            "pre_news_event", "minutes_until_news", "post_news_caution",
            "post_news_event", "next_event", "block_reason",
        ]
        for key in expected_keys:
            assert key in status, f"Missing key: {key}"


class TestPreNewsWindow:
    def test_not_blocked_when_no_events(self, checker):
        checker._events = []
        with patch.object(checker, "get_todays_events", return_value=[]):
            result = checker.is_pre_news_window()
        assert result["blocked"] is False

    def test_blocked_when_event_within_30_min(self, checker):
        future_time = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
        events = [{"country": "USD", "impact": "High", "title": "CPI",
                    "date": future_time}]
        with patch.object(checker, "get_todays_events", return_value=events):
            result = checker.is_pre_news_window()
        assert result["blocked"] is True
        assert result["event"] == "CPI"


class TestPostNewsWindow:
    def test_not_in_window_when_no_events(self, checker):
        with patch.object(checker, "get_todays_events", return_value=[]):
            result = checker.is_post_news_window()
        assert result["in_window"] is False

    def test_in_window_when_event_just_passed(self, checker):
        past_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        events = [{"country": "USD", "impact": "High", "title": "NFP",
                    "date": past_time}]
        with patch.object(checker, "get_todays_events", return_value=events):
            result = checker.is_post_news_window()
        assert result["in_window"] is True
