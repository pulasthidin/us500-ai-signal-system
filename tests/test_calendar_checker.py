"""Tests for CalendarChecker — news window detection and filtering."""

import json

import pytest
from datetime import datetime, timezone, timedelta, date
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


# ─── fetch_weekly_calendar ────────────────────────────────


class TestFetchWeeklyCalendar:
    def test_fetches_and_caches(self, checker, tmp_path):
        sample_events = [
            {"country": "USD", "impact": "High", "title": "NFP Release", "date": "2026-03-28T12:30:00+00:00"},
            {"country": "EUR", "impact": "Low", "title": "ECB Minutes", "date": "2026-03-28T14:00:00+00:00"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_events
        mock_resp.raise_for_status = MagicMock()

        cache_file = str(tmp_path / "calendar_cache.json")
        with patch("src.calendar_checker.requests.get", return_value=mock_resp), \
             patch("src.calendar_checker.CACHE_PATH", cache_file):
            result = checker.fetch_weekly_calendar()

        assert len(result) == 2
        assert checker._events == sample_events
        assert checker._fetch_failed is False
        assert checker._last_fetch_date == date.today()

    def test_http_error_returns_empty(self, checker):
        with patch("src.calendar_checker.requests.get", side_effect=Exception("Connection refused")):
            result = checker.fetch_weekly_calendar()
        assert result == []
        assert checker._fetch_failed is True
        assert checker._fail_count == 1


# ─── cache round-trip ─────────────────────────────────────


class TestCacheRoundtrip:
    def test_save_and_load_cache(self, checker, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        events = [
            {"country": "USD", "impact": "High", "title": "NFP Release", "date": "2026-03-28T12:30:00+00:00"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.raise_for_status = MagicMock()

        with patch("src.calendar_checker.CACHE_PATH", cache_file), \
             patch("src.calendar_checker.requests.get", return_value=mock_resp):
            checker.fetch_weekly_calendar()
            loaded = checker._load_cache()

        assert len(loaded) == 1
        assert loaded[0]["title"] == "NFP Release"

    def test_stale_cache_triggers_refetch(self, checker, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with open(cache_file, "w") as f:
            json.dump({"date": yesterday, "events": [{"title": "Old Event"}]}, f)

        with patch("src.calendar_checker.CACHE_PATH", cache_file):
            loaded = checker._load_cache()

        assert loaded == []


# ─── _parse_event_datetime ────────────────────────────────


class TestParseEventDatetime:
    def test_parses_standard_format(self, checker):
        event = {"date": "2026-03-28T12:30:00+00:00"}
        result = checker._parse_event_datetime(event)
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.hour == 12
        assert result.minute == 30

    def test_handles_missing_time(self, checker):
        event = {"title": "NFP Release"}
        result = checker._parse_event_datetime(event)
        assert result is None


# ─── get_todays_events ────────────────────────────────────


class TestGetTodaysEvents:
    def _preload(self, checker, events):
        """Bypass _ensure_loaded by setting internal state directly."""
        checker._events = events
        checker._last_fetch_date = date.today()
        checker._fetch_failed = False

    def test_filters_by_today(self, checker):
        today_str = date.today().isoformat()
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()
        self._preload(checker, [
            {"country": "USD", "impact": "High", "title": "CPI y/y", "date": f"{today_str}T12:30:00+00:00"},
            {"country": "USD", "impact": "High", "title": "NFP Release", "date": f"{yesterday_str}T12:30:00+00:00"},
        ])
        result = checker.get_todays_events()
        assert len(result) == 1
        assert result[0]["title"] == "CPI y/y"

    def test_empty_when_no_events_today(self, checker):
        yesterday_str = (date.today() - timedelta(days=1)).isoformat()
        self._preload(checker, [
            {"country": "USD", "impact": "High", "title": "NFP Release", "date": f"{yesterday_str}T12:30:00+00:00"},
        ])
        result = checker.get_todays_events()
        assert result == []


# ─── _ensure_loaded ───────────────────────────────────────


class TestEnsureLoaded:
    def test_uses_cache_on_second_call(self, checker, tmp_path):
        cache_file = str(tmp_path / "cache.json")
        events = [{"country": "USD", "impact": "High", "title": "NFP Release",
                    "date": date.today().isoformat() + "T12:30:00+00:00"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = events
        mock_resp.raise_for_status = MagicMock()

        with patch("src.calendar_checker.requests.get", return_value=mock_resp) as mock_get, \
             patch("src.calendar_checker.CACHE_PATH", cache_file):
            checker._ensure_loaded()
            first_count = mock_get.call_count
            checker._ensure_loaded()
            assert mock_get.call_count == first_count
