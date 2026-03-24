"""
Economic Calendar Checker — fetches high-impact USD events from Forex Factory
via the free faireconomy.media JSON feed.  No API key required.

Fetched once per day, cached to data/news_calendar_today.json.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta, date
from typing import Any, Dict, List, Optional

import requests

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

from config import (
    NEWS_BLOCK_MINUTES_BEFORE,
    NEWS_CAUTION_MINUTES_AFTER,
    NEWS_KEYWORDS,
)

logger = logging.getLogger("calendar")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/calendar.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_PATH = os.path.join("data", "news_calendar_today.json")


class CalendarChecker:
    """Fetches Forex Factory calendar and manages news-window blocking."""

    def __init__(self, alert_bot=None) -> None:
        self._alert_bot = alert_bot
        self._events: List[Dict[str, Any]] = []
        self._last_fetch_date: Optional[date] = None
        self._fetch_failed: bool = False
        self._fail_count: int = 0
        self._last_fail_time: float = 0.0

    # ─── fetch & cache ───────────────────────────────────────

    def fetch_weekly_calendar(self) -> List[Dict[str, Any]]:
        """
        GET this week's calendar from faireconomy.media.
        Free, no key, no limits.  Caches to disk for the day.
        """
        try:
            resp = requests.get(CALENDAR_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            events = data if isinstance(data, list) else []
            self._events = events
            self._last_fetch_date = date.today()
            self._fetch_failed = False
            self._fail_count = 0

            try:
                with open(CACHE_PATH, "w") as f:
                    json.dump({"date": date.today().isoformat(), "events": events}, f)
            except Exception:
                pass

            logger.info("Fetched %d calendar events from faireconomy.media", len(events))
            return events

        except Exception as exc:
            import time as _time
            logger.warning("Calendar fetch failed: %s — using empty calendar", exc)
            self._fetch_failed = True
            self._fail_count += 1
            self._last_fail_time = _time.time()
            return []

    def _load_cache(self) -> List[Dict[str, Any]]:
        """Load today's cache from disk if it exists and is from today."""
        try:
            if not os.path.exists(CACHE_PATH):
                return []
            with open(CACHE_PATH) as f:
                data = json.load(f)
            if data.get("date") == date.today().isoformat():
                return data.get("events", [])
        except Exception:
            pass
        return []

    def _ensure_loaded(self) -> None:
        """Make sure events are loaded for today — from cache or fresh fetch.

        After a failed fetch, applies exponential backoff (60s, 120s, 240s, ... up
        to 30 min) before retrying so we don't hammer the API on every signal check.
        """
        if self._last_fetch_date == date.today() and not self._fetch_failed:
            return

        cached = self._load_cache()
        if cached:
            self._events = cached
            self._last_fetch_date = date.today()
            self._fetch_failed = False
            self._fail_count = 0
            logger.info("Loaded %d events from cache", len(cached))
            return

        # Backoff: wait before retrying after failures
        if self._fetch_failed and self._last_fail_time > 0:
            import time as _time
            backoff_secs = min(60 * (2 ** (self._fail_count - 1)), 1800)  # max 30 min
            elapsed = _time.time() - self._last_fail_time
            if elapsed < backoff_secs:
                return

        self.fetch_weekly_calendar()

    # ─── filters ─────────────────────────────────────────────

    def filter_high_impact_usd(self, events: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Keep only USD + High impact events, optionally filtered by NEWS_KEYWORDS."""
        events = events or self._events
        filtered = []
        for e in events:
            country = str(e.get("country", "")).upper()
            impact = str(e.get("impact", "")).lower()
            title = str(e.get("title", e.get("event", "")))

            if country != "USD":
                continue
            if "high" not in impact:
                continue

            keyword_match = any(kw.lower() in title.lower() for kw in NEWS_KEYWORDS)
            if keyword_match:
                filtered.append(e)

        return filtered

    def get_todays_events(self) -> List[Dict[str, Any]]:
        """Return today's high-impact USD events, sorted by time."""
        self._ensure_loaded()

        today_str = date.today().isoformat()
        high_impact = self.filter_high_impact_usd()

        todays = []
        for e in high_impact:
            event_date = str(e.get("date", ""))
            if today_str in event_date:
                todays.append(e)

        todays.sort(key=lambda x: str(x.get("date", "")))
        return todays

    # ─── news window checks ──────────────────────────────────

    def _parse_event_datetime(self, event: Dict[str, Any]) -> Optional[datetime]:
        """Best-effort parsing of an event's date/time into a UTC datetime."""
        for key in ("date", "datetime", "time"):
            raw = event.get(key, "")
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(str(raw))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
        return None

    def is_pre_news_window(self) -> Dict[str, Any]:
        """True if current UTC time is within NEWS_BLOCK_MINUTES_BEFORE of any event."""
        try:
            now = datetime.now(timezone.utc)
            for event in self.get_todays_events():
                event_dt = self._parse_event_datetime(event)
                if event_dt is None:
                    continue
                delta = (event_dt - now).total_seconds() / 60.0
                if 0 <= delta <= NEWS_BLOCK_MINUTES_BEFORE:
                    title = event.get("title", event.get("event", "Unknown"))
                    return {"blocked": True, "event": title, "minutes_until": round(delta, 1)}
            return {"blocked": False, "event": None, "minutes_until": None}
        except Exception as exc:
            logger.error("is_pre_news_window failed: %s", exc, exc_info=True)
            return {"blocked": False, "event": None, "minutes_until": None}

    def is_post_news_window(self) -> Dict[str, Any]:
        """True if current time is within NEWS_CAUTION_MINUTES_AFTER of a past event."""
        try:
            now = datetime.now(timezone.utc)
            for event in self.get_todays_events():
                event_dt = self._parse_event_datetime(event)
                if event_dt is None:
                    continue
                delta = (now - event_dt).total_seconds() / 60.0
                if 0 <= delta <= NEWS_CAUTION_MINUTES_AFTER:
                    title = event.get("title", event.get("event", "Unknown"))
                    return {"in_window": True, "event_name": title, "minutes_since": round(delta, 1)}
            return {"in_window": False, "event_name": None, "minutes_since": None}
        except Exception as exc:
            logger.error("is_post_news_window failed: %s", exc, exc_info=True)
            return {"in_window": False, "event_name": None, "minutes_since": None}

    # ─── composite status ────────────────────────────────────

    def get_news_status(self) -> Dict[str, Any]:
        """Full news-day status dict consumed by the checklist engine."""
        try:
            todays_events = self.get_todays_events()
            pre = self.is_pre_news_window()
            post = self.is_post_news_window()

            next_event = None
            now = datetime.now(timezone.utc)
            for e in todays_events:
                edt = self._parse_event_datetime(e)
                if edt and edt > now:
                    title = e.get("title", e.get("event", ""))
                    minutes_until = (edt - now).total_seconds() / 60.0
                    next_event = {"event": title, "time": str(e.get("date", "")), "minutes_until": round(minutes_until, 1)}
                    break

            return {
                "is_news_day": len(todays_events) > 0,
                "events_today": todays_events,
                "pre_news_blocked": pre["blocked"],
                "pre_news_event": pre["event"],
                "minutes_until_news": pre["minutes_until"],
                "post_news_caution": post["in_window"],
                "post_news_event": post["event_name"],
                "next_event": next_event,
                "block_reason": f"Pre-news: {pre['event']}" if pre["blocked"] else None,
                "calendar_unavailable": self._fetch_failed,
            }

        except Exception as exc:
            logger.error("get_news_status failed: %s", exc, exc_info=True)
            return {
                "is_news_day": False, "events_today": [],
                "pre_news_blocked": False, "pre_news_event": None,
                "minutes_until_news": None, "post_news_caution": False,
                "post_news_event": None, "next_event": None, "block_reason": None,
                "calendar_unavailable": True,
            }

    # ─── helpers ─────────────────────────────────────────────

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
