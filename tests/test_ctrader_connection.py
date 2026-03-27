"""Tests for CTraderConnection — delta decoding, symbol lookup, reconnection, token expiry."""

import os
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.ctrader_connection as ctmod
from src.ctrader_connection import CTraderConnection

# The mocked protobuf modules export __all__=[] so wildcard imports are empty.
# Inject the protobuf class names the production code relies on.
for _name in (
    "ProtoOAGetTrendbarsReq", "ProtoOAGetTrendbarsRes",
    "ProtoOASymbolsListReq", "ProtoOASymbolsListRes",
    "ProtoOAApplicationAuthReq", "ProtoOAAccountAuthReq",
    "ProtoOASubscribeSpotsReq",
):
    if not hasattr(ctmod, _name):
        setattr(ctmod, _name, MagicMock)


def _make_conn(**kwargs) -> CTraderConnection:
    """Create a CTraderConnection with mocked internals ready for unit tests."""
    conn = CTraderConnection(alert_bot=kwargs.get("alert_bot"))
    conn._client = MagicMock()
    conn._is_authorized = True
    conn._account_id = 12345
    return conn


def _mock_bar(low, delta_high, delta_open, delta_close, volume, ts_minutes):
    """Build a mock trendbar with the attributes CTraderConnection.fetch_bars reads."""
    bar = SimpleNamespace(
        low=low,
        deltaHigh=delta_high,
        deltaOpen=delta_open,
        deltaClose=delta_close,
        volume=volume,
        utcTimestampInMinutes=ts_minutes,
    )
    return bar


# ═══════════════════════════════════════════════════════════
#  Initialisation
# ═══════════════════════════════════════════════════════════

class TestCTraderInit:
    def test_initial_status_disconnected(self):
        conn = CTraderConnection()
        assert conn.get_connection_status() == "DISCONNECTED"

    def test_period_map_has_all_timeframes(self):
        expected = {"M1", "M5", "M15", "H1", "H4"}
        assert set(CTraderConnection.PERIOD_MAP.keys()) == expected


# ═══════════════════════════════════════════════════════════
#  fetch_bars — delta decoding
# ═══════════════════════════════════════════════════════════

class TestFetchBarsDeltaDecoding:
    def test_decode_ohlcv_correctly(self):
        conn = _make_conn()
        bar = _mock_bar(
            low=650000000,
            delta_high=1500000,
            delta_open=500000,
            delta_close=1000000,
            volume=5000,
            ts_minutes=29000000,
        )
        resp = MagicMock()
        resp.trendbar = [bar]

        with patch.object(conn, "_send_request", return_value=MagicMock()), \
             patch.object(conn, "_parse_response", return_value=resp):
            df = conn.fetch_bars(1, "M1", 10)

        assert df is not None
        row = df.iloc[0]
        assert row["low"] == 6500.00
        assert row["high"] == 6515.00
        assert row["open"] == 6505.00
        assert row["close"] == 6510.00
        assert row["volume"] == 5000

    def test_empty_bars_returns_empty_dataframe(self):
        conn = _make_conn()
        resp = MagicMock()
        resp.trendbar = []

        with patch.object(conn, "_send_request", return_value=MagicMock()), \
             patch.object(conn, "_parse_response", return_value=resp):
            df = conn.fetch_bars(1, "M1", 10)

        assert df is not None
        assert df.empty
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_invalid_period_returns_none(self):
        conn = _make_conn()
        result = conn.fetch_bars(1, "W1", 10)
        assert result is None

    def test_returns_sorted_by_timestamp(self):
        conn = _make_conn()
        bar_late = _mock_bar(650000000, 100000, 50000, 80000, 100, 29000010)
        bar_early = _mock_bar(650000000, 100000, 50000, 80000, 200, 29000000)

        resp = MagicMock()
        resp.trendbar = [bar_late, bar_early]

        with patch.object(conn, "_send_request", return_value=MagicMock()), \
             patch.object(conn, "_parse_response", return_value=resp):
            df = conn.fetch_bars(1, "M5", 10)

        assert df is not None
        assert df.iloc[0]["volume"] == 200, "Earlier bar should come first"
        assert df.iloc[1]["volume"] == 100


# ═══════════════════════════════════════════════════════════
#  get_current_price
# ═══════════════════════════════════════════════════════════

class TestGetCurrentPrice:
    def test_returns_last_close(self):
        conn = _make_conn()
        fake_df = pd.DataFrame({
            "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)] * 2,
            "open": [6500.0, 6510.0],
            "high": [6520.0, 6530.0],
            "low": [6490.0, 6500.0],
            "close": [6510.0, 6525.50],
            "volume": [100, 200],
        })
        with patch.object(conn, "fetch_bars", return_value=fake_df):
            price = conn.get_current_price(1)
        assert price == 6525.50

    def test_returns_none_on_empty_bars(self):
        conn = _make_conn()
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        with patch.object(conn, "fetch_bars", return_value=empty):
            price = conn.get_current_price(1)
        assert price is None


# ═══════════════════════════════════════════════════════════
#  get_symbol_id
# ═══════════════════════════════════════════════════════════

class TestGetSymbolId:
    def _build_symbol_response(self, symbols):
        """Return a mock response whose .symbol list has objects with symbolName/symbolId."""
        entries = [SimpleNamespace(symbolName=name, symbolId=sid) for name, sid in symbols]
        resp = MagicMock()
        resp.symbol = entries
        return resp

    def test_us500_alias_lookup(self):
        conn = _make_conn()
        resp = self._build_symbol_response([("SPX500", 42), ("EURUSD", 99)])

        with patch.object(conn, "_send_request", return_value=MagicMock()), \
             patch.object(conn, "_parse_response", return_value=resp):
            sid = conn.get_symbol_id("US500")

        assert sid == 42

    def test_cache_hit_skips_request(self):
        conn = _make_conn()
        conn._symbol_cache["US500"] = 77

        sid = conn.get_symbol_id("US500")
        assert sid == 77
        conn._client.send.assert_not_called()

    def test_unknown_symbol_returns_none(self):
        conn = _make_conn()
        resp = self._build_symbol_response([("EURUSD", 99)])

        with patch.object(conn, "_send_request", return_value=MagicMock()), \
             patch.object(conn, "_parse_response", return_value=resp):
            sid = conn.get_symbol_id("US500")

        assert sid is None


# ═══════════════════════════════════════════════════════════
#  Reconnection
# ═══════════════════════════════════════════════════════════

class TestReconnection:
    def test_handle_disconnect_sets_reconnecting(self):
        conn = _make_conn()
        with patch.object(conn, "_try_reconnect"):
            conn.handle_disconnect()
        assert conn.get_connection_status() == "RECONNECTING"

    def test_exponential_backoff_delays(self):
        from config import CTRADER_RECONNECT_DELAYS

        conn = _make_conn()
        recorded_delays = []

        def fake_call_later(delay, _fn):
            recorded_delays.append(delay)

        with patch("src.ctrader_connection.reactor") as mock_reactor:
            mock_reactor.callLater.side_effect = fake_call_later
            for i in range(len(CTRADER_RECONNECT_DELAYS)):
                conn._reconnect_attempt = i
                conn._try_reconnect()

        assert recorded_delays == CTRADER_RECONNECT_DELAYS

    def test_max_retries_continues_at_slowest(self):
        from config import CTRADER_RECONNECT_DELAYS, CTRADER_MAX_RETRIES

        conn = _make_conn()
        recorded_delays = []

        def fake_call_later(delay, _fn):
            recorded_delays.append(delay)

        with patch("src.ctrader_connection.reactor") as mock_reactor:
            mock_reactor.callLater.side_effect = fake_call_later
            conn._reconnect_attempt = CTRADER_MAX_RETRIES + 5
            conn._try_reconnect()

        assert recorded_delays == [CTRADER_RECONNECT_DELAYS[-1]]


# ═══════════════════════════════════════════════════════════
#  Token refresh
# ═══════════════════════════════════════════════════════════

class TestRefreshToken:
    def test_warns_when_token_near_expiry(self):
        alert_bot = MagicMock()
        conn = _make_conn(alert_bot=alert_bot)

        near_expiry = (datetime.now(timezone.utc) - timedelta(days=55)).isoformat()
        with patch.dict(os.environ, {"CTRADER_TOKEN_CREATED": near_expiry}):
            conn.refresh_token()

        alert_bot.send_system_alert.assert_called_once()
        args = alert_bot.send_system_alert.call_args
        assert "WARNING" in args[0]

    def test_no_warning_when_token_fresh(self):
        alert_bot = MagicMock()
        conn = _make_conn(alert_bot=alert_bot)

        fresh = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        with patch.dict(os.environ, {"CTRADER_TOKEN_CREATED": fresh}):
            conn.refresh_token()

        alert_bot.send_system_alert.assert_not_called()
