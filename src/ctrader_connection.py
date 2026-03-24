"""
cTrader Open API connection via Twisted async.
Handles authentication, bar fetching, reconnection with exponential backoff,
and token expiry warnings.

The Twisted reactor is inherently asynchronous — Deferred callbacks fire
only when the reactor runs.  We use `blockingCallFromThread` inside
threaded callers (the `schedule` main loop) so that fetch_bars / get_symbol_id
wait for the response synchronously without blocking the reactor itself.
"""

from __future__ import annotations

import logging
import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
from ctrader_open_api import Client, TcpProtocol
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from twisted.internet import reactor, defer, threads

from config import (
    CTRADER_RECONNECT_DELAYS,
    CTRADER_MAX_RETRIES,
    CTRADER_TOKEN_WARN_DAYS,
    US500_SYMBOL_ALIASES,
    USTEC_SYMBOL_ALIASES,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("ctrader")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _file_handler = logging.FileHandler("logs/ctrader.log", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_file_handler)


class CTraderConnection:
    """Manages a persistent cTrader Open API connection with auto-reconnect."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"

    def __init__(self, alert_bot=None) -> None:
        self._host: str = os.getenv("CTRADER_HOST", "live.ctraderapi.com")
        self._port: int = int(os.getenv("CTRADER_PORT", "5035"))
        self._client_id: str = os.getenv("CTRADER_CLIENT_ID", "")
        self._client_secret: str = os.getenv("CTRADER_CLIENT_SECRET", "")
        self._access_token: str = os.getenv("CTRADER_ACCESS_TOKEN", "")
        self._account_id: int = int(os.getenv("CTRADER_ACCOUNT_ID", "0"))

        self._status: str = self.DISCONNECTED
        self._client: Optional[Client] = None
        self._symbol_cache: Dict[str, int] = {}
        self._reconnect_attempt: int = 0
        self._disconnect_time: Optional[datetime] = None
        self._alert_bot = alert_bot
        self._is_authorized: bool = False
        self._live_callbacks: Dict[int, Callable] = {}
        self._reconnect_lock = threading.Lock()

    # ─── connection lifecycle ────────────────────────────────

    _reactor_running = False
    _reactor_lock = threading.Lock()

    @classmethod
    def _start_reactor(cls) -> None:
        """Start the Twisted reactor in a daemon thread (once only)."""
        with cls._reactor_lock:
            if cls._reactor_running:
                return
            cls._reactor_running = True

        def _run():
            from twisted.internet import reactor as r
            r.run(installSignalHandlers=False)

        t = threading.Thread(target=_run, daemon=True, name="twisted-reactor")
        t.start()
        time.sleep(0.3)
        logger.info("Twisted reactor started in background thread")

    def connect(self) -> None:
        """Open TCP connection to cTrader via Twisted."""
        try:
            self._start_reactor()
            self._client = Client(self._host, self._port, TcpProtocol)
            self._client.setConnectedCallback(self._on_connected)
            self._client.setDisconnectedCallback(self._on_disconnected)
            self._client.setMessageReceivedCallback(self._on_message)
            self._client.startService()

            deadline = time.time() + 10
            while self._status != self.CONNECTED and time.time() < deadline:
                time.sleep(0.2)

            if self._status != self.CONNECTED:
                raise ConnectionError("TCP connect timed out after 10s")

            logger.info("TCP connected to %s:%s", self._host, self._port)
        except Exception as exc:
            logger.error("Connect failed: %s", exc, exc_info=True)
            self._send_system_alert("CRITICAL", "ctrader_connect", str(exc))
            raise

    def _on_connected(self, client=None) -> None:
        logger.info("TCP connected")
        self._status = self.CONNECTED
        if self._reconnect_attempt > 0:
            self.handle_reconnect_success()
        self._reconnect_attempt = 0

    def _on_disconnected(self, client=None, reason=None) -> None:
        logger.warning("TCP disconnected: %s", reason)
        if self._status != self.DISCONNECTED:
            threads.deferToThread(self.handle_disconnect)

    def _on_message(self, client=None, message=None) -> None:
        """Generic message router — dispatches spot updates to registered callbacks."""
        if message is None:
            return
        msg_type = type(message).__name__
        if "SpotEvent" in msg_type or "Spot" in msg_type:
            try:
                sym_id = getattr(message, "symbolId", None)
                cb = self._live_callbacks.get(sym_id) if sym_id else None
                if cb:
                    threads.deferToThread(cb, message)
            except Exception:
                logger.debug("Spot callback error for %s", msg_type, exc_info=True)
        else:
            logger.debug("Unhandled cTrader message: %s", msg_type)

    # ─── blocking helper for Deferred ────────────────────────

    def _send_request(self, request, timeout: float = 30.0):
        """
        Send a protobuf request and block the calling thread until the Deferred resolves.
        The Twisted reactor runs in a background daemon thread.
        """
        if self._client is None:
            raise ConnectionError("cTrader client is not connected")

        from twisted.internet import defer

        def _do_send():
            d = self._client.send(request)
            d.addTimeout(timeout, reactor)
            return d

        return threads.blockingCallFromThread(reactor, _do_send)

    @staticmethod
    def _parse_response(raw, response_class):
        """Parse a raw ProtoMessage wrapper into a typed response."""
        if hasattr(raw, "payload"):
            resp = response_class()
            resp.ParseFromString(raw.payload)
            return resp
        return raw

    # ─── authentication ──────────────────────────────────────

    def authenticate_app(self) -> None:
        """Send ProtoOAApplicationAuthReq with client credentials."""
        try:
            request = ProtoOAApplicationAuthReq()
            request.clientId = self._client_id
            request.clientSecret = self._client_secret
            self._send_request(request)
            logger.info("Application authenticated")
        except Exception as exc:
            logger.error("App auth failed: %s", exc, exc_info=True)
            self._send_system_alert("CRITICAL", "ctrader_app_auth", str(exc))
            raise

    def authenticate_account(self) -> None:
        """Send ProtoOAAccountAuthReq for the trading account."""
        try:
            request = ProtoOAAccountAuthReq()
            request.ctidTraderAccountId = self._account_id
            request.accessToken = self._access_token
            self._send_request(request)
            self._is_authorized = True
            logger.info("Account %s authenticated", self._account_id)
        except Exception as exc:
            logger.error("Account auth failed: %s", exc, exc_info=True)
            self._send_system_alert("CRITICAL", "ctrader_account_auth", str(exc))
            raise

    # ─── token refresh ───────────────────────────────────────

    def refresh_token(self) -> None:
        """
        cTrader access tokens expire ~60 days.
        We cannot auto-refresh — warn the user via Telegram when close to expiry.
        The user must manually refresh in cTrader ID manager.
        """
        try:
            token_created = os.getenv("CTRADER_TOKEN_CREATED", "")
            if not token_created:
                logger.warning("CTRADER_TOKEN_CREATED not set — cannot check expiry")
                return

            created_dt = datetime.fromisoformat(token_created)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            expiry_dt = created_dt + timedelta(days=60)
            days_left = (expiry_dt - datetime.now(timezone.utc)).days

            if days_left <= CTRADER_TOKEN_WARN_DAYS:
                msg = (
                    f"cTrader token expires in {days_left} days.\n"
                    "Refresh in cTrader ID manager immediately."
                )
                logger.warning(msg)
                self._send_system_alert("WARNING", "ctrader_token", msg)
            else:
                logger.info("Token valid for %d more days", days_left)
        except Exception as exc:
            logger.error("Token refresh check failed: %s", exc, exc_info=True)

    # ─── symbol lookup ───────────────────────────────────────

    def get_symbol_id(self, symbol_name: str) -> Optional[int]:
        """
        Resolve a symbol name to its cTrader symbolId.
        Tries aliases since brokers may use different names (US500/SPX500/SP500).
        Results are cached.
        """
        if symbol_name in self._symbol_cache:
            return self._symbol_cache[symbol_name]

        aliases = US500_SYMBOL_ALIASES if "500" in symbol_name or "SPX" in symbol_name else USTEC_SYMBOL_ALIASES

        try:
            request = ProtoOASymbolsListReq()
            request.ctidTraderAccountId = self._account_id
            raw = self._send_request(request)
            response = self._parse_response(raw, ProtoOASymbolsListRes)

            for symbol in response.symbol:
                if symbol.symbolName in aliases or symbol_name.upper() in symbol.symbolName.upper():
                    self._symbol_cache[symbol_name] = symbol.symbolId
                    logger.info("Resolved %s -> symbolId %d", symbol.symbolName, symbol.symbolId)
                    return symbol.symbolId

            logger.error("Symbol %s not found (tried aliases: %s)", symbol_name, aliases)
            self._send_system_alert("WARNING", "symbol_lookup", f"{symbol_name} not found")
            return None
        except Exception as exc:
            logger.error("Symbol lookup failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "symbol_lookup", str(exc))
            return None

    # ─── bar fetching ────────────────────────────────────────

    PERIOD_MAP = {
        "M1": 1,
        "M5": 5,
        "M15": 7,
        "H1": 9,
        "H4": 10,
    }

    def fetch_bars(self, symbol_id: int, period: str, count: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV bars via ProtoOAGetTrendbarsReq.
        cTrader returns delta-encoded prices that need decoding.
        """
        try:
            proto_period = self.PERIOD_MAP.get(period)
            if proto_period is None:
                logger.error("Invalid period: %s", period)
                return None

            now_ms = int(time.time() * 1000)
            period_seconds = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400}
            from_ms = now_ms - (count * period_seconds.get(period, 3600) * 1000)

            request = ProtoOAGetTrendbarsReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId = symbol_id
            request.period = proto_period
            request.fromTimestamp = from_ms
            request.toTimestamp = now_ms

            raw = self._send_request(request)
            response = self._parse_response(raw, ProtoOAGetTrendbarsRes)
            bars_data: list = []

            for bar in response.trendbar:
                low = bar.low / 100000.0
                high = (bar.low + bar.deltaHigh) / 100000.0
                open_ = (bar.low + bar.deltaOpen) / 100000.0
                close = (bar.low + bar.deltaClose) / 100000.0
                volume = bar.volume if hasattr(bar, "volume") else 0
                ts = datetime.fromtimestamp(bar.utcTimestampInMinutes * 60, tz=timezone.utc)
                bars_data.append({
                    "timestamp": ts,
                    "open": round(open_, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": volume,
                })

            if not bars_data:
                logger.warning("No bars returned for symbol %s period %s", symbol_id, period)
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(bars_data)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info("Fetched %d %s bars for symbol %s", len(df), period, symbol_id)
            return df

        except Exception as exc:
            logger.error("fetch_bars failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "fetch_bars", str(exc))
            return None

    def get_current_price(self, symbol_id: int) -> Optional[float]:
        """Fetch the last M1 bar close as a proxy for current price."""
        try:
            df = self.fetch_bars(symbol_id, "M1", 2)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
            logger.warning("Could not get current price for symbol %s", symbol_id)
            return None
        except Exception as exc:
            logger.error("get_current_price failed: %s", exc, exc_info=True)
            return None

    # ─── live subscriptions ──────────────────────────────────

    def subscribe_live_bars(self, symbol_id: int, period: str, callback: Callable) -> None:
        """Subscribe to live spot updates and call *callback* on each completed bar."""
        try:
            if self._client is None:
                raise ConnectionError("cTrader client is not connected")
            self._live_callbacks[symbol_id] = callback
            request = ProtoOASubscribeSpotsReq()
            request.ctidTraderAccountId = self._account_id
            request.symbolId.append(symbol_id)
            deferred = self._client.send(request)
            deferred.addErrback(self._on_error, "subscribe_spots")
            logger.info("Subscribed to live spots for symbol %s", symbol_id)
        except Exception as exc:
            logger.error("subscribe_live_bars failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "subscribe_bars", str(exc))

    # ─── reconnection ────────────────────────────────────────

    def handle_disconnect(self) -> None:
        """Begin exponential-backoff reconnection loop."""
        self._status = self.RECONNECTING
        self._disconnect_time = datetime.now(timezone.utc)
        self._send_system_alert("CRITICAL", "ctrader", "Disconnected — starting reconnect loop")
        logger.warning("Starting reconnect loop")
        self._try_reconnect()

    def _try_reconnect(self) -> None:
        if self._reconnect_attempt >= CTRADER_MAX_RETRIES:
            msg = f"Failed to reconnect after {CTRADER_MAX_RETRIES} attempts — retrying at slowest rate"
            logger.critical(msg)
            self._send_system_alert("CRITICAL", "ctrader_reconnect", msg)

        delay = CTRADER_RECONNECT_DELAYS[
            min(self._reconnect_attempt, len(CTRADER_RECONNECT_DELAYS) - 1)
        ]
        self._reconnect_attempt += 1
        logger.info("Reconnect attempt %d in %ds", self._reconnect_attempt, delay)
        reactor.callLater(delay, self._schedule_reconnect)

    def _schedule_reconnect(self) -> None:
        """Trampoline: callLater fires on the reactor thread, so we jump to a worker."""
        threads.deferToThread(self._do_reconnect)

    def _do_reconnect(self) -> None:
        if not self._reconnect_lock.acquire(blocking=False):
            logger.debug("Reconnect already in progress — skipping duplicate attempt")
            return
        try:
            if self._client is not None:
                try:
                    self._client.stopService()
                except Exception:
                    pass
            self.connect()
            self.authenticate_app()
            self.authenticate_account()
            # Re-subscribe to live spots for all previously registered callbacks
            self._resubscribe_live_spots()
        except Exception as exc:
            logger.error("Reconnect attempt failed: %s", exc, exc_info=True)
            reactor.callFromThread(self._try_reconnect)
        finally:
            self._reconnect_lock.release()

    def _resubscribe_live_spots(self) -> None:
        """Re-subscribe to all symbol IDs that had live callbacks before disconnect."""
        for sym_id in list(self._live_callbacks.keys()):
            try:
                request = ProtoOASubscribeSpotsReq()
                request.ctidTraderAccountId = self._account_id
                request.symbolId.append(sym_id)
                self._send_request(request)
                logger.info("Re-subscribed to live spots for symbol %s after reconnect", sym_id)
            except Exception as exc:
                logger.error("Failed to re-subscribe symbol %s: %s", sym_id, exc, exc_info=True)

    def handle_reconnect_success(self) -> None:
        """Log recovery and alert the user with downtime duration."""
        self._status = self.CONNECTED
        downtime = ""
        if self._disconnect_time:
            delta = datetime.now(timezone.utc) - self._disconnect_time
            downtime = f" Downtime: {int(delta.total_seconds())}s"
        msg = f"Reconnected to cTrader.{downtime}"
        logger.info(msg)
        self._send_system_alert("INFO", "ctrader", msg)
        self._disconnect_time = None

    # ─── status ──────────────────────────────────────────────

    def get_connection_status(self) -> str:
        """Return CONNECTED / DISCONNECTED / RECONNECTING."""
        return self._status

    # ─── helpers ─────────────────────────────────────────────

    def _on_error(self, failure, context: str = "") -> None:
        logger.error("Deferred error [%s]: %s", context, failure)
        self._send_system_alert("WARNING", f"ctrader_{context}", str(failure))

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert for %s", component, exc_info=True)
