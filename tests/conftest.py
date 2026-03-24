"""
Shared test fixtures — mock cTrader, mock alert bot, sample DataFrames.
Mocks unavailable third-party packages so tests run without them installed.
"""

import os
import pathlib
import sys
import tempfile
import types
from unittest.mock import MagicMock

# ─── Mock missing packages before any src imports ────────────
_MOCK_MODULES = [
    "telegram",
    "smartmoneyconcepts",
    "smartmoneyconcepts.smc",
    "pandas_ta",
    "ctrader_open_api",
    "ctrader_open_api.messages",
    "ctrader_open_api.messages.OpenApiCommonMessages_pb2",
    "ctrader_open_api.messages.OpenApiMessages_pb2",
    "twisted",
    "twisted.internet",
    "twisted.internet.reactor",
    "twisted.internet.defer",
    "twisted.internet.task",
    "twisted.internet.threads",
    "groq",
    "shap",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        mock_mod = types.ModuleType(mod_name)
        mock_mod.__dict__.update({
            "__all__": [],
            "Client": MagicMock,
            "TcpProtocol": MagicMock,
            "Protobuf": MagicMock,
            "EndPoints": MagicMock,
            "Bot": MagicMock,
            "reactor": MagicMock(),
            "defer": MagicMock(),
            "task": MagicMock(),
            "threads": MagicMock(),
            "TreeExplainer": MagicMock,
            "Groq": MagicMock,
            "swing_highs_lows": MagicMock(return_value=MagicMock()),
            "bos_choch": MagicMock(return_value=MagicMock()),
            "fvg": MagicMock(return_value=MagicMock()),
            "ema": MagicMock(return_value=None),
        })
        sys.modules[mod_name] = mock_mod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

os.makedirs("logs", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("models", exist_ok=True)


@pytest.fixture
def tmp_path(request):
    """Override tmp_path to avoid Windows PermissionError on the default base temp dir."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="pytest_"))
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_alert_bot():
    bot = MagicMock()
    bot.send_system_alert = MagicMock()
    bot.send_trade_alert = MagicMock()
    bot.send_morning_brief = MagicMock()
    bot.send_weekly_report = MagicMock()
    return bot


@pytest.fixture
def sample_ohlcv_df():
    """50-bar OHLCV DataFrame with integer index (like cTrader returns)."""
    np.random.seed(42)
    n = 50
    base_price = 6500.0
    closes = base_price + np.cumsum(np.random.randn(n) * 5)
    opens = closes + np.random.randn(n) * 2
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 3)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 3)
    volumes = np.random.randint(1000, 50000, n)

    timestamps = pd.date_range("2025-03-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes,
    })
    return df


@pytest.fixture
def sample_h4_df():
    """200-bar H4 DataFrame."""
    np.random.seed(123)
    n = 200
    base = 6400.0
    closes = base + np.cumsum(np.random.randn(n) * 10)
    opens = closes + np.random.randn(n) * 5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 8)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 8)
    volumes = np.random.randint(5000, 100000, n)

    timestamps = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volumes,
    })


@pytest.fixture
def mock_ctrader(sample_ohlcv_df, sample_h4_df):
    """Mock CTraderConnection that returns sample data."""
    conn = MagicMock()
    conn.get_connection_status.return_value = "CONNECTED"
    conn.get_current_price.return_value = 6544.50

    def _fetch_bars(symbol_id, period, count):
        if period == "H4":
            return sample_h4_df.copy()
        elif period == "H1":
            return sample_h4_df.tail(48).copy().reset_index(drop=True)
        else:
            return sample_ohlcv_df.copy()

    conn.fetch_bars = MagicMock(side_effect=_fetch_bars)
    return conn


@pytest.fixture
def sample_macro_data():
    return {
        "bias": "LONG",
        "vix_bucket": "normal",
        "vix_value": 17.5,
        "vix_pct": -2.5,
        "size_label": "normal",
        "trade_allowed": True,
        "short_only": False,
        "vix_direction_bias": "BUY_BIAS",
        "us10y_direction": "falling",
        "oil_direction": "falling",
        "dxy_direction": "flat",
        "rut_direction": "green",
        "groq_sentiment": "NEUTRAL",
        "bullish_count": 3,
        "bearish_count": 0,
        "score_contribution": 1,
        "raw_data": {
            "vix": {"value": 17.5, "pct_change": -2.5, "direction": "falling", "prev_close": 18.0},
            "dxy": {"value": 99.5, "pct_change": 0.1, "direction": "flat", "prev_close": 99.4},
            "us10y": {"value": 4.3, "pct_change": -0.1, "direction": "falling", "prev_close": 4.31},
            "oil": {"value": 72.0, "pct_change": -0.8, "direction": "falling", "prev_close": 72.6},
            "rut": {"value": 2100.0, "pct_change": 0.5, "direction": "green", "prev_close": 2089.5},
            "fetch_time": datetime.now(timezone.utc).isoformat(),
        },
    }


@pytest.fixture
def sample_news_data():
    return {
        "is_news_day": False,
        "events_today": [],
        "pre_news_blocked": False,
        "pre_news_event": None,
        "minutes_until_news": None,
        "post_news_caution": False,
        "post_news_event": None,
        "next_event": None,
        "block_reason": None,
    }
