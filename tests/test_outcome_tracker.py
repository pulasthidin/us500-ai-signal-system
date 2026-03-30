"""Tests for OutcomeTracker — triple barrier logic, PnL, catch-up, checkpoint."""

import json
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from src.outcome_tracker import OutcomeTracker, CHECKPOINT_PATH


@pytest.fixture
def tracker(mock_ctrader, mock_alert_bot):
    signal_logger = MagicMock()
    signal_logger.get_all_null_outcome_signals.return_value = []
    signal_logger.get_partial_win_signals.return_value = []
    signal_logger.get_pending_signals.return_value = []
    return OutcomeTracker(mock_ctrader, us500_id=1, signal_logger=signal_logger, alert_bot=mock_alert_bot)


def _make_signal(direction="LONG", entry=6500.0, sl=6485.0, tp=6525.0):
    return {
        "id": 1,
        "direction": direction,
        "entry_price": entry,
        "sl_price": sl,
        "tp_price": tp,
        "timestamp": "2025-03-20T10:00:00+00:00",
    }


def _make_bars(prices):
    """Create a simple bars DataFrame from a list of (high, low) tuples."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-03-20T10:05", periods=len(prices), freq="5min", tz="UTC"),
        "open": [p[1] + 1 for p in prices],
        "high": [p[0] for p in prices],
        "low": [p[1] for p in prices],
        "close": [p[0] - 1 for p in prices],
        "volume": [1000] * len(prices),
    })


class TestTripleBarrierLong:
    def test_win_when_tp_hit(self, tracker):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        bars = _make_bars([
            (6510, 6495),  # neither hit
            (6520, 6500),  # neither hit
            (6530, 6510),  # TP hit (high >= 6525)
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "WIN"
        assert result["label"] == 1
        assert result["bars_to_outcome"] == 3

    def test_loss_when_sl_hit(self, tracker):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        bars = _make_bars([
            (6510, 6495),
            (6505, 6480),  # SL hit (low <= 6485)
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "LOSS"
        assert result["label"] == -1
        assert result["bars_to_outcome"] == 2

    def test_both_barriers_same_bar_uses_distance(self, tracker):
        """When both SL and TP breached in same bar, resolve by comparing
        which extreme is further from entry (proxy for which hit first)."""
        signal = _make_signal("LONG", 6500, 6485, 6525)
        # LONG: tp_dist = |high - entry| = 30, sl_dist = |entry - low| = 20
        # TP distance > SL distance -> resolves as WIN
        bars = _make_bars([
            (6530, 6480),  # both hit, but TP extreme is further from entry
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "WIN"
        assert result["label"] == 1

    def test_both_barriers_same_bar_sl_closer_is_loss(self, tracker):
        """When SL extreme is further from entry than TP, resolve as LOSS."""
        signal = _make_signal("LONG", 6500, 6485, 6525)
        # LONG: tp_dist = |high - entry| = 26, sl_dist = |entry - low| = 30
        # SL distance > TP distance -> resolves as LOSS
        bars = _make_bars([
            (6526, 6470),  # both hit, but SL extreme is further from entry
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "LOSS"
        assert result["label"] == -1

    def test_skip_when_too_few_bars(self, tracker):
        """With fewer bars than OUTCOME_MIN_BARS_FOR_TIMEOUT, return None (skip)."""
        signal = _make_signal("LONG", 6500, 6485, 6525)
        bars = _make_bars([
            (6510, 6490),
            (6515, 6495),
            (6512, 6492),
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result is None, "Should skip (None) with only 3 bars, not mark TIMEOUT"

    def test_timeout_only_after_enough_bars(self, tracker):
        """TIMEOUT should only fire after checking sufficient bars."""
        signal = _make_signal("LONG", 6500, 6485, 6525)
        n = 80
        bars = _make_bars([(6510, 6490)] * n)
        result = tracker.triple_barrier_check(signal, bars)
        assert result is not None
        assert result["outcome"] == "TIMEOUT"
        assert result["bars_to_outcome"] == n


class TestTripleBarrierShort:
    def test_win_when_tp_hit(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475)
        bars = _make_bars([
            (6505, 6490),
            (6495, 6470),  # TP hit (low <= 6475)
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "WIN"

    def test_loss_when_sl_hit(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475)
        bars = _make_bars([
            (6510, 6495),
            (6520, 6505),  # SL hit (high >= 6515)
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "LOSS"


class TestPnLCalculation:
    def test_long_win_pnl(self):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        pnl = OutcomeTracker.calculate_pnl(signal, "WIN")
        assert pnl == 25.0  # tp - entry

    def test_long_loss_pnl(self):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        pnl = OutcomeTracker.calculate_pnl(signal, "LOSS")
        assert pnl == -15.0  # sl - entry

    def test_short_win_pnl(self):
        signal = _make_signal("SHORT", 6500, 6515, 6475)
        pnl = OutcomeTracker.calculate_pnl(signal, "WIN")
        assert pnl == 25.0  # entry - tp

    def test_short_loss_pnl(self):
        signal = _make_signal("SHORT", 6500, 6515, 6475)
        pnl = OutcomeTracker.calculate_pnl(signal, "LOSS")
        assert pnl == -15.0  # entry - sl

    def test_timeout_pnl_zero(self):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        assert OutcomeTracker.calculate_pnl(signal, "TIMEOUT") == 0.0


class TestMissingFields:
    def test_missing_sl_tp_returns_none(self, tracker):
        signal = {"id": 1, "direction": "LONG", "entry_price": 6500,
                  "sl_price": None, "tp_price": None, "timestamp": "2025-03-20T10:00:00"}
        bars = _make_bars([(6510, 6490)])
        result = tracker.triple_barrier_check(signal, bars)
        assert result is None


class TestStartupCatchup:
    def test_empty_db_returns_zero_summary(self, tracker):
        tracker._signal_logger.get_all_null_outcome_signals.return_value = []
        result = tracker.run_startup_catchup()
        assert result["checked"] == 0
        assert result["wins"] == 0

    def test_skips_young_signals(self, tracker):
        # signal 10 minutes old — well under the 30-minute OUTCOME_CHECK_DELAY_SECONDS threshold
        now = datetime.now(timezone.utc)
        young = {"id": 1, "direction": "LONG", "entry_price": 6500,
                 "sl_price": 6485, "tp_price": 6525,
                 "timestamp": (now - timedelta(minutes=10)).isoformat()}
        tracker._signal_logger.get_all_null_outcome_signals.return_value = [young]
        result = tracker.run_startup_catchup()
        assert result["skipped"] == 1
        assert result["checked"] == 0

    def test_resolves_old_signals(self, tracker):
        now = datetime.now(timezone.utc)
        n = 100
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=9), periods=n, freq="5min", tz="UTC"),
            "open": [6500] * n, "high": [6530] * n,
            "low": [6480] * n, "close": [6510] * n, "volume": [1000] * n,
        })
        tracker._ctrader.fetch_bars = MagicMock(return_value=bars)

        old_signal = {"id": 1, "direction": "LONG", "entry_price": 6500,
                      "sl_price": 6485, "tp_price": 6525,
                      "timestamp": (now - timedelta(hours=8)).isoformat()}
        tracker._signal_logger.get_all_null_outcome_signals.return_value = [old_signal]
        result = tracker.run_startup_catchup()
        assert result["checked"] >= 1
        tracker._signal_logger.update_outcome.assert_called()

    def test_h1_fallback_when_m5_empty(self, tracker, sample_h4_df):
        now = datetime.now(timezone.utc)
        old_signal = {"id": 2, "direction": "SHORT", "entry_price": 6500,
                      "sl_price": 6515, "tp_price": 6475,
                      "timestamp": (now - timedelta(hours=50)).isoformat()}
        tracker._signal_logger.get_all_null_outcome_signals.return_value = [old_signal]

        n = 200
        h1_bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=200), periods=n, freq="1h", tz="UTC"),
            "open": [6500] * n, "high": [6520] * n,
            "low": [6470] * n, "close": [6510] * n, "volume": [5000] * n,
        })

        def side_effect(sym, period, count):
            if period == "M5":
                return pd.DataFrame()
            return h1_bars.copy()
        tracker._ctrader.fetch_bars = MagicMock(side_effect=side_effect)

        result = tracker.run_startup_catchup()
        assert result["checked"] >= 1 or result["estimated"] >= 1
        tracker._signal_logger.update_outcome.assert_called()


    def test_resolves_partial_win_phase2_on_startup(self, tracker):
        """PARTIAL_WIN signals should be resolved during startup catchup (Phase 2)."""
        now = datetime.now(timezone.utc)
        n = 100
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=9), periods=n, freq="5min", tz="UTC"),
            "open": [6500] * n, "high": [6510] * n,
            "low": [6470] * n, "close": [6490] * n, "volume": [1000] * n,
        })
        tracker._ctrader.fetch_bars = MagicMock(return_value=bars)

        partial_signal = {
            "id": 10, "direction": "SHORT", "entry_price": 6520.0,
            "sl_price": 6535.0, "tp_price": 6490.0, "tp1_price": 6505.0,
            "outcome": "PARTIAL_WIN", "tp1_hit": 1,
            "tp1_hit_timestamp": (now - timedelta(hours=8)).isoformat(),
            "timestamp": (now - timedelta(hours=8, minutes=30)).isoformat(),
        }
        tracker._signal_logger.get_all_null_outcome_signals.return_value = []
        tracker._signal_logger.get_partial_win_signals.return_value = [partial_signal]

        result = tracker.run_startup_catchup()
        assert result["checked"] >= 1
        assert result["upgraded"] >= 1

    def test_catchup_summary_includes_upgraded_key(self, tracker):
        tracker._signal_logger.get_all_null_outcome_signals.return_value = []
        tracker._signal_logger.get_partial_win_signals.return_value = []
        result = tracker.run_startup_catchup()
        assert "upgraded" in result


class TestCheckpoint:
    def test_save_and_clear(self, tracker, tmp_path):
        import src.outcome_tracker as mod
        cp = str(tmp_path / "cp.json")
        mod.CHECKPOINT_PATH = cp
        tracker._save_checkpoint(42)
        assert os.path.exists(cp)
        with open(cp) as f:
            data = json.load(f)
        assert data["checking_signal_id"] == 42
        tracker._clear_checkpoint()
        assert not os.path.exists(cp)

    def test_resume_interrupted(self, tracker, tmp_path):
        import src.outcome_tracker as mod
        cp = str(tmp_path / "cp.json")
        mod.CHECKPOINT_PATH = cp
        with open(cp, "w") as f:
            json.dump({"checking_signal_id": 7, "started": "2025-01-01T00:00:00"}, f)
        now = datetime.now(timezone.utc)
        n = 100
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=6), periods=n, freq="5min", tz="UTC"),
            "open": [6500] * n, "high": [6530] * n,
            "low": [6480] * n, "close": [6510] * n, "volume": [1000] * n,
        })
        tracker._ctrader.fetch_bars = MagicMock(return_value=bars)

        sig = {"id": 7, "direction": "LONG", "entry_price": 6500,
               "sl_price": 6485, "tp_price": 6525,
               "timestamp": (now - timedelta(hours=5)).isoformat()}
        tracker._signal_logger.get_all_null_outcome_signals.return_value = [sig]
        tracker._resume_interrupted_checkpoint()
        tracker._signal_logger.update_outcome.assert_called()
        assert not os.path.exists(cp)


class TestEstimatedPnL:
    def test_estimated_timeout_pnl_zero(self):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        assert OutcomeTracker.calculate_pnl(signal, "ESTIMATED_TIMEOUT") == 0.0

    def test_estimated_win_pnl(self):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        pnl = OutcomeTracker.calculate_pnl(signal, "ESTIMATED_WIN")
        assert pnl == 25.0

    def test_estimated_loss_pnl(self):
        signal = _make_signal("SHORT", 6500, 6515, 6475)
        pnl = OutcomeTracker.calculate_pnl(signal, "ESTIMATED_LOSS")
        assert pnl == -15.0
