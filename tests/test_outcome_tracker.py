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

    def test_both_barriers_same_bar_is_loss(self, tracker):
        """When both SL and TP are breached in the same bar, assume LOSS
        because SL (1.5×ATR) is closer to entry than TP (2.5×ATR)."""
        signal = _make_signal("LONG", 6500, 6485, 6525)
        bars = _make_bars([
            (6530, 6480),  # both SL and TP breached in same bar
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "LOSS"
        assert result["label"] == -1

    def test_timeout(self, tracker):
        signal = _make_signal("LONG", 6500, 6485, 6525)
        bars = _make_bars([
            (6510, 6490),
            (6515, 6495),
            (6512, 6492),
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "TIMEOUT"
        assert result["label"] == 0


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
        old_signal = {"id": 1, "direction": "LONG", "entry_price": 6500,
                      "sl_price": 6485, "tp_price": 6525,
                      "timestamp": (now - timedelta(hours=8)).isoformat()}
        tracker._signal_logger.get_all_null_outcome_signals.return_value = [old_signal]
        result = tracker.run_startup_catchup()
        assert result["checked"] >= 0
        tracker._signal_logger.update_outcome.assert_called()

    def test_h1_fallback_when_m5_empty(self, tracker, sample_h4_df):
        now = datetime.now(timezone.utc)
        old_signal = {"id": 2, "direction": "SHORT", "entry_price": 6500,
                      "sl_price": 6515, "tp_price": 6475,
                      "timestamp": (now - timedelta(hours=10)).isoformat()}
        tracker._signal_logger.get_all_null_outcome_signals.return_value = [old_signal]

        call_count = [0]
        def side_effect(sym, period, count):
            call_count[0] += 1
            if period == "M5":
                return pd.DataFrame()
            return sample_h4_df.copy()
        tracker._ctrader.fetch_bars = MagicMock(side_effect=side_effect)

        result = tracker.run_startup_catchup()
        assert call_count[0] >= 2
        tracker._signal_logger.update_outcome.assert_called()


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
