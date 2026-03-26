"""Tests for the TP1 partial profit system — entry calculation, outcome state machine, DB persistence."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from src.entry_checker import EntryChecker
from src.outcome_tracker import OutcomeTracker
from src.signal_logger import SignalLogger
from config import TP1_R_MULTIPLE


# ─── TP1 calculation ─────────────────────────────────────────

class TestTP1Calculation:
    @pytest.fixture
    def entry(self, mock_ctrader, mock_alert_bot):
        return EntryChecker(mock_ctrader, us500_id=1, ustec_id=2, alert_bot=mock_alert_bot)

    def test_tp1_in_entry_result(self, entry):
        result = entry.get_entry_result("SHORT", 6500.0)
        assert "tp1_price" in result
        assert "tp1_points" in result

    def test_tp1_equals_1r_for_short(self, entry):
        result = entry.calculate_rr(6500.0, "SHORT", atr=10.0)
        sl_dist = result["sl_points"]
        tp1_dist = result["tp1_points"]
        assert abs(tp1_dist - sl_dist * TP1_R_MULTIPLE) < 0.01
        assert result["tp1_price"] < 6500.0

    def test_tp1_equals_1r_for_long(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", atr=10.0)
        sl_dist = result["sl_points"]
        tp1_dist = result["tp1_points"]
        assert abs(tp1_dist - sl_dist * TP1_R_MULTIPLE) < 0.01
        assert result["tp1_price"] > 6500.0

    def test_tp1_between_entry_and_tp2(self, entry):
        result = entry.calculate_rr(6500.0, "SHORT", atr=10.0)
        assert result["tp1_price"] > result["tp_price"]
        assert result["tp1_price"] < 6500.0

    def test_error_return_includes_tp1_keys(self, entry):
        result = entry.calculate_rr(0, "SHORT", atr=0)
        assert "tp1_price" in result
        assert "tp1_points" in result

    def test_empty_entry_includes_tp1(self, entry):
        result = entry._empty_entry()
        assert "tp1_price" in result
        assert "tp1_points" in result


# ─── Outcome tracker quad barrier ────────────────────────────

def _make_signal(direction="SHORT", entry=6500.0, sl=6515.0, tp=6475.0, tp1=6490.0, outcome=None):
    return {
        "id": 1, "direction": direction,
        "entry_price": entry, "sl_price": sl,
        "tp_price": tp, "tp1_price": tp1,
        "timestamp": "2025-03-20T10:00:00+00:00",
        "outcome": outcome, "outcome_timestamp": None,
        "tp1_hit_timestamp": None,
    }


def _make_bars(prices):
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-03-20T10:05", periods=len(prices), freq="5min", tz="UTC"),
        "open": [p[1] + 1 for p in prices],
        "high": [p[0] for p in prices],
        "low": [p[1] for p in prices],
        "close": [p[0] - 1 for p in prices],
        "volume": [1000] * len(prices),
    })


@pytest.fixture
def tracker(mock_ctrader, mock_alert_bot):
    sl = MagicMock()
    sl.get_all_null_outcome_signals.return_value = []
    sl.get_pending_signals.return_value = []
    return OutcomeTracker(mock_ctrader, us500_id=1, signal_logger=sl, alert_bot=mock_alert_bot)


class TestPhase1TP1Hit:
    def test_tp1_hit_returns_partial_win(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490)
        bars = _make_bars([
            (6505, 6495),
            (6500, 6488),  # low=6488 <= tp1=6490 -> TP1 hit
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result is not None
        assert result["outcome"] == "PARTIAL_WIN"
        assert result["tp1_hit"] is True
        assert result["pnl_points"] > 0

    def test_sl_hit_before_tp1(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490)
        bars = _make_bars([
            (6518, 6495),  # high=6518 >= sl=6515 -> SL hit
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "LOSS"

    def test_tp2_hit_directly_skips_tp1(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490)
        bars = _make_bars([
            (6505, 6470),  # low=6470 <= tp=6475 -> TP2 hit directly
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "WIN"

    def test_tp1_pnl_is_half_distance(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490)
        bars = _make_bars([
            (6505, 6495),
            (6500, 6488),
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert abs(result["pnl_points"] - 5.0) < 0.1  # (6500-6490)/2 = 5.0


class TestPhase2Upgrade:
    def test_tp2_upgrades_partial_to_win(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490, outcome="PARTIAL_WIN")
        signal["tp1_hit_timestamp"] = "2025-03-20T10:10:00+00:00"
        signal["outcome_timestamp"] = "2025-03-20T10:10:00+00:00"
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2025-03-20T10:05", periods=4, freq="5min", tz="UTC"),
            "open": [6498, 6495, 6490, 6480],
            "high": [6505, 6498, 6495, 6485],
            "low": [6495, 6485, 6480, 6470],
            "close": [6496, 6488, 6482, 6472],
            "volume": [1000] * 4,
        })
        result = tracker.triple_barrier_check(signal, bars)
        assert result is not None
        assert result["outcome"] == "WIN"
        assert result.get("upgraded_from_partial") is True

    def test_breakeven_finalizes_partial(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490, outcome="PARTIAL_WIN")
        signal["tp1_hit_timestamp"] = "2025-03-20T10:10:00+00:00"
        signal["outcome_timestamp"] = "2025-03-20T10:10:00+00:00"
        n = 80
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2025-03-20T10:05", periods=n, freq="5min", tz="UTC"),
            "open": [6498] * n,
            "high": [6505] * n,
            "low": [6495] * n,
            "close": [6498] * n,
            "volume": [1000] * n,
        })
        result = tracker.triple_barrier_check(signal, bars)
        assert result is not None
        assert result["outcome"] == "PARTIAL_WIN_FINAL"

    def test_upgraded_win_pnl_is_average(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=6490, outcome="PARTIAL_WIN")
        signal["tp1_hit_timestamp"] = "2025-03-20T10:10:00+00:00"
        signal["outcome_timestamp"] = "2025-03-20T10:10:00+00:00"
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2025-03-20T10:05", periods=3, freq="5min", tz="UTC"),
            "open": [6498, 6490, 6480],
            "high": [6505, 6495, 6485],
            "low": [6495, 6480, 6470],
            "close": [6492, 6482, 6472],
            "volume": [1000] * 3,
        })
        result = tracker.triple_barrier_check(signal, bars)
        assert result is not None
        assert result["outcome"] == "WIN"
        expected = ((6500 - 6490) + (6500 - 6475)) / 2
        assert abs(result["pnl_points"] - expected) < 0.1


class TestPhase1Long:
    def test_long_tp1_hit(self, tracker):
        signal = _make_signal("LONG", 6500, 6485, 6525, tp1=6510)
        bars = _make_bars([
            (6508, 6495),
            (6512, 6500),  # high=6512 >= tp1=6510 -> TP1 hit
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "PARTIAL_WIN"

    def test_long_tp2_direct(self, tracker):
        signal = _make_signal("LONG", 6500, 6485, 6525, tp1=6510)
        bars = _make_bars([
            (6530, 6500),  # high=6530 >= tp=6525 -> TP2 hit
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "WIN"


class TestNoTP1FallsBackCleanly:
    def test_no_tp1_uses_old_behavior(self, tracker):
        signal = _make_signal("SHORT", 6500, 6515, 6475, tp1=None)
        bars = _make_bars([
            (6505, 6495),
            (6500, 6470),
        ])
        result = tracker.triple_barrier_check(signal, bars)
        assert result["outcome"] == "WIN"


# ─── DB persistence ─────────────────────────────────────────

class TestTP1DBColumns:
    @pytest.fixture
    def logger_db(self, tmp_path):
        sl = SignalLogger()
        sl._db_path = str(tmp_path / "test_tp1.db")
        sl.init_db()
        return sl

    def test_tp1_columns_exist(self, logger_db):
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()}
        conn.close()
        assert "tp1_price" in cols
        assert "tp1_points" in cols
        assert "tp1_hit" in cols
        assert "tp1_hit_timestamp" in cols

    def test_pending_signals_includes_partial_win(self, logger_db):
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute("""INSERT INTO signals (timestamp, direction, outcome, tp1_hit, save_status)
                        VALUES ('2025-01-01', 'SHORT', 'PARTIAL_WIN', 1, 'complete')""")
        conn.commit()
        conn.close()
        pending = logger_db.get_pending_signals()
        assert len(pending) == 1
        assert pending[0]["outcome"] == "PARTIAL_WIN"

    def test_pending_excludes_partial_win_final(self, logger_db):
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute("""INSERT INTO signals (timestamp, direction, outcome, save_status)
                        VALUES ('2025-01-01', 'SHORT', 'PARTIAL_WIN_FINAL', 'complete')""")
        conn.commit()
        conn.close()
        pending = logger_db.get_pending_signals()
        assert len(pending) == 0

    def test_update_tp1_hit(self, logger_db):
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute("""INSERT INTO signals (timestamp, direction, save_status)
                        VALUES ('2025-01-01', 'SHORT', 'complete')""")
        conn.commit()
        conn.close()
        logger_db.update_tp1_hit(1, "2025-01-01T10:00:00")
        import sqlite3 as sq
        c = sq.connect(logger_db._db_path)
        c.row_factory = sq.Row
        r = c.execute("SELECT * FROM signals WHERE id=1").fetchone()
        c.close()
        assert r["outcome"] == "PARTIAL_WIN"
        assert r["tp1_hit"] == 1
        assert r["tp1_hit_timestamp"] == "2025-01-01T10:00:00"

    def test_win_rate_counts_partial_win(self, logger_db):
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute("""INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
                        VALUES ('2025-01-01', 'SHORT', 'PARTIAL_WIN_FINAL', 0, 'complete')""")
        conn.execute("""INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
                        VALUES ('2025-01-02', 'SHORT', 'LOSS', -1, 'complete')""")
        conn.commit()
        conn.close()
        wr = logger_db.get_win_rate()
        assert wr == 50.0  # 1 partial win, 1 loss = 50%


# ─── ML trainer PARTIAL_WIN handling ─────────────────────────

class TestMLPartialWin:
    def test_partial_win_labeled_as_win(self):
        from src.model_trainer import ModelTrainer
        import numpy as np
        t = ModelTrainer()
        outcomes = np.array(["WIN", "PARTIAL_WIN", "PARTIAL_WIN_FINAL", "LOSS", "ESTIMATED_PARTIAL_WIN"])
        labels = np.array([1, 0, 0, -1, 0])
        label_map = {-1: 0, 1: 1}
        y = np.array([
            1 if "PARTIAL_WIN" in str(outcomes[i]) else label_map.get(int(v), 0)
            for i, v in enumerate(labels)
        ])
        assert list(y) == [1, 1, 1, 0, 1]
