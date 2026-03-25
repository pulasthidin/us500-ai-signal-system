"""Tests for SignalLogger — database init, logging, dedup, stats."""

import os
import pytest
from datetime import datetime, timezone, timedelta
from src.signal_logger import SignalLogger


@pytest.fixture
def logger_db(tmp_path):
    """SignalLogger with an isolated temp database."""
    sl = SignalLogger()
    sl._db_path = str(tmp_path / "test_signals.db")
    sl.init_db()
    return sl


def _make_result(direction="LONG", price=6500.0, score=4, decision="FULL_SEND", grade="A"):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sl_time": "14:00 SL",
        "session": "London",
        "day_name": "Tuesday",
        "is_monday": False,
        "is_friday": False,
        "is_news_day": False,
        "current_price": price,
        "layer1": {
            "vix_value": 17.0, "vix_pct": -2.0, "vix_bucket": "normal",
            "vix_direction_bias": "BUY_BIAS", "us10y_direction": "falling",
            "oil_direction": "falling", "dxy_direction": "flat",
            "rut_direction": "green", "bias": "LONG",
            "groq_sentiment": "NEUTRAL", "bullish_count": 3, "bearish_count": 0,
        },
        "layer2": {
            "above_ema200": True, "above_ema50": True,
            "bos_direction": "bullish", "wyckoff": "markup",
            "structure_bias": "long", "choch_recent": False,
        },
        "layer3": {
            "at_zone": True, "zone_type": "round", "zone_level": 6500.0,
            "distance": 3.0, "zone_direction": "support",
        },
        "layer4": {
            "delta_direction": "buyers", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": True,
        },
        "entry": {
            "fvg_present": True, "fvg_details": {"top": 6510, "bottom": 6505},
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr": 2.5, "atr": 10.0,
            "sl_price": 6485.0, "tp_price": 6525.0,
            "sl_points": 15.0, "tp_points": 25.0,
            "tp_source": "atr",
        },
        "score": score,
        "entry_ready": True,
        "direction": direction,
        "decision": decision,
        "grade": grade,
        "size_label": "normal",
        "caution_flags": [],
    }


class TestInitDB:
    def test_creates_table(self, logger_db):
        assert os.path.exists(logger_db._db_path)

    def test_double_init_safe(self, logger_db):
        logger_db.init_db()


class TestLogSignal:
    def test_log_returns_positive_id(self, logger_db):
        sid = logger_db.log_signal(_make_result())
        assert sid > 0

    def test_log_multiple_signals(self, logger_db):
        id1 = logger_db.log_signal(_make_result())
        id2 = logger_db.log_signal(_make_result(direction="SHORT"))
        assert id2 > id1

    def test_tp_source_saved(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT tp_source FROM signals WHERE id=?", (sid,)).fetchone()
        conn.close()
        assert row["tp_source"] == "atr"

    def test_tp_source_none_when_not_provided(self, logger_db):
        import sqlite3
        result = _make_result()
        result["entry"]["tp_source"] = None
        sid = logger_db.log_signal(result)
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT tp_source FROM signals WHERE id=?", (sid,)).fetchone()
        conn.close()
        assert row["tp_source"] is None


class TestOutcomeUpdate:
    def test_update_outcome(self, logger_db):
        sid = logger_db.log_signal(_make_result())
        logger_db.update_outcome(sid, "WIN", 1, 12, 25.0, datetime.now(timezone.utc).isoformat())
        signals = logger_db.get_recent_signals(1)
        assert signals[0]["outcome"] == "WIN"
        assert signals[0]["pnl_points"] == 25.0


class TestStats:
    def test_win_rate_empty_db(self, logger_db):
        assert logger_db.get_win_rate() == 0.0

    def test_win_rate_calculation(self, logger_db):
        for i in range(4):
            sid = logger_db.log_signal(_make_result())
            outcome = "WIN" if i < 3 else "LOSS"
            label = 1 if outcome == "WIN" else -1
            logger_db.update_outcome(sid, outcome, label, 10, 20.0 if label == 1 else -15.0,
                                     datetime.now(timezone.utc).isoformat())
        assert logger_db.get_win_rate() == 75.0

    def test_signal_count(self, logger_db):
        sid = logger_db.log_signal(_make_result())
        logger_db.update_outcome(sid, "WIN", 1, 10, 20.0, datetime.now(timezone.utc).isoformat())
        assert logger_db.get_signal_count() == 1

    def test_stats_by_session(self, logger_db):
        sid = logger_db.log_signal(_make_result())
        logger_db.update_outcome(sid, "WIN", 1, 10, 20.0, datetime.now(timezone.utc).isoformat())
        stats = logger_db.get_stats_by_session()
        assert "London" in stats

    def test_stats_by_grade(self, logger_db):
        sid = logger_db.log_signal(_make_result(grade="A"))
        logger_db.update_outcome(sid, "WIN", 1, 10, 20.0, datetime.now(timezone.utc).isoformat())
        stats = logger_db.get_stats_by_grade()
        assert "A" in stats


class TestIsDuplicate:
    def test_not_duplicate_initially(self, logger_db):
        assert logger_db.is_duplicate("LONG", 6500.0) is False

    def test_duplicate_after_logging(self, logger_db):
        logger_db.log_signal(_make_result())
        assert logger_db.is_duplicate("LONG", 6502.0) is True

    def test_different_direction_not_duplicate(self, logger_db):
        logger_db.log_signal(_make_result(direction="LONG"))
        assert logger_db.is_duplicate("SHORT", 6500.0) is False

    def test_none_direction_safe(self, logger_db):
        assert logger_db.is_duplicate(None, 6500.0) is False

    def test_none_price_safe(self, logger_db):
        assert logger_db.is_duplicate("LONG", None) is False


# ─── pattern alerts ──────────────────────────────────────────

class TestLogPatternAlert:
    def test_log_returns_positive_id(self, logger_db):
        pa_id = logger_db.log_pattern_alert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "win_probability": 0.85,
            "snapshot_json": "{}",
            "direction": "LONG",
        })
        assert pa_id > 0

    def test_log_multiple_alerts(self, logger_db):
        id1 = logger_db.log_pattern_alert({"direction": "LONG", "win_probability": 0.8})
        id2 = logger_db.log_pattern_alert({"direction": "SHORT", "win_probability": 0.9})
        assert id2 > id1


class TestGetRecentPatternAlerts:
    def test_returns_logged_alerts(self, logger_db):
        logger_db.log_pattern_alert({"direction": "LONG", "win_probability": 0.85})
        logger_db.log_pattern_alert({"direction": "SHORT", "win_probability": 0.90})
        alerts = logger_db.get_recent_pattern_alerts(10)
        assert len(alerts) == 2
        assert alerts[0]["direction"] == "SHORT"

    def test_empty_when_none_logged(self, logger_db):
        assert logger_db.get_recent_pattern_alerts(10) == []


class TestUpdatePatternAlertMatch:
    def test_match_updates_column(self, logger_db):
        pa_id = logger_db.log_pattern_alert({"direction": "LONG", "win_probability": 0.85})
        sig_id = logger_db.log_signal(_make_result())
        logger_db.update_pattern_alert_match(pa_id, sig_id)
        alerts = logger_db.get_recent_pattern_alerts(1)
        assert alerts[0]["matched_signal_id"] == sig_id


class TestPatternAlertAccuracy:
    def test_empty_db_returns_zeros(self, logger_db):
        stats = logger_db.get_pattern_alert_accuracy()
        assert stats["total_alerts"] == 0
        assert stats["match_rate"] == 0

    def test_calculates_match_rate(self, logger_db):
        pa1 = logger_db.log_pattern_alert({"direction": "LONG", "win_probability": 0.85})
        pa2 = logger_db.log_pattern_alert({"direction": "SHORT", "win_probability": 0.80})
        sid = logger_db.log_signal(_make_result())
        logger_db.update_outcome(sid, "WIN", 1, 10, 20.0, datetime.now(timezone.utc).isoformat())
        logger_db.update_pattern_alert_match(pa1, sid)
        stats = logger_db.get_pattern_alert_accuracy()
        assert stats["total_alerts"] == 2
        assert stats["matched_signals"] == 1
        assert stats["match_rate"] == 50.0
        assert stats["wins_after_match"] == 1
        assert stats["win_rate_after_match"] == 100.0


class TestGetRecentUnmatchedPatternAlert:
    def test_finds_recent_unmatched(self, logger_db):
        logger_db.log_pattern_alert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "LONG", "win_probability": 0.85,
        })
        result = logger_db.get_recent_unmatched_pattern_alert("LONG", minutes=30)
        assert result is not None
        assert result["direction"] == "LONG"

    def test_returns_none_for_wrong_direction(self, logger_db):
        logger_db.log_pattern_alert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "SHORT", "win_probability": 0.85,
        })
        assert logger_db.get_recent_unmatched_pattern_alert("LONG") is None

    def test_returns_none_when_already_matched(self, logger_db):
        pa_id = logger_db.log_pattern_alert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "LONG", "win_probability": 0.85,
        })
        logger_db.update_pattern_alert_match(pa_id, 999)
        assert logger_db.get_recent_unmatched_pattern_alert("LONG") is None

    def test_returns_none_when_too_old(self, logger_db):
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=60)).isoformat()
        logger_db.log_pattern_alert({
            "timestamp": old_ts, "direction": "LONG", "win_probability": 0.85,
        })
        assert logger_db.get_recent_unmatched_pattern_alert("LONG", minutes=30) is None


# ─── crash recovery ──────────────────────────────────────────

class TestGetAllNullOutcomeSignals:
    def test_returns_all_null_signals(self, logger_db):
        logger_db.log_signal(_make_result())
        logger_db.log_signal(_make_result(direction="SHORT"))
        results = logger_db.get_all_null_outcome_signals()
        assert len(results) == 2

    def test_excludes_resolved_signals(self, logger_db):
        sid = logger_db.log_signal(_make_result())
        logger_db.update_outcome(sid, "WIN", 1, 10, 20.0, datetime.now(timezone.utc).isoformat())
        logger_db.log_signal(_make_result(direction="SHORT"))
        results = logger_db.get_all_null_outcome_signals()
        assert len(results) == 1
        assert results[0]["direction"] == "SHORT"

    def test_empty_db(self, logger_db):
        assert logger_db.get_all_null_outcome_signals() == []


class TestFixPartialSaves:
    def test_removes_partial_rows(self, logger_db):
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute(
            "INSERT INTO signals (timestamp, direction, save_status) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), "LONG", "partial"),
        )
        conn.commit()
        conn.close()
        removed = logger_db.fix_partial_saves()
        assert removed == 1

    def test_keeps_complete_rows(self, logger_db):
        logger_db.log_signal(_make_result())
        removed = logger_db.fix_partial_saves()
        assert removed == 0
        assert len(logger_db.get_recent_signals(10)) == 1

    def test_zero_on_empty_db(self, logger_db):
        assert logger_db.fix_partial_saves() == 0


class TestGetLastSignalTimestamp:
    def test_returns_latest(self, logger_db):
        logger_db.log_signal(_make_result())
        ts = logger_db.get_last_signal_timestamp()
        assert ts is not None

    def test_returns_none_empty_db(self, logger_db):
        assert logger_db.get_last_signal_timestamp() is None


class TestTransactionalSave:
    def test_log_signal_has_complete_status(self, logger_db):
        logger_db.log_signal(_make_result())
        import sqlite3
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT save_status FROM signals ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        assert row["save_status"] == "complete"
