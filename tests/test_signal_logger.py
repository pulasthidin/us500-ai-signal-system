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
            "above_ema50": True,
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


# ─── New columns (range, sweep, displacement, ML features) ─────

def _make_result_v2(direction="LONG", price=6500.0, score=4, decision="FULL_SEND", grade="A"):
    """Full result dict including all v2 columns."""
    base = _make_result(direction, price, score, decision, grade)
    base["range_condition"] = {
        "is_ranging": True, "range_strength": "strong",
        "adx_value": 15.3, "adx_ranging": True,
        "atr_compressing": True, "atr_ratio": 0.65,
        "range_high": 6530.0, "range_low": 6470.0,
    }
    base["is_ranging"] = True
    base["has_liquidity_sweep"] = True
    base["layer2"]["range_condition"] = base["range_condition"]
    base["layer2"]["ob_bullish_nearby"] = True
    base["layer2"]["ob_bearish_nearby"] = False
    base["layer3"]["asian_range"] = {"asian_high": 6510.0, "asian_low": 6490.0, "valid": True}
    base["layer3"]["liquidity_sweeps"] = [
        {"level_price": 6470.0, "level_type": "pdl", "sweep_side": "sell_side", "bars_ago": 5}
    ]
    base["entry"] = {
        "fvg_present": True,
        "fvg_details": {"top": 6510.0, "bottom": 6505.0, "age_bars": 3, "displacement_valid": True},
        "m5_bos_confirmed": True, "ustec_agrees": True,
        "rr": 2.5, "atr": 10.0,
        "sl_price": 6465.0, "tp_price": 6525.0,
        "sl_points": 35.0, "tp_points": 25.0,
        "tp_source": "eqh",
        "displacement_valid": True,
        "has_liquidity_sweep": True,
        "sweep_details": {"level_type": "pdl", "sweep_side": "sell_side", "bars_ago": 5},
        "sl_method": "range",
        "entry_confidence": "full",
    }
    base["entry_confidence"] = "full"
    base["direction_confidence"] = "high"
    return base


class TestNewColumnsV2:
    def test_log_signal_stores_range_data(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        assert sid is not None
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["is_ranging"] == 1
        assert row["range_strength"] == "strong"
        assert abs(row["adx_value"] - 15.3) < 0.1
        assert row["atr_compressing"] == 1

    def test_log_signal_stores_sweep_data(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["has_liquidity_sweep"] == 1
        assert row["sweep_side"] == "sell_side"
        assert row["sweep_level_type"] == "pdl"
        assert row["sweep_bars_ago"] == 5

    def test_log_signal_stores_displacement(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["displacement_valid"] == 1
        assert row["sl_method"] == "range"

    def test_log_signal_stores_computed_features(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["fvg_age_bars"] == 3
        assert abs(row["fvg_size_points"] - 5.0) < 0.1
        assert abs(row["range_size_points"] - 60.0) < 0.1
        assert row["price_in_range_pct"] is not None
        assert 0 <= row["price_in_range_pct"] <= 100
        assert abs(row["asian_range_size"] - 20.0) < 0.1
        assert row["hour_utc"] is not None

    def test_data_version_is_2_for_new_signals(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT data_version FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert row["data_version"] == 2

    def test_asian_range_stored(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT asian_high, asian_low FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert abs(row["asian_high"] - 6510.0) < 0.1
        assert abs(row["asian_low"] - 6490.0) < 0.1

    def test_range_boundaries_stored(self, logger_db):
        import sqlite3
        sid = logger_db.log_signal(_make_result_v2())
        conn = sqlite3.connect(logger_db._db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT range_high, range_low FROM signals WHERE id = ?", (sid,)).fetchone()
        conn.close()
        assert abs(row["range_high"] - 6530.0) < 0.1
        assert abs(row["range_low"] - 6470.0) < 0.1


class TestMigration:
    def test_migration_adds_new_columns_to_old_db(self, tmp_path):
        """Simulate an old DB without new columns and verify migration adds them."""
        import sqlite3
        db_path = str(tmp_path / "old_signals.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, direction TEXT, score INTEGER,
            outcome TEXT DEFAULT NULL, save_status TEXT DEFAULT 'complete'
        )""")
        conn.execute("""CREATE TABLE pattern_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT
        )""")
        conn.commit()
        conn.close()

        sl = SignalLogger()
        sl._db_path = db_path
        sl.init_db()

        conn = sqlite3.connect(db_path)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        conn.close()

        for col in ("is_ranging", "adx_value", "has_liquidity_sweep",
                     "displacement_valid", "data_version", "fvg_age_bars",
                     "range_size_points", "hour_utc", "sweep_bars_ago",
                     "asian_range_size", "sl_method"):
            assert col in columns, f"Migration did not add column: {col}"

    def test_old_rows_get_data_version_1(self, tmp_path):
        """data_version migration DEFAULT is 1 for existing rows."""
        import sqlite3
        db_path = str(tmp_path / "old_signals2.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, direction TEXT, score INTEGER,
            outcome TEXT DEFAULT NULL, save_status TEXT DEFAULT 'complete'
        )""")
        conn.execute("""CREATE TABLE pattern_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT
        )""")
        conn.execute("INSERT INTO signals (timestamp, direction, score) VALUES ('2025-01-01', 'LONG', 4)")
        conn.commit()
        conn.close()

        sl = SignalLogger()
        sl._db_path = db_path
        sl.init_db()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT data_version FROM signals LIMIT 1").fetchone()
        conn.close()
        assert row["data_version"] == 1


class TestTrainingDataExport:
    def test_get_training_data_includes_new_columns(self, logger_db):
        logger_db.log_signal(_make_result_v2())
        logger_db.update_outcome(1, "WIN", 1, 10, 25.0, datetime.now(timezone.utc).isoformat())
        df = logger_db.get_training_data()
        assert not df.empty
        for col in ("is_ranging", "adx_value", "has_liquidity_sweep",
                     "displacement_valid", "data_version", "fvg_size_points"):
            assert col in df.columns, f"Training data missing column: {col}"

    def test_get_stats_by_direction(self, logger_db):
        logger_db.log_signal(_make_result_v2("LONG"))
        logger_db.update_outcome(1, "WIN", 1, 5, 25.0, datetime.now(timezone.utc).isoformat())
        logger_db.log_signal(_make_result_v2("SHORT"))
        logger_db.update_outcome(2, "LOSS", -1, 3, -15.0, datetime.now(timezone.utc).isoformat())
        stats = logger_db.get_stats_by_direction()
        assert "LONG" in stats
        assert "SHORT" in stats
        assert stats["LONG"]["wins"] == 1
        assert stats["SHORT"]["wins"] == 0

    def test_get_weekly_summary(self, logger_db):
        logger_db.log_signal(_make_result_v2())
        logger_db.update_outcome(1, "WIN", 1, 5, 25.0, datetime.now(timezone.utc).isoformat())
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        summary = logger_db.get_weekly_summary(cutoff)
        assert summary["total"] >= 1
        assert summary["wins"] >= 1


def _result_ts(ago: timedelta, **kwargs):
    r = _make_result(**kwargs)
    r["timestamp"] = (datetime.now(timezone.utc) - ago).isoformat()
    return r


class TestGetRecentContext:
    def test_empty_db_returns_zeros(self, logger_db):
        ctx = logger_db.get_recent_context("LONG")
        assert ctx["signals_last_60min"] == 0
        assert ctx["losses_last_60min"] == 0
        assert ctx["minutes_since_last_signal"] is None
        assert ctx["recent_avg_pnl"] is None

    def test_counts_signals_in_last_60_min(self, logger_db):
        logger_db.log_signal(_result_ts(timedelta(minutes=30)))
        logger_db.log_signal(_result_ts(timedelta(minutes=10)))
        logger_db.log_signal(_result_ts(timedelta(hours=2)))
        ctx = logger_db.get_recent_context("LONG")
        assert ctx["signals_last_60min"] == 2

    def test_counts_losses_same_direction(self, logger_db):
        now_iso = datetime.now(timezone.utc).isoformat()
        s1 = logger_db.log_signal(_result_ts(timedelta(minutes=20), direction="SHORT"))
        s2 = logger_db.log_signal(_result_ts(timedelta(minutes=15), direction="SHORT"))
        s3 = logger_db.log_signal(_result_ts(timedelta(minutes=5), direction="SHORT"))
        logger_db.update_outcome(s1, "LOSS", -1, 5, -10.0, now_iso)
        logger_db.update_outcome(s2, "LOSS", -1, 5, -10.0, now_iso)
        logger_db.update_outcome(s3, "WIN", 1, 5, 10.0, now_iso)
        ctx = logger_db.get_recent_context("SHORT")
        assert ctx["losses_last_60min"] == 2

    def test_losses_ignores_other_direction(self, logger_db):
        now_iso = datetime.now(timezone.utc).isoformat()
        logger_db.update_outcome(
            logger_db.log_signal(_result_ts(timedelta(minutes=12), direction="SHORT")),
            "LOSS", -1, 3, -5.0, now_iso,
        )
        logger_db.update_outcome(
            logger_db.log_signal(_result_ts(timedelta(minutes=11), direction="SHORT")),
            "LOSS", -1, 3, -5.0, now_iso,
        )
        logger_db.update_outcome(
            logger_db.log_signal(_result_ts(timedelta(minutes=10), direction="LONG")),
            "LOSS", -1, 3, -5.0, now_iso,
        )
        ctx = logger_db.get_recent_context("SHORT")
        assert ctx["losses_last_60min"] == 2

    def test_minutes_since_last_signal(self, logger_db):
        logger_db.log_signal(_result_ts(timedelta(minutes=15)))
        ctx = logger_db.get_recent_context("LONG")
        assert ctx["minutes_since_last_signal"] is not None
        assert 14.0 <= ctx["minutes_since_last_signal"] <= 16.0

    def test_recent_avg_pnl(self, logger_db):
        now_iso = datetime.now(timezone.utc).isoformat()
        pnls = [10.0, -5.0, 15.0, -20.0, 10.0]
        for pnl in pnls:
            sid = logger_db.log_signal(_make_result())
            logger_db.update_outcome(sid, "WIN" if pnl > 0 else "LOSS", 1 if pnl > 0 else -1, 5, pnl, now_iso)
        ctx = logger_db.get_recent_context("LONG")
        assert ctx["recent_avg_pnl"] == 2.0

    def test_recent_avg_pnl_fewer_than_5(self, logger_db):
        now_iso = datetime.now(timezone.utc).isoformat()
        s1 = logger_db.log_signal(_make_result())
        s2 = logger_db.log_signal(_make_result())
        logger_db.update_outcome(s1, "WIN", 1, 5, 10.0, now_iso)
        logger_db.update_outcome(s2, "WIN", 1, 5, 30.0, now_iso)
        ctx = logger_db.get_recent_context("LONG")
        assert ctx["recent_avg_pnl"] == 20.0

    def test_error_returns_safe_fallback(self, logger_db, monkeypatch):
        def _boom(*_a, **_kw):
            raise RuntimeError("db closed")

        monkeypatch.setattr(logger_db, "_query", _boom)
        ctx = logger_db.get_recent_context("LONG")
        assert ctx == {
            "signals_last_60min": None,
            "losses_last_60min": None,
            "minutes_since_last_signal": None,
            "recent_avg_pnl": None,
        }
