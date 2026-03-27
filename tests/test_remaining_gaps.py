"""Tests for remaining coverage gaps across SignalLogger, EvolutionManager, OutcomeTracker, PatternScanner."""

import json
import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.signal_logger import SignalLogger
from src.evolution_manager import EvolutionManager, STAGE_THRESHOLDS
from src.outcome_tracker import OutcomeTracker
from src.pattern_scanner import PatternScanner


# ─── shared helpers ──────────────────────────────────────────

def _make_result(direction="LONG", price=6500.0, score=4, decision="FULL_SEND",
                 grade="A", tp_source="atr", session="London"):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sl_time": "14:00 SL",
        "session": session,
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
        "layer2": {"above_ema50": True, "bos_direction": "bullish",
                    "wyckoff": "markup", "structure_bias": "long", "choch_recent": False},
        "layer3": {"at_zone": True, "zone_type": "round", "zone_level": 6500.0,
                    "distance": 3.0, "zone_direction": "support"},
        "layer4": {"delta_direction": "buyers", "divergence": "none",
                    "vix_spiking_now": False, "confirms_bias": True},
        "entry": {
            "fvg_present": True, "fvg_details": {"top": 6510, "bottom": 6505},
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr": 2.5, "atr": 10.0,
            "sl_price": 6485.0, "tp_price": 6525.0,
            "sl_points": 15.0, "tp_points": 25.0,
            "tp_source": tp_source,
        },
        "score": score,
        "entry_ready": True,
        "direction": direction,
        "decision": decision,
        "grade": grade,
        "size_label": "normal",
        "caution_flags": [],
    }


@pytest.fixture
def logger_db(tmp_path):
    sl = SignalLogger()
    sl._db_path = str(tmp_path / "test_signals.db")
    sl.init_db()
    return sl


# ═════════════════════════════════════════════════════════════
# SignalLogger — stats & cleanup
# ═════════════════════════════════════════════════════════════

class TestSignalLoggerStats:
    def test_get_signals_since_filters_by_date(self, logger_db):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        mid_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()

        for ts in (old_ts, mid_ts, new_ts):
            r = _make_result()
            r["timestamp"] = ts
            logger_db.log_signal(r)

        cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        results = logger_db.get_signals_since(cutoff)
        assert len(results) == 2

    def test_get_win_rate_since(self, logger_db):
        now = datetime.now(timezone.utc)
        for i in range(5):
            r = _make_result()
            r["timestamp"] = (now - timedelta(hours=i)).isoformat()
            sid = logger_db.log_signal(r)
            outcome = "WIN" if i < 3 else "LOSS"
            label = 1 if outcome == "WIN" else 0
            logger_db.update_outcome(sid, outcome, label, 10, 20.0 if label == 1 else -15.0, now.isoformat())

        cutoff = (now - timedelta(days=1)).isoformat()
        rate = logger_db.get_win_rate_since(cutoff)
        assert rate == 60.0  # 3 wins / 5 total

    def test_get_tp_source_breakdown(self, logger_db):
        now = datetime.now(timezone.utc)
        sources = [("atr", "WIN", 1), ("eqh", "WIN", 1), ("eqh", "LOSS", 0), ("atr", "LOSS", 0)]
        for tp_src, outcome, label in sources:
            r = _make_result(tp_source=tp_src)
            r["timestamp"] = now.isoformat()
            sid = logger_db.log_signal(r)
            logger_db.update_outcome(sid, outcome, label, 10, 20.0 if label == 1 else -15.0, now.isoformat())

        breakdown = logger_db.get_tp_source_breakdown()
        assert "atr" in breakdown
        assert "eqh" in breakdown
        assert breakdown["atr"]["total"] == 2
        assert breakdown["atr"]["win_rate"] == 50.0
        assert breakdown["eqh"]["total"] == 2
        assert breakdown["eqh"]["win_rate"] == 50.0

    def test_cleanup_old_pattern_alerts(self, logger_db):
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=3)).isoformat()
        recent_ts = now.isoformat()

        logger_db.log_pattern_alert({"timestamp": old_ts, "direction": "LONG", "win_probability": 0.8})
        logger_db.log_pattern_alert({"timestamp": old_ts, "direction": "SHORT", "win_probability": 0.7})
        logger_db.log_pattern_alert({"timestamp": recent_ts, "direction": "LONG", "win_probability": 0.9})

        removed = logger_db.cleanup_old_pattern_alerts(days=1)
        assert removed == 2
        remaining = logger_db.get_recent_pattern_alerts(10)
        assert len(remaining) == 1
        assert remaining[0]["win_probability"] == 0.9

    def test_run_migrations_partial_win_fix(self, tmp_path):
        db_path = str(tmp_path / "migration_test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, direction TEXT, score INTEGER,
            outcome TEXT DEFAULT NULL, outcome_label INTEGER DEFAULT NULL,
            save_status TEXT DEFAULT 'complete'
        )""")
        conn.execute("""CREATE TABLE pattern_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
            win_probability REAL, snapshot_json TEXT, direction TEXT,
            matched_signal_id INTEGER DEFAULT NULL, outcome TEXT DEFAULT NULL
        )""")
        conn.execute(
            "INSERT INTO signals (timestamp, direction, outcome, outcome_label) VALUES (?, ?, ?, ?)",
            ("2025-01-01T00:00:00", "LONG", "PARTIAL_WIN", 0),
        )
        conn.commit()
        conn.close()

        sl = SignalLogger()
        sl._db_path = db_path
        sl.init_db()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT outcome_label FROM signals WHERE id = 1").fetchone()
        conn.close()
        assert row["outcome_label"] == 1


# ═════════════════════════════════════════════════════════════
# EvolutionManager — stage-skip bug
# ═════════════════════════════════════════════════════════════

class TestEvolutionManagerStageSkip:
    @pytest.fixture
    def evo(self, tmp_path, mock_alert_bot):
        sig_log = MagicMock()
        trainer = MagicMock()
        mgr = EvolutionManager(sig_log, trainer, alert_bot=mock_alert_bot)
        stage_file = str(tmp_path / "stage.json")
        import src.evolution_manager as mod
        mod.STAGE_FILE = stage_file
        return mgr, sig_log, trainer, mock_alert_bot, stage_file

    def test_stage_1_failure_blocks_stage_2(self, evo):
        mgr, sig_log, trainer, bot, _ = evo
        sig_log.get_signal_count.return_value = 600
        trainer.train_meta_label_model.return_value = False

        result = mgr.check_evolution_stage()
        assert result.get("stage", 0) == 0
        trainer.train_meta_label_model.assert_called_once()

    def test_normal_stage_progression(self, evo):
        mgr, sig_log, trainer, bot, _ = evo
        trainer.train_meta_label_model.return_value = True

        sig_log.get_signal_count.return_value = 200
        r1 = mgr.check_evolution_stage()
        assert r1["stage"] == 1

        sig_log.get_signal_count.return_value = 500
        r2 = mgr.check_evolution_stage()
        assert r2["stage"] == 2
        assert r2["stage"] >= r1["stage"]


# ═════════════════════════════════════════════════════════════
# OutcomeTracker — track_pending_outcomes loop
# ═════════════════════════════════════════════════════════════

class TestOutcomeTrackerLoop:
    def test_track_pending_outcomes_processes_signals(self, mock_ctrader, mock_alert_bot):
        import pandas as pd
        now = datetime.now(timezone.utc)
        sig_log = MagicMock()
        sig_log.get_pending_signals.return_value = [
            {"id": 1, "direction": "LONG", "entry_price": 6500, "sl_price": 6485,
             "tp_price": 6525, "outcome": None,
             "timestamp": (now - timedelta(hours=3)).isoformat()},
            {"id": 2, "direction": "SHORT", "entry_price": 6500, "sl_price": 6515,
             "tp_price": 6475, "outcome": None,
             "timestamp": (now - timedelta(hours=3)).isoformat()},
        ]
        sig_log.get_all_null_outcome_signals.return_value = []
        sig_log.get_partial_win_signals.return_value = []

        n = 100
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=9), periods=n, freq="5min", tz="UTC"),
            "open": [6500] * n, "high": [6530] * n,
            "low": [6470] * n, "close": [6510] * n, "volume": [1000] * n,
        })
        mock_ctrader.fetch_bars = MagicMock(return_value=bars)

        tracker = OutcomeTracker(mock_ctrader, us500_id=1, signal_logger=sig_log, alert_bot=mock_alert_bot)
        tracker.track_pending_outcomes()

        assert sig_log.update_outcome.call_count >= 1

    def test_track_pending_outcomes_handles_empty_list(self, mock_ctrader, mock_alert_bot):
        sig_log = MagicMock()
        sig_log.get_pending_signals.return_value = []
        sig_log.get_all_null_outcome_signals.return_value = []
        sig_log.get_partial_win_signals.return_value = []

        tracker = OutcomeTracker(mock_ctrader, us500_id=1, signal_logger=sig_log, alert_bot=mock_alert_bot)
        tracker.track_pending_outcomes()

        sig_log.update_outcome.assert_not_called()


# ═════════════════════════════════════════════════════════════
# PatternScanner — error handling
# ═════════════════════════════════════════════════════════════

class TestPatternScannerErrors:
    def test_run_pattern_scan_catches_exception(self, tmp_path, mock_alert_bot):
        import src.pattern_scanner as mod
        stage_file = str(tmp_path / "stage.json")
        mod.STAGE_FILE = stage_file
        with open(stage_file, "w") as f:
            json.dump({"stage": 2}, f)

        checklist = MagicMock()
        checklist.is_trading_session.return_value = True

        macro = MagicMock()
        macro.get_layer1_result.side_effect = RuntimeError("API down")

        ps = PatternScanner(
            macro_checker=macro,
            zone_calculator=MagicMock(),
            orderflow_analyzer=MagicMock(),
            checklist_engine=checklist,
            ctrader_connection=MagicMock(),
            us500_id=1,
            signal_logger=MagicMock(),
            model_predictor=MagicMock(),
            alert_bot=mock_alert_bot,
        )

        ps.run_pattern_scan()
        mock_alert_bot.send_trade_message.assert_not_called()
