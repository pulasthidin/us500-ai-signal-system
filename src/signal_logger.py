"""
Signal persistence layer — SQLite database for all signal events + outcomes.
Thread-safe, transactional, crash-resilient.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from config import SIGNAL_DEDUP_MINUTES, OUTCOME_CHECK_DELAY_SECONDS

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("signal_logger")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/signal_logger.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

DB_PATH = os.path.join("database", "signals.db")

CREATE_SIGNALS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    sl_time TEXT,
    session TEXT,
    day_name TEXT,
    is_monday INTEGER,
    is_friday INTEGER,
    is_news_day INTEGER,

    vix_level REAL,
    vix_pct REAL,
    vix_bucket TEXT,
    vix_direction_bias TEXT,
    us10y_direction TEXT,
    oil_direction TEXT,
    dxy_direction TEXT,
    rut_direction TEXT,
    macro_bias TEXT,
    groq_sentiment TEXT,
    bullish_count INTEGER,
    bearish_count INTEGER,

    above_ema200 INTEGER,
    above_ema50 INTEGER,
    h4_bos TEXT,
    wyckoff TEXT,
    structure_bias TEXT,
    choch_recent INTEGER,
    ob_bullish_nearby INTEGER,
    ob_bearish_nearby INTEGER,

    at_zone INTEGER,
    zone_type TEXT,
    zone_level REAL,
    zone_distance REAL,
    zone_direction TEXT,
    eqh_nearby INTEGER,
    eql_nearby INTEGER,
    eqh_level REAL,
    eql_level REAL,

    delta_direction TEXT,
    divergence TEXT,
    vix_spiking_now INTEGER,
    confirms_bias INTEGER,

    fvg_present INTEGER,
    fvg_top REAL,
    fvg_bottom REAL,
    m5_bos INTEGER,
    ustec_agrees INTEGER,
    rr REAL,
    atr REAL,

    score INTEGER,
    entry_ready INTEGER,
    entry_confidence TEXT,
    direction_confidence TEXT,
    direction TEXT,
    decision TEXT,
    grade TEXT,
    size_label TEXT,
    caution_flags TEXT,

    entry_price REAL,
    sl_price REAL,
    tp_price REAL,
    sl_points REAL,
    tp_points REAL,
    tp_source TEXT,

    outcome TEXT DEFAULT NULL,
    outcome_label INTEGER DEFAULT NULL,
    bars_to_outcome INTEGER DEFAULT NULL,
    pnl_points REAL DEFAULT NULL,
    outcome_timestamp TEXT DEFAULT NULL,
    save_status TEXT DEFAULT 'complete'
);
"""

CREATE_PATTERN_ALERTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pattern_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    win_probability REAL,
    snapshot_json TEXT,
    direction TEXT,
    matched_signal_id INTEGER DEFAULT NULL,
    outcome TEXT DEFAULT NULL
);
"""


class SignalLogger:
    """SQLite-backed signal store with dedup, stats, and training-data export."""

    def __init__(self) -> None:
        self._db_path = DB_PATH
        self._dedup_lock = threading.Lock()

    # ─── database init ───────────────────────────────────────

    def init_db(self) -> None:
        """Create the signals and pattern_alerts tables if they do not exist."""
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")  # persists to file after first set
                conn.execute(CREATE_SIGNALS_TABLE_SQL)
                conn.execute(CREATE_PATTERN_ALERTS_TABLE_SQL)
                self._run_migrations(conn)
                conn.commit()
            logger.info("Database initialised at %s", self._db_path)
        except Exception as exc:
            logger.error("init_db failed: %s", exc, exc_info=True)
            raise

    @staticmethod
    def _run_migrations(conn: sqlite3.Connection) -> None:
        """Add columns that were introduced after the initial schema."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        migrations = [
            ("entry_confidence", "TEXT"),
            ("direction_confidence", "TEXT"),
            ("tp_source", "TEXT"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
                logger.info("Migration: added column '%s' to signals table", col_name)

    # ─── helpers ─────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        # WAL mode is set once in init_db and persists in the database file;
        # no need to re-issue the pragma on every connection.
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, params: tuple = (), fetchall: bool = True):
        """Execute a read query with proper connection close."""
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall() if fetchall else conn.execute(sql, params).fetchone()
            return rows
        finally:
            conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write query with proper connection close."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(sql, params)
        finally:
            conn.close()

    @staticmethod
    def _bool_to_int(val) -> Optional[int]:
        if val is None:
            return None
        return 1 if val else 0

    # ─── write ───────────────────────────────────────────────

    def log_signal(self, result: Dict[str, Any]) -> int:
        """
        Insert a new signal row atomically.
        Uses SQLite transaction: either the full row commits or nothing does.
        """
        try:
            layer1 = result.get("layer1", {})
            layer2 = result.get("layer2", {})
            layer3 = result.get("layer3", {})
            layer4 = result.get("layer4", {})
            entry = result.get("entry") or {}
            caution = json.dumps(result.get("caution_flags", []))

            conn = self._connect()
            try:
                with conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO signals (
                            timestamp, sl_time, session, day_name, is_monday, is_friday, is_news_day,
                            vix_level, vix_pct, vix_bucket, vix_direction_bias,
                            us10y_direction, oil_direction, dxy_direction, rut_direction,
                            macro_bias, groq_sentiment, bullish_count, bearish_count,
                            above_ema200, above_ema50, h4_bos, wyckoff, structure_bias, choch_recent, ob_bullish_nearby, ob_bearish_nearby,
                            at_zone, zone_type, zone_level, zone_distance, zone_direction, eqh_nearby, eql_nearby, eqh_level, eql_level,
                            delta_direction, divergence, vix_spiking_now, confirms_bias,
                            fvg_present, fvg_top, fvg_bottom, m5_bos, ustec_agrees, rr, atr,
                            score, entry_ready, entry_confidence, direction_confidence, direction, decision, grade, size_label, caution_flags,
                            entry_price, sl_price, tp_price, sl_points, tp_points, tp_source,
                            save_status
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?,
                            ?
                        )
                        """,
                        (
                            result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                            result.get("sl_time"),
                            result.get("session"),
                            result.get("day_name"),
                            self._bool_to_int(result.get("is_monday")),
                            self._bool_to_int(result.get("is_friday")),
                            self._bool_to_int(result.get("is_news_day")),

                            layer1.get("vix_value"),
                            layer1.get("vix_pct"),
                            layer1.get("vix_bucket"),
                            layer1.get("vix_direction_bias"),
                            layer1.get("us10y_direction"),
                            layer1.get("oil_direction"),
                            layer1.get("dxy_direction"),
                            layer1.get("rut_direction"),
                            layer1.get("bias"),
                            layer1.get("groq_sentiment"),
                            layer1.get("bullish_count"),
                            layer1.get("bearish_count"),

                            self._bool_to_int(layer2.get("above_ema200")),
                            self._bool_to_int(layer2.get("above_ema50")),
                            layer2.get("bos_direction"),
                            layer2.get("wyckoff"),
                            layer2.get("structure_bias"),
                            self._bool_to_int(layer2.get("choch_recent")),
                            self._bool_to_int(layer2.get("ob_bullish_nearby")),
                            self._bool_to_int(layer2.get("ob_bearish_nearby")),

                            self._bool_to_int(layer3.get("at_zone")),
                            layer3.get("zone_type"),
                            layer3.get("zone_level"),
                            layer3.get("distance"),
                            layer3.get("zone_direction"),
                            self._bool_to_int(result.get("eqh_nearby")),
                            self._bool_to_int(result.get("eql_nearby")),
                            result.get("eqh_level"),
                            result.get("eql_level"),

                            layer4.get("delta_direction"),
                            layer4.get("divergence"),
                            self._bool_to_int(layer4.get("vix_spiking_now")),
                            self._bool_to_int(layer4.get("confirms_bias")),

                            self._bool_to_int(entry.get("fvg_present")),
                            entry.get("fvg_details", {}).get("top"),
                            entry.get("fvg_details", {}).get("bottom"),
                            self._bool_to_int(entry.get("m5_bos_confirmed")),
                            self._bool_to_int(entry.get("ustec_agrees")),
                            entry.get("rr"),
                            entry.get("atr"),

                            result.get("score"),
                            self._bool_to_int(result.get("entry_ready")),
                            result.get("entry_confidence"),
                            result.get("direction_confidence"),
                            result.get("direction"),
                            result.get("decision"),
                            result.get("grade"),
                            result.get("size_label"),
                            caution,

                            result.get("current_price"),
                            entry.get("sl_price"),
                            entry.get("tp_price"),
                            entry.get("sl_points"),
                            entry.get("tp_points"),
                            entry.get("tp_source"),

                            "complete",
                        ),
                    )
                    signal_id = cursor.lastrowid
                logger.info("Logged signal id=%d  decision=%s", signal_id, result.get("decision"))
                return signal_id
            finally:
                conn.close()

        except Exception as exc:
            logger.error("log_signal failed: %s", exc, exc_info=True)
            return None

    # ─── outcome update ──────────────────────────────────────

    def update_outcome(
        self, signal_id: int, outcome: str, label: int, bars: int, pnl: float, timestamp: str
    ) -> None:
        """Write the trade outcome back to an existing signal row."""
        try:
            self._execute(
                """
                UPDATE signals
                SET outcome = ?, outcome_label = ?, bars_to_outcome = ?,
                    pnl_points = ?, outcome_timestamp = ?
                WHERE id = ?
                """,
                (outcome, label, bars, round(pnl, 2), timestamp, signal_id),
            )
            logger.info("Updated outcome for signal %d: %s  pnl=%.2f", signal_id, outcome, pnl)
        except Exception as exc:
            logger.error("update_outcome failed: %s", exc, exc_info=True)

    # ─── reads ───────────────────────────────────────────────

    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Return signals that have no outcome and are older than the check delay."""
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=OUTCOME_CHECK_DELAY_SECONDS)).isoformat()
            rows = self._query(
                "SELECT * FROM signals WHERE outcome IS NULL AND timestamp < ? ORDER BY timestamp ASC",
                (cutoff,),
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_pending_signals failed: %s", exc, exc_info=True)
            return []

    def get_recent_signals(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last *n* signals ordered newest first."""
        try:
            rows = self._query("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (n,))
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_recent_signals failed: %s", exc, exc_info=True)
            return []

    def get_signals_since(self, since_iso: str) -> List[Dict[str, Any]]:
        """Return all signals with timestamp >= *since_iso*, newest first."""
        try:
            rows = self._query(
                "SELECT * FROM signals WHERE timestamp >= ? ORDER BY id DESC",
                (since_iso,),
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_signals_since failed: %s", exc, exc_info=True)
            return []

    def get_training_data(self) -> pd.DataFrame:
        """Return all signals with an outcome as a pandas DataFrame."""
        conn = self._connect()
        try:
            return pd.read_sql_query("SELECT * FROM signals WHERE outcome IS NOT NULL", conn)
        except Exception as exc:
            logger.error("get_training_data failed: %s", exc, exc_info=True)
            return pd.DataFrame()
        finally:
            conn.close()

    def get_signal_count(self) -> int:
        """Total labelled signals."""
        try:
            row = self._query(
                "SELECT COUNT(*) as cnt FROM signals WHERE outcome IS NOT NULL",
                fetchall=False,
            )
            return row["cnt"] if row else 0
        except Exception as exc:
            logger.error("get_signal_count failed: %s", exc, exc_info=True)
            return 0

    def get_win_rate(self) -> float:
        """Win % among labelled signals.
        Counts both 'WIN' and 'ESTIMATED_WIN' (H1-fallback resolved) as wins.
        """
        try:
            row = self._query(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins
                   FROM signals WHERE outcome IS NOT NULL""",
                fetchall=False,
            )
            if not row or row["total"] == 0:
                return 0.0
            return round(row["wins"] / row["total"] * 100, 1)
        except Exception as exc:
            logger.error("get_win_rate failed: %s", exc, exc_info=True)
            return 0.0

    def get_stats_by_session(self) -> Dict[str, Any]:
        """Win rate broken down by session."""
        try:
            rows = self._query("""
                SELECT session, COUNT(*) as total,
                       SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins
                FROM signals WHERE outcome IS NOT NULL GROUP BY session
            """)
            return {
                r["session"]: {
                    "total": r["total"], "wins": r["wins"],
                    "win_rate": round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
                }
                for r in rows
            }
        except Exception as exc:
            logger.error("get_stats_by_session failed: %s", exc, exc_info=True)
            return {}

    def get_stats_by_grade(self) -> Dict[str, Any]:
        """Win rate by signal grade (A vs B)."""
        try:
            rows = self._query("""
                SELECT grade, COUNT(*) as total,
                       SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins
                FROM signals WHERE outcome IS NOT NULL AND grade IS NOT NULL GROUP BY grade
            """)
            return {
                r["grade"]: {
                    "total": r["total"], "wins": r["wins"],
                    "win_rate": round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
                }
                for r in rows
            }
        except Exception as exc:
            logger.error("get_stats_by_grade failed: %s", exc, exc_info=True)
            return {}

    def get_stats_by_direction(self) -> Dict[str, Any]:
        """Win rate and PnL broken down by LONG vs SHORT."""
        try:
            rows = self._query("""
                SELECT direction, COUNT(*) as total,
                       SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins,
                       SUM(COALESCE(pnl_points, 0)) as total_pnl
                FROM signals WHERE outcome IS NOT NULL AND direction IS NOT NULL GROUP BY direction
            """)
            return {
                r["direction"]: {
                    "total": r["total"], "wins": r["wins"],
                    "win_rate": round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
                    "total_pnl": round(r["total_pnl"], 1),
                }
                for r in rows
            }
        except Exception as exc:
            logger.error("get_stats_by_direction failed: %s", exc, exc_info=True)
            return {}

    def get_tp_source_breakdown(self) -> Dict[str, Any]:
        """Win rate per TP method (eqh, eql, pdh, pdl, round, poc, atr)."""
        try:
            rows = self._query("""
                SELECT tp_source, COUNT(*) as total,
                       SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins
                FROM signals WHERE outcome IS NOT NULL AND tp_source IS NOT NULL GROUP BY tp_source
                ORDER BY total DESC
            """)
            return {
                r["tp_source"]: {
                    "total": r["total"], "wins": r["wins"],
                    "win_rate": round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0,
                }
                for r in rows
            }
        except Exception as exc:
            logger.error("get_tp_source_breakdown failed: %s", exc, exc_info=True)
            return {}

    def get_weekly_summary(self, since_iso: str) -> Dict[str, Any]:
        """Total signals, resolved count, PnL, and win rate since *since_iso*."""
        try:
            row = self._query("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                       SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins,
                       SUM(COALESCE(pnl_points, 0)) as total_pnl
                FROM signals WHERE timestamp >= ?
            """, (since_iso,), fetchall=False)
            total = row["total"] or 0
            resolved = row["resolved"] or 0
            wins = row["wins"] or 0
            total_pnl = round(row["total_pnl"] or 0.0, 1)
            win_rate = round(wins / resolved * 100, 1) if resolved > 0 else 0.0
            return {"total": total, "resolved": resolved, "wins": wins,
                    "total_pnl": total_pnl, "win_rate": win_rate}
        except Exception as exc:
            logger.error("get_weekly_summary failed: %s", exc, exc_info=True)
            return {"total": 0, "resolved": 0, "wins": 0, "total_pnl": 0.0, "win_rate": 0.0}

    def get_win_rate_since(self, since_iso: str) -> float:
        """Win rate for signals resolved since *since_iso* (used for trend comparison)."""
        try:
            row = self._query("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins
                FROM signals WHERE outcome IS NOT NULL AND timestamp >= ?
            """, (since_iso,), fetchall=False)
            if not row or not row["total"]:
                return 0.0
            return round(row["wins"] / row["total"] * 100, 1)
        except Exception as exc:
            logger.error("get_win_rate_since failed: %s", exc, exc_info=True)
            return 0.0

    # ─── crash recovery ────────────────────────────────────────

    def get_all_null_outcome_signals(self) -> List[Dict[str, Any]]:
        """Return ALL signals with NULL outcome, no time limit, oldest first."""
        try:
            rows = self._query(
                "SELECT * FROM signals WHERE outcome IS NULL AND save_status = 'complete' ORDER BY timestamp ASC"
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_all_null_outcome_signals failed: %s", exc, exc_info=True)
            return []

    def fix_partial_saves(self) -> int:
        """
        Delete rows with save_status != 'complete' (partial writes from a crash).
        Returns the number of rows removed.
        """
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM signals WHERE save_status != 'complete'")
                removed = cursor.rowcount
            if removed > 0:
                logger.warning("Removed %d partial-save rows", removed)
            return removed
        except Exception as exc:
            logger.error("fix_partial_saves failed: %s", exc, exc_info=True)
            return 0
        finally:
            conn.close()

    def get_last_signal_timestamp(self) -> Optional[str]:
        """Return the timestamp of the most recent signal, or None if empty."""
        try:
            row = self._query("SELECT timestamp FROM signals ORDER BY id DESC LIMIT 1", fetchall=False)
            return row["timestamp"] if row else None
        except Exception as exc:
            logger.error("get_last_signal_timestamp failed: %s", exc, exc_info=True)
            return None

    # ─── dedup ───────────────────────────────────────────────

    def is_duplicate(self, direction: str, entry_price: float) -> bool:
        """True if the same direction was logged within SIGNAL_DEDUP_MINUTES.

        Uses a threading lock to prevent two concurrent checks from both
        concluding they're unique and firing duplicate alerts.

        Matching is on direction + time window only — price-based matching
        produced false negatives when price moved > 5 points between checks.
        """
        try:
            if direction is None:
                return False
            with self._dedup_lock:
                cutoff = (datetime.now(timezone.utc) - timedelta(minutes=SIGNAL_DEDUP_MINUTES)).isoformat()
                row = self._query(
                    """SELECT COUNT(*) as cnt FROM signals
                       WHERE direction = ? AND timestamp > ?""",
                    (direction, cutoff),
                    fetchall=False,
                )
                return row["cnt"] > 0 if row else False
        except Exception as exc:
            logger.error("is_duplicate failed: %s", exc, exc_info=True)
            return False

    # ─── pattern alerts ──────────────────────────────────────

    def log_pattern_alert(self, data: Dict[str, Any]) -> Optional[int]:
        """Insert a pattern-scanner early-warning alert and return its id."""
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO pattern_alerts (timestamp, win_probability, snapshot_json, direction) VALUES (?, ?, ?, ?)",
                    (data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                     data.get("win_probability"), data.get("snapshot_json"), data.get("direction")),
                )
                pa_id = cursor.lastrowid
            logger.info("Logged pattern alert id=%d", pa_id)
            return pa_id
        except Exception as exc:
            logger.error("log_pattern_alert failed: %s", exc, exc_info=True)
            return None
        finally:
            conn.close()

    def get_recent_pattern_alerts(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last *n* pattern alerts, newest first."""
        try:
            rows = self._query("SELECT * FROM pattern_alerts ORDER BY id DESC LIMIT ?", (n,))
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_recent_pattern_alerts failed: %s", exc, exc_info=True)
            return []

    def update_pattern_alert_match(self, alert_id: int, signal_id: int) -> None:
        """Record that a checklist signal confirmed this pattern alert."""
        if signal_id is None or signal_id < 0:
            return
        try:
            self._execute("UPDATE pattern_alerts SET matched_signal_id = ? WHERE id = ?", (signal_id, alert_id))
            logger.info("Pattern alert %d matched signal %d", alert_id, signal_id)
        except Exception as exc:
            logger.error("update_pattern_alert_match failed: %s", exc, exc_info=True)

    def get_pattern_alert_accuracy(self) -> Dict[str, Any]:
        """How often pattern alerts led to actual winning trades."""
        try:
            row = self._query("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN matched_signal_id IS NOT NULL THEN 1 ELSE 0 END) as matched
                FROM pattern_alerts
            """, fetchall=False)
            total = row["total"] if row else 0
            matched = row["matched"] if row else 0
            wins = 0
            if matched > 0:
                w = self._query("""
                    SELECT COUNT(*) as cnt FROM pattern_alerts pa
                    JOIN signals s ON pa.matched_signal_id = s.id
                    WHERE s.outcome IN ('WIN', 'ESTIMATED_WIN')
                """, fetchall=False)
                wins = w["cnt"] if w else 0
            match_rate = round(matched / total * 100, 1) if total > 0 else 0.0
            win_rate = round(wins / matched * 100, 1) if matched > 0 else 0.0
            return {"total_alerts": total, "matched_signals": matched, "match_rate": match_rate,
                    "wins_after_match": wins, "win_rate_after_match": win_rate}
        except Exception as exc:
            logger.error("get_pattern_alert_accuracy failed: %s", exc, exc_info=True)
            return {"total_alerts": 0, "matched_signals": 0, "match_rate": 0,
                    "wins_after_match": 0, "win_rate_after_match": 0}

    def cleanup_old_pattern_alerts(self, days: int = 7) -> int:
        """Delete unmatched pattern alerts older than *days* to prevent table bloat."""
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            conn = self._connect()
            try:
                with conn:
                    cursor = conn.execute(
                        "DELETE FROM pattern_alerts WHERE matched_signal_id IS NULL AND timestamp < ?",
                        (cutoff,),
                    )
                    removed = cursor.rowcount
                if removed > 0:
                    logger.info("Cleaned up %d old unmatched pattern alerts", removed)
                return removed
            finally:
                conn.close()
        except Exception as exc:
            logger.error("cleanup_old_pattern_alerts failed: %s", exc, exc_info=True)
            return 0

    def get_recent_unmatched_pattern_alert(self, direction: str, minutes: int = 30) -> Optional[Dict[str, Any]]:
        """Find the most recent pattern alert for *direction* within *minutes* that hasn't been matched yet."""
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
            row = self._query(
                """SELECT * FROM pattern_alerts
                   WHERE direction = ? AND timestamp > ? AND matched_signal_id IS NULL
                   ORDER BY id DESC LIMIT 1""",
                (direction, cutoff), fetchall=False,
            )
            return dict(row) if row else None
        except Exception as exc:
            logger.error("get_recent_unmatched_pattern_alert failed: %s", exc, exc_info=True)
            return None
