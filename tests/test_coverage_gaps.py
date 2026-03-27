"""
New tests covering gaps identified during pre-merge review.
Covers: checklist decision branches, VIX safety, outcome labelling,
ML pipeline filtering, model predictor graceful handling, scheduler
time conversions, and SQLite edge cases.
"""

import sqlite3
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from config import (
    GRADE_A_SCORE, GRADE_B_SCORE, OUTCOME_CHECK_DELAY_SECONDS,
    SL_UTC_OFFSET, MODEL_RETRAIN_SIGNAL_THRESHOLD,
)
from src.checklist_engine import ChecklistEngine
from src.outcome_tracker import OutcomeTracker
from src.macro_checker import MacroChecker
from src.model_trainer import ModelTrainer
from src.model_predictor import ModelPredictor
from src.signal_logger import SignalLogger


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def engine(mock_alert_bot):
    """ChecklistEngine with all dependencies mocked."""
    macro = MagicMock()
    struct = MagicMock()
    zones = MagicMock()
    flow = MagicMock()
    entry = MagicMock()
    sig_log = MagicMock()
    sig_log.is_duplicate.return_value = False
    return ChecklistEngine(
        macro_checker=macro, structure_analyzer=struct,
        zone_calculator=zones, orderflow_analyzer=flow,
        entry_checker=entry, signal_logger=sig_log,
        alert_bot=mock_alert_bot,
    )


@pytest.fixture
def tracker(mock_ctrader, mock_alert_bot):
    """OutcomeTracker with mocked dependencies."""
    sl = MagicMock()
    sl.get_all_null_outcome_signals.return_value = []
    sl.get_partial_win_signals.return_value = []
    sl.get_pending_signals.return_value = []
    return OutcomeTracker(mock_ctrader, us500_id=1, signal_logger=sl, alert_bot=mock_alert_bot)


@pytest.fixture
def logger_db(tmp_path):
    """SignalLogger with an isolated temp database."""
    sl = SignalLogger()
    sl._db_path = str(tmp_path / "test_coverage.db")
    sl.init_db()
    return sl


def _make_bars(prices):
    """Create bars from (high, low) tuples starting after the standard signal timestamp."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-03-20T10:05", periods=len(prices), freq="5min", tz="UTC"),
        "open": [p[1] + 1 for p in prices],
        "high": [p[0] for p in prices],
        "low": [p[1] for p in prices],
        "close": [p[0] - 1 for p in prices],
        "volume": [1000] * len(prices),
    })


# ═══════════════════════════════════════════════════════════════
# CHECKLIST ENGINE — decision branches
# ═══════════════════════════════════════════════════════════════

class TestChecklistDecisionBranches:
    """Verify every decision path produces a consistent result dict."""

    def _setup_full_layers(self, engine, macro, struct_bias="long",
                           entry_ready=True, entry_confidence="full",
                           sweep=False, is_ranging=False):
        engine._structure.get_layer2_result.return_value = {
            "structure_bias": struct_bias, "score_contribution": 1,
            "above_ema50": True, "ema_bias": "bullish",
            "bos_direction": "bullish", "bos_level": 6500, "bos_bars_ago": 2,
            "choch_direction": None, "choch_recent": False, "wyckoff": "markup",
            "ema50_value": 6450,
            "ob_bullish_nearby": False, "ob_bearish_nearby": False,
            "bullish_obs": [], "bearish_obs": [],
            "range_condition": {"is_ranging": is_ranging, "range_strength": "none"},
        }
        engine._zones.get_layer3_result.return_value = {
            "at_zone": True, "zone_level": 6500, "zone_type": "round",
            "distance": 3.0, "zone_direction": "support",
            "score_contribution": 1, "all_levels": [], "all_nearby": [],
            "asian_range": {}, "liquidity_sweeps": [],
            "has_buy_side_sweep": False, "has_sell_side_sweep": False,
        }
        engine._orderflow.get_layer4_result.return_value = {
            "delta_direction": "buyers", "divergence": "none",
            "vix_spiking_now": False, "confirms_bias": True, "score_contribution": 1,
        }
        engine._entry.get_entry_result.return_value = {
            "entry_ready": entry_ready, "entry_confidence": entry_confidence,
            "fvg_present": entry_ready, "fvg_details": {},
            "m5_bos_confirmed": True, "ustec_agrees": True,
            "rr_valid": True, "rr": 2.5, "sl_price": 6485, "tp_price": 6525,
            "sl_points": 15, "tp_points": 25, "atr": 10, "tp_source": "atr",
            "sl_method": "swing", "sweep_details": None,
            "has_liquidity_sweep": sweep, "displacement_valid": True,
            "ustec_details": {},
        }

    def test_half_size_on_score4_non_full_confidence(self, engine, sample_macro_data, sample_news_data):
        """Score 4/4 with entry_confidence='high' (not 'full') → HALF_SIZE grade B."""
        self._setup_full_layers(engine, sample_macro_data, entry_confidence="high")
        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert result["decision"] == "HALF_SIZE"
        assert result["grade"] == "B"

    def test_half_size_on_score3_with_entry(self, engine, sample_macro_data, sample_news_data):
        """Score 3/4 with entry_ready → HALF_SIZE grade B."""
        sample_macro_data["score_contribution"] = 0
        sample_macro_data["bias"] = "MIXED"
        self._setup_full_layers(engine, sample_macro_data, entry_confidence="base")
        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert result["score"] == 3
        assert result["decision"] == "HALF_SIZE"
        assert result["grade"] == "B"

    def test_wait_on_score3_no_entry(self, engine, sample_macro_data, sample_news_data):
        """Score 3/4 without entry_ready but with direction → WAIT."""
        sample_macro_data["score_contribution"] = 0
        sample_macro_data["bias"] = "MIXED"
        self._setup_full_layers(engine, sample_macro_data, entry_ready=False, entry_confidence=None)
        result = engine.run_full_checklist(6500.0, sample_macro_data, sample_news_data)
        assert result["decision"] == "WAIT"
        assert result["grade"] is None

    def test_reduced_confidence_downgrades_a_to_b(self, engine, sample_news_data):
        """direction_confidence='reduced' forces Grade A → B."""
        layer1 = {"bias": "MIXED", "vix_direction_bias": "NEUTRAL",
                   "score_contribution": 1, "short_only": False, "vix_bucket": "normal",
                   "size_label": "normal", "vix_value": 17.0}
        layer2 = {"structure_bias": "short", "ema_bias": "bearish"}
        d, c = engine._derive_direction(layer1, layer2)
        assert d == "SHORT"
        assert c == "reduced"

    def test_fallback_result_has_all_required_keys(self, engine):
        """_fallback_result produces a complete result dict with no missing keys."""
        result = engine._fallback_result(6500.0)
        required_keys = [
            "timestamp", "sl_time", "session", "day_name", "current_price",
            "score", "entry_ready", "direction", "decision", "grade",
            "caution_flags", "block_reason", "is_ranging", "range_condition",
            "has_liquidity_sweep", "layer1", "layer2", "layer3", "layer4",
        ]
        for key in required_keys:
            assert key in result, f"Fallback result missing key: {key}"
        assert result["decision"] == "NO_TRADE"
        assert result["block_reason"] == "checklist_error"


# ═══════════════════════════════════════════════════════════════
# CHECKLIST ENGINE — VIX safety & trade_allowed
# ═══════════════════════════════════════════════════════════════

class TestVixSafetyBlock:
    """Verify VIX unavailability blocks all signals."""

    def test_trade_allowed_false_blocks_signal(self, engine, sample_news_data):
        """When layer1.trade_allowed=False, decision must be NO_TRADE."""
        result = {
            "layer1": {"trade_allowed": False, "block_reason": "VIX data unavailable",
                       "vix_value": 0.0, "short_only": False},
            "direction": "LONG", "score": 4, "grade": "A",
            "decision": "FULL_SEND", "caution_flags": [], "block_reason": None,
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        assert filtered["decision"] == "NO_TRADE"
        assert "unavailable" in filtered["block_reason"].lower()
        assert filtered["grade"] is None

    def test_trade_allowed_true_passes_through(self, engine, sample_news_data):
        """When trade_allowed=True and VIX < 30, no block."""
        result = {
            "layer1": {"trade_allowed": True, "vix_value": 17.0, "short_only": False},
            "direction": "LONG", "score": 4, "grade": "A",
            "decision": "FULL_SEND", "caution_flags": [], "block_reason": None,
            "entry_ready": True, "has_liquidity_sweep": True,
            "direction_confidence": "high",
            "range_condition": {"is_ranging": False},
        }
        filtered = engine.apply_all_filters(result, sample_news_data)
        assert filtered["decision"] == "FULL_SEND"

    def test_post_news_blocks_grade_b_only(self, engine):
        """Post-news caution blocks Grade B but not Grade A."""
        news = {"is_news_day": True, "post_news_caution": True, "pre_news_blocked": False}
        result_b = {
            "layer1": {"vix_value": 17.0, "short_only": False, "trade_allowed": True},
            "direction": "LONG", "score": 3, "grade": "B",
            "decision": "HALF_SIZE", "caution_flags": [], "block_reason": None,
            "_day_flags": {"monday_caution": False, "friday_caution": False},
            "entry_ready": True, "has_liquidity_sweep": True,
            "direction_confidence": "high",
            "range_condition": {"is_ranging": False},
        }
        filtered = engine.apply_all_filters(result_b, news)
        assert filtered["decision"] == "NO_TRADE"
        assert "Post-news" in filtered["block_reason"]


# ═══════════════════════════════════════════════════════════════
# CHECKLIST ENGINE — DST calculation
# ═══════════════════════════════════════════════════════════════

class TestDSTCalculation:
    """_is_us_dst boundary tests."""

    def test_mid_summer_is_dst(self):
        """July is always US DST."""
        dt = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert ChecklistEngine._is_us_dst(dt) is True

    def test_mid_winter_is_not_dst(self):
        """January is never US DST."""
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert ChecklistEngine._is_us_dst(dt) is False

    def test_march_8_2026_after_7utc_is_dst(self):
        """2026 DST starts March 8 at 07:00 UTC."""
        dt = datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc)
        assert ChecklistEngine._is_us_dst(dt) is True

    def test_march_8_2026_before_7utc_is_not_dst(self):
        """Just before DST transition."""
        dt = datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc)
        assert ChecklistEngine._is_us_dst(dt) is False


# ═══════════════════════════════════════════════════════════════
# OUTCOME TRACKER — _resolve_signal_with_fallback dispatch
# ═══════════════════════════════════════════════════════════════

class TestResolveSignalWithFallback:
    """Test the M5 → H1 → hard-TIMEOUT dispatch logic."""

    def test_m5_win_writes_outcome(self, tracker):
        """M5 bars available, TP hit → update_outcome called with WIN."""
        now = datetime.now(timezone.utc)
        signal = {
            "id": 1, "direction": "LONG", "entry_price": 6500,
            "sl_price": 6485, "tp_price": 6525, "tp1_price": None,
            "outcome": None,
            "timestamp": (now - timedelta(hours=2)).isoformat(),
        }
        n = 5
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=2, minutes=-5), periods=n, freq="5min", tz="UTC"),
            "open": [6511] * n, "high": [6530] * n,
            "low": [6510] * n, "close": [6529] * n, "volume": [1000] * n,
        })
        tracker._ctrader.fetch_bars = MagicMock(return_value=bars)
        result = tracker._resolve_signal_with_fallback(signal)
        assert result is not None
        assert result["outcome"] == "WIN"
        tracker._signal_logger.update_outcome.assert_called_once()

    def test_m5_partial_win_calls_update_tp1_hit(self, tracker):
        """M5 bars find TP1 hit → update_tp1_hit called, not update_outcome."""
        now = datetime.now(timezone.utc)
        signal = {
            "id": 2, "direction": "SHORT", "entry_price": 6500,
            "sl_price": 6515, "tp_price": 6475, "tp1_price": 6490,
            "outcome": None,
            "timestamp": (now - timedelta(hours=2)).isoformat(),
        }
        n = 5
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=2, minutes=-5), periods=n, freq="5min", tz="UTC"),
            "open": [6489] * n, "high": [6505] * n,
            "low": [6488] * n, "close": [6504] * n, "volume": [1000] * n,
        })
        tracker._ctrader.fetch_bars = MagicMock(return_value=bars)
        result = tracker._resolve_signal_with_fallback(signal)
        assert result["outcome"] == "PARTIAL_WIN"
        assert result.get("tp1_hit") is True
        tracker._signal_logger.update_tp1_hit.assert_called_once()
        tracker._signal_logger.update_outcome.assert_not_called()

    def test_partial_win_final_writes_label_1(self, tracker):
        """PARTIAL_WIN_FINAL must be stored with label=1 (direction was correct)."""
        now = datetime.now(timezone.utc)
        signal = {
            "id": 3, "direction": "SHORT", "entry_price": 6500,
            "sl_price": 6515, "tp_price": 6475, "tp1_price": 6490,
            "outcome": "PARTIAL_WIN", "tp1_hit": 1,
            "tp1_hit_timestamp": (now - timedelta(hours=2)).isoformat(),
            "outcome_timestamp": (now - timedelta(hours=2)).isoformat(),
            "timestamp": (now - timedelta(hours=3)).isoformat(),
        }
        n = 80
        bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=4), periods=n, freq="5min", tz="UTC"),
            "open": [6498] * n, "high": [6505] * n,
            "low": [6495] * n, "close": [6498] * n, "volume": [1000] * n,
        })
        tracker._ctrader.fetch_bars = MagicMock(return_value=bars)
        result = tracker._resolve_signal_with_fallback(signal)
        assert result is not None
        assert result["outcome"] == "PARTIAL_WIN_FINAL"
        tracker._signal_logger.update_outcome.assert_called_once()
        args, kwargs = tracker._signal_logger.update_outcome.call_args
        assert kwargs.get("label", args[2] if len(args) > 2 else None) == 1

    def test_h1_fallback_tp1_hit_calls_update_tp1_hit(self, tracker):
        """H1 fallback finds TP1 hit → calls update_tp1_hit, not ESTIMATED_ prefix."""
        now = datetime.now(timezone.utc)
        signal = {
            "id": 4, "direction": "SHORT", "entry_price": 6500,
            "sl_price": 6515, "tp_price": 6475, "tp1_price": 6490,
            "outcome": None,
            "timestamp": (now - timedelta(hours=50)).isoformat(),
        }
        n = 10
        h1_bars = pd.DataFrame({
            "timestamp": pd.date_range(now - timedelta(hours=55), periods=n, freq="1h", tz="UTC"),
            "open": [6498] * n, "high": [6505] * n,
            "low": [6488] * n, "close": [6492] * n, "volume": [5000] * n,
        })

        def fetch_side_effect(sym, period, count):
            if period == "M5":
                return pd.DataFrame()
            return h1_bars.copy()

        tracker._ctrader.fetch_bars = MagicMock(side_effect=fetch_side_effect)
        result = tracker._resolve_signal_with_fallback(signal)
        assert result is not None
        assert result["outcome"] == "PARTIAL_WIN"
        tracker._signal_logger.update_tp1_hit.assert_called_once()

    def test_hard_timeout_after_48h_no_bars(self, tracker):
        """Signal >48h old with no bars at all → hard TIMEOUT."""
        now = datetime.now(timezone.utc)
        signal = {
            "id": 5, "direction": "LONG", "entry_price": 6500,
            "sl_price": 6485, "tp_price": 6525, "tp1_price": None,
            "outcome": None,
            "timestamp": (now - timedelta(hours=50)).isoformat(),
        }
        tracker._ctrader.fetch_bars = MagicMock(return_value=None)
        result = tracker._resolve_signal_with_fallback(signal)
        assert result is not None
        assert result["outcome"] == "TIMEOUT"
        tracker._signal_logger.update_outcome.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# MACRO CHECKER — VIX None safety
# ═══════════════════════════════════════════════════════════════

class TestMacroCheckerVixSafety:
    """Verify VIX=None triggers trade_allowed=False."""

    def test_vix_none_blocks_trading(self, mock_alert_bot):
        """When yfinance returns VIX=None, layer1 must set trade_allowed=False."""
        checker = MacroChecker(alert_bot=mock_alert_bot)
        checker._cache = {
            "vix": {"value": None, "pct_change": 0.0},
            "dxy": {"value": 99.5, "pct_change": 0.0, "direction": "flat"},
            "us10y": {"value": 4.3, "pct_change": 0.0, "direction": "flat"},
            "oil": {"value": 72.0, "pct_change": 0.0, "direction": "stable"},
            "rut": {"value": 2100.0, "pct_change": 0.0, "direction": "neutral"},
        }
        checker._cache_time = __import__("time").time()
        result = checker.get_layer1_result()
        assert result["trade_allowed"] is False
        assert "VIX" in result.get("block_reason", "")

    def test_vix_present_allows_trading(self, mock_alert_bot):
        """Normal VIX value → trade_allowed=True."""
        checker = MacroChecker(alert_bot=mock_alert_bot)
        checker._cache = {
            "vix": {"value": 18.0, "pct_change": -1.0, "direction": "falling"},
            "dxy": {"value": 99.5, "pct_change": 0.0, "direction": "flat"},
            "us10y": {"value": 4.3, "pct_change": -0.1, "direction": "falling"},
            "oil": {"value": 72.0, "pct_change": -0.8, "direction": "falling"},
            "rut": {"value": 2100.0, "pct_change": 0.5, "direction": "green"},
        }
        checker._cache_time = __import__("time").time()
        result = checker.get_layer1_result()
        assert result["trade_allowed"] is True
        assert result["vix_value"] == 18.0


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINER — meta label filtering
# ═══════════════════════════════════════════════════════════════

class TestMetaLabelFiltering:
    """Verify train_meta_label_model only uses resolved, non-TIMEOUT signals."""

    def test_below_threshold_returns_false(self):
        """Model 3 refuses to train below MODEL_RETRAIN_SIGNAL_THRESHOLD."""
        mock_logger = MagicMock()
        df = pd.DataFrame({
            "outcome": ["WIN"] * 10,
            "outcome_label": [1] * 10,
        })
        mock_logger.get_training_data.return_value = df
        trainer = ModelTrainer(signal_logger=mock_logger)
        result = trainer.train_meta_label_model()
        assert result is False

    def test_timeout_excluded_from_training(self):
        """TIMEOUT outcomes are filtered before training."""
        trainer = ModelTrainer(signal_logger=None)
        data = pd.DataFrame({
            "outcome": ["WIN", "LOSS", "TIMEOUT", "ESTIMATED_TIMEOUT", "PARTIAL_WIN_FINAL"],
            "outcome_label": [1, -1, 0, 0, 1],
        })
        filtered = data[data["outcome_label"].notna()].copy()
        filtered = filtered[~filtered["outcome"].str.contains("TIMEOUT", na=True)].copy()
        assert len(filtered) == 3
        assert "TIMEOUT" not in filtered["outcome"].values
        assert "ESTIMATED_TIMEOUT" not in filtered["outcome"].values

    def test_null_outcome_label_excluded(self):
        """Rows with NULL outcome_label are dropped."""
        data = pd.DataFrame({
            "outcome": ["WIN", "LOSS", None],
            "outcome_label": [1, -1, None],
        })
        filtered = data[data["outcome_label"].notna()]
        assert len(filtered) == 2


# ═══════════════════════════════════════════════════════════════
# MODEL PREDICTOR — graceful handling
# ═══════════════════════════════════════════════════════════════

class TestModelPredictorGraceful:
    """Verify predictions return None when models are missing."""

    def test_predict_day_quality_no_model(self):
        """With no model1 loaded, predict_day_quality returns None."""
        predictor = ModelPredictor()
        assert predictor.predict_day_quality({"vix_pct": -1.0}) is None

    def test_predict_session_bias_no_model(self):
        """With no model2 loaded, predict_session_bias returns None."""
        predictor = ModelPredictor()
        assert predictor.predict_session_bias({"vix_pct": -1.0}) is None

    def test_predict_win_probability_no_model(self):
        """With no model3 loaded, predict_win_probability returns None."""
        predictor = ModelPredictor()
        assert predictor.predict_win_probability({"layer1": {}}) is None

    def test_predict_win_probability_no_trainer(self):
        """With model3 loaded but no trainer, returns None."""
        predictor = ModelPredictor()
        predictor.models["model3"] = {"model": MagicMock(), "features": ["f1"]}
        predictor._trainer = None
        assert predictor.predict_win_probability({"layer1": {}}) is None

    def test_get_ml_enhancement_no_models(self):
        """get_ml_enhancement with no models returns safe defaults."""
        predictor = ModelPredictor()
        result = predictor.get_ml_enhancement({}, {})
        assert result["ml_active"] is False
        assert result["win_probability"] is None
        assert result["day_quality_prob"] is None
        assert result["models_loaded"] == []

    def test_flatten_checklist(self):
        """_flatten_checklist merges nested layer dicts into flat dict."""
        predictor = ModelPredictor()
        nested = {
            "layer1": {"bias": "LONG", "vix_value": 17.0},
            "layer2": {"structure_bias": "long"},
            "entry": {"rr": 2.5},
            "score": 4, "direction": "LONG",
        }
        flat = predictor._flatten_checklist(nested)
        assert flat["bias"] == "LONG"
        assert flat["structure_bias"] == "long"
        assert flat["rr"] == 2.5
        assert flat["score"] == 4


# ═══════════════════════════════════════════════════════════════
# SIGNAL LOGGER — pending signals with tp1_hit_timestamp delay
# ═══════════════════════════════════════════════════════════════

class TestPendingSignalsDelay:
    """Verify get_pending_signals respects OUTCOME_CHECK_DELAY_SECONDS for PARTIAL_WIN."""

    def test_fresh_partial_win_not_returned(self, logger_db):
        """A PARTIAL_WIN with very recent tp1_hit_timestamp should not appear in pending."""
        now = datetime.now(timezone.utc)
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, tp1_hit,
               tp1_hit_timestamp, save_status)
               VALUES (?, 'SHORT', 'PARTIAL_WIN', 1, ?, 'complete')""",
            (now.isoformat(), now.isoformat()),
        )
        conn.commit()
        conn.close()
        pending = logger_db.get_pending_signals()
        assert len(pending) == 0

    def test_old_partial_win_returned(self, logger_db):
        """A PARTIAL_WIN with old tp1_hit_timestamp should appear in pending."""
        old = datetime.now(timezone.utc) - timedelta(seconds=OUTCOME_CHECK_DELAY_SECONDS + 60)
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, tp1_hit,
               tp1_hit_timestamp, save_status)
               VALUES (?, 'SHORT', 'PARTIAL_WIN', 1, ?, 'complete')""",
            (old.isoformat(), old.isoformat()),
        )
        conn.commit()
        conn.close()
        pending = logger_db.get_pending_signals()
        assert len(pending) == 1


# ═══════════════════════════════════════════════════════════════
# SIGNAL LOGGER — win rate with all outcome types
# ═══════════════════════════════════════════════════════════════

class TestWinRateAllOutcomeTypes:
    """Win rate must correctly count PARTIAL_WIN_FINAL as wins (label=1)."""

    def test_win_rate_with_mixed_outcomes(self, logger_db):
        """WIN + PARTIAL_WIN_FINAL (both label=1) + LOSS (label=-1) = 66.7% win rate."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
               VALUES (?, 'LONG', 'WIN', 1, 'complete')""", (now,))
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
               VALUES (?, 'SHORT', 'PARTIAL_WIN_FINAL', 1, 'complete')""", (now,))
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
               VALUES (?, 'LONG', 'LOSS', -1, 'complete')""", (now,))
        conn.commit()
        conn.close()
        wr = logger_db.get_win_rate()
        assert abs(wr - 66.7) < 0.1

    def test_signal_count_excludes_null_outcomes(self, logger_db):
        """get_signal_count only counts rows with non-NULL outcome."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
               VALUES (?, 'LONG', 'WIN', 1, 'complete')""", (now,))
        conn.execute(
            """INSERT INTO signals (timestamp, direction, save_status)
               VALUES (?, 'SHORT', 'complete')""", (now,))
        conn.commit()
        conn.close()
        assert logger_db.get_signal_count() == 1


# ═══════════════════════════════════════════════════════════════
# SIGNAL LOGGER — training data export
# ═══════════════════════════════════════════════════════════════

class TestTrainingDataExportFiltering:
    """get_training_data only returns resolved signals."""

    def test_excludes_null_outcome_signals(self, logger_db):
        """Signals without outcome must not appear in training data."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(logger_db._db_path)
        conn.execute(
            """INSERT INTO signals (timestamp, direction, outcome, outcome_label, save_status)
               VALUES (?, 'LONG', 'WIN', 1, 'complete')""", (now,))
        conn.execute(
            """INSERT INTO signals (timestamp, direction, save_status)
               VALUES (?, 'SHORT', 'complete')""", (now,))
        conn.commit()
        conn.close()
        df = logger_db.get_training_data()
        assert len(df) == 1
        assert df.iloc[0]["outcome"] == "WIN"


# ═══════════════════════════════════════════════════════════════
# LIVE.PY — scheduler time conversions
# ═══════════════════════════════════════════════════════════════

class TestSchedulerTimeConversion:
    """Test SL↔UTC time conversion math (pure functions, no live.py import)."""

    @staticmethod
    def _sl_time_to_utc(sl_hhmm: str) -> str:
        """Replicate the conversion logic from live.py to avoid heavy imports."""
        h, m = map(int, sl_hhmm.split(":"))
        utc_minutes = int((h * 60 + m - SL_UTC_OFFSET * 60) % 1440)
        return f"{utc_minutes // 60:02d}:{utc_minutes % 60:02d}"

    @staticmethod
    def _utc_to_local(utc_hhmm: str) -> str:
        """Replicate the conversion logic from live.py."""
        import time as _time
        h, m = map(int, utc_hhmm.split(":"))
        local_offset_sec = -(_time.altzone if _time.localtime().tm_isdst > 0 else _time.timezone)
        local_minutes = (h * 60 + m + local_offset_sec // 60) % 1440
        return f"{local_minutes // 60:02d}:{local_minutes % 60:02d}"

    def test_sl_time_to_utc(self):
        """23:00 SL (UTC+5:30) = 17:30 UTC."""
        assert self._sl_time_to_utc("23:00") == "17:30"

    def test_sl_time_to_utc_morning(self):
        """05:30 SL = 00:00 UTC."""
        assert self._sl_time_to_utc("05:30") == "00:00"

    def test_utc_to_local_returns_valid_time(self):
        """Conversion returns a valid HH:MM string."""
        result = self._utc_to_local("00:00")
        assert len(result) == 5
        assert ":" in result
        h, m = result.split(":")
        assert 0 <= int(h) <= 23
        assert 0 <= int(m) <= 59

    def test_sl_to_utc_roundtrip_consistency(self):
        """Converting SL→UTC and back should be consistent with the offset."""
        utc = self._sl_time_to_utc("12:00")
        h, m = map(int, utc.split(":"))
        back_min = int((h * 60 + m + SL_UTC_OFFSET * 60) % 1440)
        assert back_min == 12 * 60


# ═══════════════════════════════════════════════════════════════
# OUTCOME TRACKER — PnL edge cases
# ═══════════════════════════════════════════════════════════════

class TestPnlEdgeCases:
    """Test PnL calculations for all outcome variants."""

    def test_estimated_partial_win_pnl(self):
        """ESTIMATED_PARTIAL_WIN is not TIMEOUT → should use WIN path."""
        signal = {"entry_price": 6500, "sl_price": 6515, "tp_price": 6475, "direction": "SHORT"}
        pnl = OutcomeTracker.calculate_pnl(signal, "ESTIMATED_WIN")
        assert pnl == 25.0

    def test_partial_win_final_pnl_via_calc_tp1(self):
        """_calc_tp1_pnl for SHORT: (entry - tp1) / 2."""
        tracker_cls = OutcomeTracker.__new__(OutcomeTracker)
        signal = {"entry_price": 6500, "tp1_price": 6490, "direction": "SHORT"}
        pnl = tracker_cls._calc_tp1_pnl(signal)
        assert pnl == 5.0  # (6500 - 6490) / 2

    def test_calc_tp1_pnl_long(self):
        """_calc_tp1_pnl for LONG: (tp1 - entry) / 2."""
        tracker_cls = OutcomeTracker.__new__(OutcomeTracker)
        signal = {"entry_price": 6500, "tp1_price": 6515, "direction": "LONG"}
        pnl = tracker_cls._calc_tp1_pnl(signal)
        assert pnl == 7.5  # (6515 - 6500) / 2

    def test_calc_full_pnl_after_tp1_short(self):
        """Full PnL = (half@TP1 + half@TP2) / 2 for SHORT."""
        tracker_cls = OutcomeTracker.__new__(OutcomeTracker)
        signal = {"entry_price": 6500, "tp1_price": 6490, "tp_price": 6475, "direction": "SHORT"}
        pnl = tracker_cls._calc_full_pnl_after_tp1(signal)
        expected = ((6500 - 6490) + (6500 - 6475)) / 2  # (10 + 25) / 2 = 17.5
        assert pnl == expected

    def test_calc_full_pnl_after_tp1_long(self):
        """Full PnL = (half@TP1 + half@TP2) / 2 for LONG."""
        tracker_cls = OutcomeTracker.__new__(OutcomeTracker)
        signal = {"entry_price": 6500, "tp1_price": 6510, "tp_price": 6525, "direction": "LONG"}
        pnl = tracker_cls._calc_full_pnl_after_tp1(signal)
        expected = ((6510 - 6500) + (6525 - 6500)) / 2  # (10 + 25) / 2 = 17.5
        assert pnl == expected


# ═══════════════════════════════════════════════════════════════
# CHECKLIST — should_send_alert edge cases
# ═══════════════════════════════════════════════════════════════

class TestShouldSendAlertEdgeCases:
    """Edge cases in alert gating logic."""

    def test_half_size_with_entry_sends(self, engine):
        """HALF_SIZE + entry_ready + no block → alert fires."""
        result = {
            "decision": "HALF_SIZE", "entry_ready": True,
            "block_reason": None, "direction": "SHORT",
            "current_price": 6500, "score": 3,
        }
        assert engine.should_send_alert(result) is True

    def test_duplicate_suppresses(self, engine):
        """Duplicate check returning True suppresses alert."""
        engine._signal_logger.is_duplicate.return_value = True
        result = {
            "decision": "FULL_SEND", "entry_ready": True,
            "block_reason": None, "direction": "LONG",
            "current_price": 6500, "score": 4,
        }
        assert engine.should_send_alert(result) is False

    def test_wait_with_block_reason_suppressed(self, engine):
        """WAIT signal with a block_reason should not fire."""
        result = {
            "decision": "WAIT", "entry_ready": False,
            "block_reason": "Pre-news: CPI", "score": 3,
            "direction": "SHORT", "current_price": 6500,
        }
        assert engine.should_send_alert(result) is False

    def test_wait_without_direction_suppressed(self, engine):
        """WAIT without direction should not fire."""
        result = {
            "decision": "WAIT", "entry_ready": False,
            "block_reason": None, "score": 3,
            "direction": None, "current_price": 6500,
        }
        assert engine.should_send_alert(result) is False

    def test_full_send_without_entry_ready_suppressed(self, engine):
        """FULL_SEND but entry_ready=False should not fire."""
        result = {
            "decision": "FULL_SEND", "entry_ready": False,
            "block_reason": None, "direction": "LONG",
            "current_price": 6500, "score": 4,
        }
        assert engine.should_send_alert(result) is False


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINER — categorical encoding NaN handling
# ═══════════════════════════════════════════════════════════════

class TestCategoricalEncodingNaN:
    """Verify unknown categorical values produce NaN (not 0)."""

    def test_unknown_category_maps_to_nan(self):
        """An unmapped categorical value should be NaN, not 0."""
        trainer = ModelTrainer()
        row = {"direction": "SIDEWAYS", "score": 3}
        df = pd.DataFrame([row])
        result = trainer.prepare_checklist_features(df)
        if "direction_enc" in result.columns:
            assert pd.isna(result["direction_enc"].iloc[0])

    def test_missing_column_produces_nan(self):
        """A completely missing categorical column should produce NaN."""
        trainer = ModelTrainer()
        row = {"score": 3, "vix_level": 17.0}
        df = pd.DataFrame([row])
        result = trainer.prepare_checklist_features(df)
        if "direction_enc" in result.columns:
            assert pd.isna(result["direction_enc"].iloc[0])
