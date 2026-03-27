"""Tests that every layer returns its safe fallback when an exception occurs."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd


# ══════════════════════════════════════════════════════════════
# LAYER 2 — Structure Analyzer
# ══════════════════════════════════════════════════════════════

class TestStructureAnalyzerFallback:
    def test_get_layer2_fallback_on_exception(self, mock_ctrader, mock_alert_bot):
        from src.structure_analyzer import StructureAnalyzer

        mock_ctrader.fetch_bars.side_effect = RuntimeError("connection lost")
        sa = StructureAnalyzer(mock_ctrader, 12345, alert_bot=mock_alert_bot)
        result = sa.get_layer2_result(6500.0)

        assert result["structure_bias"] == "unclear"
        assert result["score_contribution"] == 0
        assert result["ema_bias"] == "unclear"
        assert result["above_ema50"] is None
        assert result["ema50_value"] is None
        assert result["bos_direction"] is None
        assert result["bos_level"] is None
        assert result["bos_bars_ago"] == 0
        assert result["choch_direction"] is None
        assert result["choch_recent"] is False
        assert result["wyckoff"] == "unclear"
        assert result["ob_bullish_nearby"] is False
        assert result["ob_bearish_nearby"] is False
        assert result["bullish_obs"] == []
        assert result["bearish_obs"] == []
        assert result["range_condition"]["is_ranging"] is False


# ══════════════════════════════════════════════════════════════
# LAYER 3 — Zone Calculator
# ══════════════════════════════════════════════════════════════

class TestZoneCalculatorFallback:
    def test_get_layer3_fallback_on_exception(self, mock_ctrader, mock_alert_bot):
        from src.zone_calculator import ZoneCalculator

        zc = ZoneCalculator(mock_ctrader, 12345, alert_bot=mock_alert_bot)
        with patch.object(zc, "get_all_levels", side_effect=RuntimeError("level calc exploded")):
            result = zc.get_layer3_result(6500.0)

        assert result["at_zone"] is False
        assert result["zone_level"] is None
        assert result["zone_type"] is None
        assert result["distance"] == 999.0
        assert result["zone_direction"] is None
        assert result["score_contribution"] == 0
        assert result["all_levels"] == []
        assert result["all_nearby"] == []
        assert result["asian_range"] == {"asian_high": None, "asian_low": None, "valid": False}
        assert result["liquidity_sweeps"] == []
        assert result["has_buy_side_sweep"] is False
        assert result["has_sell_side_sweep"] is False


# ══════════════════════════════════════════════════════════════
# LAYER 4 — Order Flow Analyzer
# ══════════════════════════════════════════════════════════════

class TestOrderFlowFallback:
    def test_get_layer4_fallback_on_exception(self, mock_ctrader, mock_alert_bot):
        from src.orderflow_analyzer import OrderFlowAnalyzer

        mock_ctrader.fetch_bars.side_effect = RuntimeError("connection lost")
        ofa = OrderFlowAnalyzer(mock_ctrader, 12345, alert_bot=mock_alert_bot)
        result = ofa.get_layer4_result("SHORT", {})

        assert result["delta_direction"] == "mixed"
        assert result["divergence"] == "none"
        assert result["vix_spiking_now"] is False
        assert result["confirms_bias"] is False
        assert result["score_contribution"] == 0
        assert result["intrabar_active"] is False


# ══════════════════════════════════════════════════════════════
# LAYER 5 — Entry Checker
# ══════════════════════════════════════════════════════════════

class TestEntryCheckerFallback:
    def test_get_entry_fallback_on_exception(self, mock_ctrader, mock_alert_bot):
        from src.entry_checker import EntryChecker

        mock_ctrader.fetch_bars.side_effect = RuntimeError("connection lost")
        ec = EntryChecker(mock_ctrader, 12345, 67890, alert_bot=mock_alert_bot)
        result = ec.get_entry_result("SHORT", 6500.0)

        assert result["fvg_present"] is False
        assert result["entry_ready"] is False
        assert result["displacement_valid"] is False
        assert result["m5_bos_confirmed"] is False
        assert result["ustec_agrees"] is False
        assert result["rr_valid"] is False
        assert result["entry_confidence"] is None
        assert result["has_liquidity_sweep"] is False
        assert result["sweep_details"] is None
        assert result["sl_price"] == 0
        assert result["tp_price"] == 0
        assert result["tp1_price"] == 0
        assert result["rr"] == 0
        assert result["tp_source"] is None
        assert result["sl_method"] is None
        fvg = result["fvg_details"]
        assert fvg["present"] is False
        assert fvg["direction"] is None


# ══════════════════════════════════════════════════════════════
# LAYER 1 — Macro Checker
# ══════════════════════════════════════════════════════════════

class TestMacroCheckerFallback:
    def test_get_layer1_fallback_on_exception(self, mock_alert_bot):
        from src.macro_checker import MacroChecker

        mc = MacroChecker(alert_bot=mock_alert_bot)
        mc.fetch_macro_data = MagicMock(side_effect=RuntimeError("yfinance down"))
        result = mc.get_layer1_result()

        assert result["bias"] == "MIXED"
        assert result["vix_bucket"] == "normal"
        assert result["vix_value"] == 0.0
        assert result["vix_pct"] == 0.0
        assert result["size_label"] == "normal"
        assert result["trade_allowed"] is True
        assert result["short_only"] is False
        assert result["vix_direction_bias"] == "NEUTRAL"
        assert result["us10y_direction"] == "flat"
        assert result["oil_direction"] == "stable"
        assert result["dxy_direction"] == "flat"
        assert result["rut_direction"] == "neutral"
        assert result["groq_sentiment"] == "NEUTRAL"
        assert result["bullish_count"] == 0
        assert result["bearish_count"] == 0
        assert result["score_contribution"] == 0
        assert "raw_data" in result
        assert "vix" in result["raw_data"]


# ══════════════════════════════════════════════════════════════
# CHECKLIST ENGINE
# ══════════════════════════════════════════════════════════════

class TestChecklistEngineFallback:
    def test_run_full_checklist_fallback_on_exception(self, mock_alert_bot):
        from src.checklist_engine import ChecklistEngine

        mock_macro = MagicMock()
        mock_macro.get_layer1_result.side_effect = RuntimeError("macro boom")

        engine = ChecklistEngine(
            macro_checker=mock_macro,
            structure_analyzer=MagicMock(),
            zone_calculator=MagicMock(),
            orderflow_analyzer=MagicMock(),
            entry_checker=MagicMock(),
            signal_logger=MagicMock(),
            alert_bot=mock_alert_bot,
        )

        result = engine.run_full_checklist(6500.0, {}, {})

        assert result["score"] == 0
        assert result["decision"] == "NO_TRADE"
        assert result["entry_ready"] is False
        assert result["direction"] is None
        assert result["block_reason"] == "checklist_error"
        assert result["size_label"] == "normal"
        assert result["vix_bucket"] == "normal"
        assert result["grade"] is None
        assert result["caution_flags"] == []
        assert result["current_price"] == 6500.0
        assert result["is_news_day"] is False
        assert result["is_ranging"] is False
        assert result["has_liquidity_sweep"] is False
        assert "timestamp" in result
        assert "session" in result


# ══════════════════════════════════════════════════════════════
# MODEL PREDICTOR
# ══════════════════════════════════════════════════════════════

class TestModelPredictorFallback:
    def test_get_ml_enhancement_fallback_on_exception(self):
        from src.model_predictor import ModelPredictor

        mp = ModelPredictor(signal_logger=MagicMock(), model_trainer=MagicMock())
        with patch.object(mp, "_extract_macro_features", side_effect=RuntimeError("kaboom")):
            result = mp.get_ml_enhancement({}, {})

        assert result["ml_active"] is False
        assert result["day_quality_prob"] is None
        assert result["session_bias_direction"] is None
        assert result["session_bias_confidence"] is None
        assert result["win_probability"] is None
        assert result["shap_top_features"] is None
        assert result["models_loaded"] == []


# ══════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════

class TestModelTrainerFallback:
    def test_train_meta_label_returns_false_on_exception(self):
        from src.model_trainer import ModelTrainer

        mock_logger = MagicMock()
        mock_logger.get_training_data.side_effect = RuntimeError("db corrupted")
        mt = ModelTrainer(signal_logger=mock_logger, alert_bot=MagicMock())
        result = mt.train_meta_label_model()

        assert result is False


# ══════════════════════════════════════════════════════════════
# OUTCOME TRACKER
# ══════════════════════════════════════════════════════════════

class TestOutcomeTrackerFallback:
    def test_track_pending_no_crash_on_exception(self, mock_ctrader, mock_alert_bot):
        from src.outcome_tracker import OutcomeTracker

        mock_sig_logger = MagicMock()
        mock_sig_logger.get_pending_signals.side_effect = RuntimeError("db locked")
        ot = OutcomeTracker(mock_ctrader, 12345, mock_sig_logger, alert_bot=mock_alert_bot)

        ot.track_pending_outcomes()

        mock_alert_bot.send_system_alert.assert_called()


# ══════════════════════════════════════════════════════════════
# PATTERN SCANNER
# ══════════════════════════════════════════════════════════════

class TestPatternScannerFallback:
    def test_run_pattern_scan_no_crash_on_exception(self, mock_ctrader, mock_alert_bot):
        from src.pattern_scanner import PatternScanner

        mock_checklist = MagicMock()
        mock_checklist.is_trading_session.return_value = True

        ps = PatternScanner(
            macro_checker=MagicMock(),
            zone_calculator=MagicMock(),
            orderflow_analyzer=MagicMock(),
            checklist_engine=mock_checklist,
            ctrader_connection=mock_ctrader,
            us500_id=12345,
            signal_logger=MagicMock(),
            model_predictor=MagicMock(),
            alert_bot=mock_alert_bot,
        )

        with patch.object(ps, "is_active", return_value=True):
            with patch.object(ps, "get_current_market_snapshot", side_effect=RuntimeError("boom")):
                ps.run_pattern_scan()


# ══════════════════════════════════════════════════════════════
# WEEKLY REPORT — error isolation
# ══════════════════════════════════════════════════════════════

def _import_live():
    """Import live.py without letting it wrap sys.stdout/sys.stderr on Windows."""
    import sys
    _orig_platform = sys.platform
    sys.platform = "testing"
    try:
        import live
    finally:
        sys.platform = _orig_platform
    return live


class TestWeeklyReportErrorIsolation:
    def test_retrain_failure_does_not_block_report(self):
        live = _import_live()

        mock_trainer = MagicMock()
        mock_trainer.check_and_retrain.side_effect = RuntimeError("retrain exploded")
        mock_predictor = MagicMock()
        mock_sig_logger = MagicMock()
        mock_bot = MagicMock()
        mock_health = MagicMock()
        mock_evo = MagicMock()

        with patch.multiple(
            "live",
            model_trainer=mock_trainer,
            model_predictor=mock_predictor,
            signal_logger=mock_sig_logger,
            alert_bot=mock_bot,
            health_monitor=mock_health,
            evolution_manager=mock_evo,
        ):
            live.weekly_report_and_retrain()

        mock_bot.send_system_alert.assert_any_call(
            "WARNING", "weekly_retrain", str(RuntimeError("retrain exploded")),
        )
        mock_bot.send_weekly_report.assert_called_once()


# ══════════════════════════════════════════════════════════════
# DAILY RETRAIN — skips on retrain day
# ══════════════════════════════════════════════════════════════

class TestDailyRetrainSkipsSunday:
    def test_skips_on_retrain_day(self):
        live = _import_live()
        from config import MODEL_RETRAIN_DAY

        mock_trainer = MagicMock()
        mock_bot = MagicMock()
        mock_evo = MagicMock()

        fake_now = MagicMock()
        fake_now.strftime.return_value = MODEL_RETRAIN_DAY.capitalize()

        with patch.multiple(
            "live",
            model_trainer=mock_trainer,
            alert_bot=mock_bot,
            evolution_manager=mock_evo,
        ):
            with patch("live.datetime") as mock_dt:
                mock_dt.now.return_value = fake_now
                mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                live.daily_retrain_midnight()

        mock_trainer.train_meta_label_model.assert_not_called()
