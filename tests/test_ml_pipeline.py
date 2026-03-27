"""Tests for the full ML training and prediction pipeline.

Covers: _drop_zero_variance, train_meta_label_model, check_and_retrain,
retrain timestamp persistence, predict_win_probability,
_extract_macro_features, get_ml_enhancement.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.model_trainer import ModelTrainer
from src.model_predictor import ModelPredictor


# ─── helpers ──────────────────────────────────────────────


def _make_signal_data(n=210, timeout_count=0):
    """Generate synthetic signal data matching the DB schema."""
    np.random.seed(42)
    rows = []
    for i in range(n):
        outcome = "TIMEOUT" if i < timeout_count else np.random.choice(["WIN", "LOSS"])
        rows.append({
            "outcome": outcome,
            "outcome_label": int(np.random.choice([1, -1])),
            "direction": np.random.choice(["LONG", "SHORT"]),
            "macro_bias": np.random.choice(["LONG", "SHORT", "MIXED"]),
            "session": np.random.choice(["London", "NY_Open_Killzone", "NY_Session"]),
            "vix_bucket": np.random.choice(["low", "normal", "elevated"]),
            "vix_direction_bias": np.random.choice(["BUY_BIAS", "SELL_BIAS", "NEUTRAL"]),
            "us10y_direction": np.random.choice(["falling", "flat", "rising"]),
            "oil_direction": np.random.choice(["falling", "stable", "spiking"]),
            "dxy_direction": np.random.choice(["falling", "flat", "rising"]),
            "rut_direction": np.random.choice(["red", "neutral", "green"]),
            "structure_bias": np.random.choice(["long", "short", "unclear"]),
            "h4_bos": np.random.choice(["bullish", "bearish"]),
            "wyckoff": np.random.choice(["markup", "markdown", "unclear"]),
            "delta_direction": np.random.choice(["buyers", "sellers", "mixed"]),
            "divergence": np.random.choice(["bullish", "bearish", "none"]),
            "zone_type": np.random.choice(["round", "pdh", "pdl", "poc"]),
            "zone_direction": np.random.choice(["support", "resistance", "at_level"]),
            "groq_sentiment": np.random.choice(["BULLISH", "BEARISH", "NEUTRAL"]),
            "grade": np.random.choice(["A", "B"]),
            "tp_source": np.random.choice(["eqh", "pdh", "round", "atr"]),
            "entry_confidence": np.random.choice(["full", "high", "base"]),
            "direction_confidence": np.random.choice(["high", "medium", "reduced"]),
            "range_strength": np.random.choice(["strong", "mild", "none"]),
            "sweep_side": np.random.choice(["buy_side", "sell_side"]),
            "sweep_level_type": np.random.choice(["pdh", "eqh", "round"]),
            "sl_method": np.random.choice(["range", "swing", "fvg", "atr"]),
            "vix_level": float(np.random.uniform(12, 30)),
            "vix_pct": float(np.random.uniform(-5, 5)),
            "bullish_count": int(np.random.randint(0, 5)),
            "bearish_count": int(np.random.randint(0, 5)),
            "above_ema50": int(np.random.choice([0, 1])),
            "choch_recent": int(np.random.choice([0, 1])),
            "ob_bullish_nearby": int(np.random.choice([0, 1])),
            "ob_bearish_nearby": int(np.random.choice([0, 1])),
            "at_zone": int(np.random.choice([0, 1])),
            "zone_distance": float(np.random.uniform(0, 50)),
            "eqh_nearby": int(np.random.choice([0, 1])),
            "eql_nearby": int(np.random.choice([0, 1])),
            "confirms_bias": int(np.random.choice([0, 1])),
            "vix_spiking_now": int(np.random.choice([0, 1])),
            "fvg_present": int(np.random.choice([0, 1])),
            "m5_bos": int(np.random.choice([0, 1])),
            "ustec_agrees": int(np.random.choice([0, 1])),
            "rr": float(np.random.uniform(1, 5)),
            "atr": float(np.random.uniform(5, 20)),
            "sl_points": float(np.random.uniform(5, 30)),
            "tp_points": float(np.random.uniform(10, 50)),
            "displacement_valid": int(np.random.choice([0, 1])),
            "has_liquidity_sweep": int(np.random.choice([0, 1])),
            "fvg_age_bars": int(np.random.randint(1, 10)),
            "fvg_size_points": float(np.random.uniform(1, 10)),
            "is_ranging": int(np.random.choice([0, 1])),
            "adx_value": float(np.random.uniform(10, 40)),
            "atr_compressing": int(np.random.choice([0, 1])),
            "range_size_points": float(np.random.uniform(20, 100)),
            "price_in_range_pct": float(np.random.uniform(0, 100)),
            "asian_range_size": float(np.random.uniform(5, 40)),
            "sweep_bars_ago": int(np.random.randint(1, 20)),
            "score": int(np.random.randint(1, 6)),
            "is_monday": int(np.random.choice([0, 1])),
            "is_friday": int(np.random.choice([0, 1])),
            "is_news_day": int(np.random.choice([0, 1])),
            "hour_utc": int(np.random.randint(6, 21)),
            "data_version": np.random.choice([1, 2]),
        })
    return pd.DataFrame(rows)


def _make_checklist_result():
    """A flat checklist result with all fields populated."""
    return {
        "direction": "LONG", "macro_bias": "LONG", "session": "London",
        "vix_bucket": "normal", "vix_direction_bias": "BUY_BIAS",
        "us10y_direction": "falling", "oil_direction": "falling",
        "dxy_direction": "flat", "rut_direction": "green",
        "structure_bias": "long", "h4_bos": "bullish", "wyckoff": "markup",
        "delta_direction": "buyers", "divergence": "none",
        "zone_type": "round", "zone_direction": "support",
        "groq_sentiment": "NEUTRAL", "grade": "A",
        "tp_source": "eqh", "entry_confidence": "full",
        "direction_confidence": "high", "range_strength": "strong",
        "sweep_side": "sell_side", "sweep_level_type": "pdl",
        "sl_method": "range",
        "vix_level": 17.0, "vix_pct": -2.0,
        "bullish_count": 3, "bearish_count": 0,
        "above_ema50": 1, "choch_recent": 0,
        "ob_bullish_nearby": 1, "ob_bearish_nearby": 0,
        "at_zone": 1, "zone_distance": 3.0,
        "eqh_nearby": 0, "eql_nearby": 1,
        "confirms_bias": 1, "vix_spiking_now": 0,
        "fvg_present": 1, "m5_bos": 1, "ustec_agrees": 1,
        "rr": 2.5, "atr": 10.0, "sl_points": 15.0, "tp_points": 25.0,
        "displacement_valid": 1, "has_liquidity_sweep": 1,
        "fvg_age_bars": 3, "fvg_size_points": 5.0,
        "is_ranging": 1, "adx_value": 15.0, "atr_compressing": 1,
        "range_size_points": 60.0, "price_in_range_pct": 50.0,
        "asian_range_size": 20.0, "sweep_bars_ago": 5,
        "score": 4, "is_monday": 0, "is_friday": 0, "is_news_day": 0,
        "hour_utc": 14, "data_version": 2,
    }


def _train_tiny_model(features):
    """Train a minimal XGBClassifier for prediction tests."""
    np.random.seed(42)
    X = np.random.rand(20, len(features))
    y = np.array([0, 1] * 10)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y, verbose=False)
    return {"model": model, "features": features}


# ══════════════════════════════════════════════════════════
# _drop_zero_variance
# ══════════════════════════════════════════════════════════


class TestDropZeroVariance:
    def test_drops_all_nan_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
        result, dropped = ModelTrainer._drop_zero_variance(df)
        assert "b" in dropped
        assert "b" not in result.columns
        assert "a" in result.columns

    def test_drops_single_value_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
        result, dropped = ModelTrainer._drop_zero_variance(df)
        assert "b" in dropped
        assert "a" not in dropped

    def test_keeps_two_value_column(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": [0, 1, 0]})
        result, dropped = ModelTrainer._drop_zero_variance(df)
        assert dropped == []
        assert list(result.columns) == ["a", "b"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result, dropped = ModelTrainer._drop_zero_variance(df)
        assert result.empty
        assert dropped == []

    def test_preserves_row_count(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [7, 7, 7, 7, 7],
            "c": [10, 20, 30, 40, 50],
        })
        result, dropped = ModelTrainer._drop_zero_variance(df)
        assert len(result) == 5
        assert "b" in dropped
        assert set(result.columns) == {"a", "c"}


# ══════════════════════════════════════════════════════════
# train_meta_label_model
# ══════════════════════════════════════════════════════════


class TestTrainMetaLabelModel:
    def test_trains_with_enough_data(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_training_data.return_value = _make_signal_data(210)
        signal_logger.get_signal_count.return_value = 210
        trainer = ModelTrainer(signal_logger=signal_logger, alert_bot=None)

        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)), \
             patch("src.model_trainer.shap", None):
            result = trainer.train_meta_label_model()

        assert result is True
        assert (tmp_path / "model3_meta_label.pkl").exists()
        assert (tmp_path / "model3_features.json").exists()

    def test_rejects_insufficient_data(self):
        signal_logger = MagicMock()
        signal_logger.get_training_data.return_value = _make_signal_data(50)
        trainer = ModelTrainer(signal_logger=signal_logger)
        assert trainer.train_meta_label_model() is False

    def test_excludes_timeout_outcomes(self):
        signal_logger = MagicMock()
        signal_logger.get_training_data.return_value = _make_signal_data(200, timeout_count=50)
        trainer = ModelTrainer(signal_logger=signal_logger)
        assert trainer.train_meta_label_model() is False

    def test_saves_feature_list_json(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_training_data.return_value = _make_signal_data(210)
        signal_logger.get_signal_count.return_value = 210
        trainer = ModelTrainer(signal_logger=signal_logger)

        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)), \
             patch("src.model_trainer.shap", None):
            trainer.train_meta_label_model()

        with open(tmp_path / "model3_features.json") as f:
            features = json.load(f)
        assert isinstance(features, list)
        assert all(isinstance(feat, str) for feat in features)
        assert len(features) > 0

    def test_returns_false_without_signal_logger(self):
        trainer = ModelTrainer(signal_logger=None)
        assert trainer.train_meta_label_model() is False


# ══════════════════════════════════════════════════════════
# check_and_retrain
# ══════════════════════════════════════════════════════════


class TestCheckAndRetrain:
    def test_triggers_retrain_with_enough_new_signals(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_signal_count.return_value = 250
        trainer = ModelTrainer(signal_logger=signal_logger)

        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)), \
             patch.object(trainer, "train_meta_label_model", return_value=True) as mock_train, \
             patch.object(trainer, "fetch_historical_data", return_value=pd.DataFrame()):
            trainer.check_and_retrain()

        mock_train.assert_called_once()

    def test_skips_retrain_with_few_new_signals(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_signal_count.return_value = 260

        retrain_data = {
            "signal_count": 240,
            "last_retrain": datetime.now(timezone.utc).isoformat(),
            "model1_date": datetime.now(timezone.utc).isoformat(),
        }
        (tmp_path / "last_retrain.json").write_text(json.dumps(retrain_data))

        trainer = ModelTrainer(signal_logger=signal_logger)
        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)), \
             patch.object(trainer, "train_meta_label_model") as mock_train:
            trainer.check_and_retrain()

        mock_train.assert_not_called()

    def test_monthly_models_1_2_trigger(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_signal_count.return_value = 100

        old_date = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        retrain_data = {"signal_count": 100, "model1_date": old_date}
        (tmp_path / "last_retrain.json").write_text(json.dumps(retrain_data))

        trainer = ModelTrainer(signal_logger=signal_logger)
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)), \
             patch.object(trainer, "fetch_historical_data", return_value=mock_df) as mock_fetch, \
             patch.object(trainer, "train_day_quality_model") as mock_m1, \
             patch.object(trainer, "train_session_bias_model") as mock_m2, \
             patch.object(trainer, "train_meta_label_model"):
            trainer.check_and_retrain()

        mock_fetch.assert_called_once()
        mock_m1.assert_called_once_with(mock_df)
        mock_m2.assert_called_once_with(mock_df)

    def test_skips_monthly_if_recent(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_signal_count.return_value = 100

        recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        retrain_data = {"signal_count": 100, "model1_date": recent_date}
        (tmp_path / "last_retrain.json").write_text(json.dumps(retrain_data))

        trainer = ModelTrainer(signal_logger=signal_logger)

        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)), \
             patch.object(trainer, "fetch_historical_data") as mock_fetch:
            trainer.check_and_retrain()

        mock_fetch.assert_not_called()


# ══════════════════════════════════════════════════════════
# Retrain timestamp persistence
# ══════════════════════════════════════════════════════════


class TestRetrainTimestamp:
    def test_save_and_load_roundtrip(self, tmp_path):
        signal_logger = MagicMock()
        signal_logger.get_signal_count.return_value = 150
        trainer = ModelTrainer(signal_logger=signal_logger)

        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)):
            trainer._save_retrain_timestamp()
            loaded = trainer._load_retrain_timestamp()

        assert "last_retrain" in loaded
        assert "signal_count" in loaded
        assert loaded["signal_count"] == 150

    def test_missing_file_returns_empty(self, tmp_path):
        trainer = ModelTrainer()
        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)):
            result = trainer._load_retrain_timestamp()
        assert result == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        (tmp_path / "last_retrain.json").write_text("not valid json {{{")
        trainer = ModelTrainer()
        with patch("src.model_trainer.MODELS_DIR", str(tmp_path)):
            result = trainer._load_retrain_timestamp()
        assert result == {}


# ══════════════════════════════════════════════════════════
# predict_win_probability
# ══════════════════════════════════════════════════════════


class TestPredictWinProbability:
    def test_positive_prediction(self):
        features = ["vix_level", "rr", "score"]
        bundle = _train_tiny_model(features)

        trainer = ModelTrainer()
        predictor = ModelPredictor(model_trainer=trainer)
        predictor.models["model3"] = bundle

        result = predictor.predict_win_probability(_make_checklist_result())
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_returns_none_without_model(self):
        predictor = ModelPredictor()
        result = predictor.predict_win_probability(_make_checklist_result())
        assert result is None

    def test_handles_feature_mismatch(self):
        features = ["vix_level", "rr", "score"]
        bundle = _train_tiny_model(features)

        trainer = ModelTrainer()
        predictor = ModelPredictor(model_trainer=trainer)
        predictor.models["model3"] = bundle

        checklist = _make_checklist_result()
        del checklist["score"]

        result = predictor.predict_win_probability(checklist)
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ══════════════════════════════════════════════════════════
# _extract_macro_features
# ══════════════════════════════════════════════════════════


class TestExtractMacroFeatures:
    def test_all_model1_features_present(self, sample_macro_data):
        features = ModelPredictor._extract_macro_features(sample_macro_data)
        m1_cols = ModelTrainer()._get_feature_cols_m1()
        for col in m1_cols:
            assert col in features, f"Missing M1 feature: {col}"

    def test_handles_empty_raw_data(self):
        features = ModelPredictor._extract_macro_features({})
        assert isinstance(features, dict)
        assert len(features) > 0
        for v in features.values():
            assert v is not None


# ══════════════════════════════════════════════════════════
# get_ml_enhancement
# ══════════════════════════════════════════════════════════


class TestGetMlEnhancement:
    def test_returns_safe_dict_without_models(self, sample_macro_data):
        predictor = ModelPredictor()
        result = predictor.get_ml_enhancement(_make_checklist_result(), sample_macro_data)
        assert result["ml_active"] is False
        assert result["win_probability"] is None
        assert result["day_quality_prob"] is None
        assert result["session_bias_direction"] is None
        assert result["models_loaded"] == []

    def test_returns_full_dict_with_models(self, sample_macro_data):
        features = ["vix_level", "rr", "score"]
        bundle = _train_tiny_model(features)

        trainer = ModelTrainer()
        predictor = ModelPredictor(model_trainer=trainer)
        predictor.models["model3"] = bundle

        result = predictor.get_ml_enhancement(_make_checklist_result(), sample_macro_data)
        assert result["ml_active"] is True
        assert "model3" in result["models_loaded"]
        assert result["win_probability"] is not None
        assert isinstance(result["win_probability"], float)
