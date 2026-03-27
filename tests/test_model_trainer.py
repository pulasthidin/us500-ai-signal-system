"""Tests for ModelTrainer — feature preparation, encoding maps, META_FEATURE_COLS."""

import pytest
import pandas as pd
import numpy as np
from src.model_trainer import ModelTrainer


@pytest.fixture
def trainer():
    return ModelTrainer(signal_logger=None, alert_bot=None)


@pytest.fixture
def sample_signal_row():
    """A single signal row with all fields populated (as it would appear from the DB)."""
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


class TestChecklistFeatureMap:
    def test_all_categorical_mappings_exist(self, trainer):
        expected = [
            "direction", "macro_bias", "session", "vix_bucket",
            "tp_source", "entry_confidence", "direction_confidence",
            "range_strength", "sweep_side", "sweep_level_type", "sl_method",
        ]
        for col in expected:
            assert col in trainer.CHECKLIST_FEATURE_MAP, f"Missing mapping: {col}"

    def test_entry_confidence_includes_no_sweep(self, trainer):
        mapping = trainer.CHECKLIST_FEATURE_MAP["entry_confidence"]
        assert "no_sweep" in mapping
        assert "full" in mapping
        assert mapping["full"] > mapping["no_sweep"]

    def test_sweep_side_mapping(self, trainer):
        mapping = trainer.CHECKLIST_FEATURE_MAP["sweep_side"]
        assert "buy_side" in mapping
        assert "sell_side" in mapping

    def test_sl_method_mapping(self, trainer):
        mapping = trainer.CHECKLIST_FEATURE_MAP["sl_method"]
        assert "range" in mapping
        assert "swing" in mapping
        assert "atr" in mapping


class TestMetaFeatureCols:
    def test_includes_original_features(self, trainer):
        originals = ["vix_level", "rr", "atr", "score", "fvg_present", "at_zone"]
        for col in originals:
            assert col in trainer.META_FEATURE_COLS, f"Missing original: {col}"

    def test_includes_new_range_features(self, trainer):
        range_cols = ["is_ranging", "adx_value", "atr_compressing",
                      "range_size_points", "price_in_range_pct"]
        for col in range_cols:
            assert col in trainer.META_FEATURE_COLS, f"Missing range feature: {col}"

    def test_includes_new_sweep_features(self, trainer):
        sweep_cols = ["has_liquidity_sweep", "sweep_bars_ago", "sweep_side_enc"]
        for col in sweep_cols:
            assert col in trainer.META_FEATURE_COLS, f"Missing sweep feature: {col}"

    def test_includes_displacement_features(self, trainer):
        assert "displacement_valid" in trainer.META_FEATURE_COLS

    def test_includes_previously_ignored_features(self, trainer):
        previously_ignored = ["ob_bullish_nearby", "ob_bearish_nearby",
                              "eqh_nearby", "eql_nearby", "sl_points", "tp_points"]
        for col in previously_ignored:
            assert col in trainer.META_FEATURE_COLS, f"Still missing: {col}"

    def test_includes_computed_features(self, trainer):
        computed = ["fvg_age_bars", "fvg_size_points", "asian_range_size", "hour_utc", "data_version"]
        for col in computed:
            assert col in trainer.META_FEATURE_COLS, f"Missing computed: {col}"

    def test_includes_new_encoded_categoricals(self, trainer):
        new_encoded = ["tp_source_enc", "entry_confidence_enc", "direction_confidence_enc",
                       "range_strength_enc", "sweep_side_enc", "sweep_level_type_enc", "sl_method_enc"]
        for col in new_encoded:
            assert col in trainer.META_FEATURE_COLS, f"Missing encoded: {col}"

    def test_feature_count_at_least_59(self, trainer):
        assert len(trainer.META_FEATURE_COLS) >= 59


class TestPrepareChecklistFeatures:
    def test_produces_dataframe_with_all_columns(self, trainer, sample_signal_row):
        df = pd.DataFrame([sample_signal_row])
        result = trainer.prepare_checklist_features(df)
        assert not result.empty
        assert len(result.columns) > 30

    def test_encodes_direction_correctly(self, trainer, sample_signal_row):
        df = pd.DataFrame([sample_signal_row])
        result = trainer.prepare_checklist_features(df)
        assert "direction_enc" in result.columns
        assert result["direction_enc"].iloc[0] == 1

    def test_encodes_range_strength(self, trainer, sample_signal_row):
        df = pd.DataFrame([sample_signal_row])
        result = trainer.prepare_checklist_features(df)
        assert "range_strength_enc" in result.columns
        assert result["range_strength_enc"].iloc[0] == 2

    def test_encodes_sweep_side(self, trainer, sample_signal_row):
        df = pd.DataFrame([sample_signal_row])
        result = trainer.prepare_checklist_features(df)
        assert "sweep_side_enc" in result.columns
        assert result["sweep_side_enc"].iloc[0] == 1

    def test_encodes_entry_confidence_no_sweep(self, trainer, sample_signal_row):
        row = sample_signal_row.copy()
        row["entry_confidence"] = "no_sweep"
        df = pd.DataFrame([row])
        result = trainer.prepare_checklist_features(df)
        assert result["entry_confidence_enc"].iloc[0] == 1

    def test_handles_null_values_gracefully(self, trainer):
        """Simulates old v1 data with NULL in new columns."""
        row = {
            "direction": "SHORT", "session": "NY_Session", "grade": "B",
            "vix_level": 20.0, "score": 3, "rr": 1.8, "atr": 12.0,
            "is_ranging": None, "adx_value": None,
            "has_liquidity_sweep": None, "displacement_valid": None,
            "data_version": 1,
        }
        df = pd.DataFrame([row])
        result = trainer.prepare_checklist_features(df)
        assert not result.empty
        if "is_ranging" in result.columns:
            assert result["is_ranging"].iloc[0] == 0

    def test_bool_columns_filled_with_zero(self, trainer, sample_signal_row):
        row = sample_signal_row.copy()
        row["displacement_valid"] = None
        row["has_liquidity_sweep"] = None
        df = pd.DataFrame([row])
        result = trainer.prepare_checklist_features(df)
        assert result["displacement_valid"].iloc[0] == 0
        assert result["has_liquidity_sweep"].iloc[0] == 0

    def test_numeric_columns_preserved(self, trainer, sample_signal_row):
        df = pd.DataFrame([sample_signal_row])
        result = trainer.prepare_checklist_features(df)
        if "adx_value" in result.columns:
            assert abs(result["adx_value"].iloc[0] - 15.0) < 0.1
        if "fvg_size_points" in result.columns:
            assert abs(result["fvg_size_points"].iloc[0] - 5.0) < 0.1

    def test_empty_dataframe_returns_empty(self, trainer):
        result = trainer.prepare_checklist_features(pd.DataFrame())
        assert result.empty
