"""
Model Predictor — loads trained models and provides real-time predictions.
The system is fully functional without any models loaded;
ML only adds informational probability to alerts, never blocks signals.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger("predictor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/predictor.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

MODELS_DIR = "models"


class ModelPredictor:
    """Loads persisted models and returns predictions + SHAP explanations."""

    def __init__(self, signal_logger=None, model_trainer=None) -> None:
        self.models: Dict[str, Any] = {}
        self._shap_explainer = None
        self._m3_features: List[str] = []
        self._signal_logger = signal_logger
        self._trainer = model_trainer

    # ─── model loading ───────────────────────────────────────

    def load_all_models(self) -> None:
        """Attempt to load each model pkl; missing files are silently skipped."""
        model_files = {
            "model1": os.path.join(MODELS_DIR, "model1_day_quality.pkl"),
            "model2": os.path.join(MODELS_DIR, "model2_session_bias.pkl"),
            "model3": os.path.join(MODELS_DIR, "model3_meta_label.pkl"),
        }

        for name, path in model_files.items():
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        self.models[name] = pickle.load(f)
                    logger.info("Loaded %s from %s", name, path)
                else:
                    logger.info("Model file not found: %s — skipping", path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", name, exc)

        shap_path = os.path.join(MODELS_DIR, "model3_shap.pkl")
        try:
            if os.path.exists(shap_path):
                with open(shap_path, "rb") as f:
                    self._shap_explainer = pickle.load(f)
                logger.info("SHAP explainer loaded")
        except Exception as exc:
            logger.warning("SHAP explainer load failed: %s", exc)

        features_path = os.path.join(MODELS_DIR, "model3_features.json")
        try:
            if os.path.exists(features_path):
                with open(features_path) as f:
                    self._m3_features = json.load(f)
                logger.info("Model 3 feature list loaded (%d features)", len(self._m3_features))
        except Exception as exc:
            logger.warning("Model 3 features load failed: %s", exc)

    # ─── Model 1: Day Quality ────────────────────────────────

    def predict_day_quality(self, macro_features: Dict[str, Any]) -> Optional[float]:
        """Return P(good trade day) in [0, 1], or None if model unavailable.

        NOTE: Models 1/2 are trained on EOD (end-of-day) macro values but receive
        live intraday snapshots at inference.  The feature shift (Bug #10 fix)
        means we use *yesterday's* EOD features to predict *today*, which is the
        correct causal direction.  Intraday VIX/DXY pct_change still differs in
        distribution from EOD values — this is a known limitation until models
        are retrained on intraday snapshot data.
        """
        try:
            bundle = self.models.get("model1")
            if bundle is None:
                return None

            model = bundle["model"]
            feature_cols = bundle["features"]
            row = {c: macro_features.get(c, 0) for c in feature_cols}
            X = pd.DataFrame([row])[feature_cols]
            proba = model.predict_proba(X)[0]
            return float(proba[1]) if len(proba) > 1 else float(proba[0])

        except Exception as exc:
            logger.error("predict_day_quality failed: %s", exc, exc_info=True)
            return None

    # ─── Model 2: Session Bias ───────────────────────────────

    def predict_session_bias(self, macro_features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return predicted direction + confidence, or None if model unavailable."""
        try:
            bundle = self.models.get("model2")
            if bundle is None:
                return None

            model = bundle["model"]
            feature_cols = bundle["features"]
            row = {c: macro_features.get(c, 0) for c in feature_cols}
            X = pd.DataFrame([row])[feature_cols]

            proba = model.predict_proba(X)[0]
            pred = int(model.predict(X)[0])
            direction = "LONG" if pred == 1 else "SHORT"
            confidence = float(max(proba))

            return {"direction": direction, "confidence": round(confidence, 3)}

        except Exception as exc:
            logger.error("predict_session_bias failed: %s", exc, exc_info=True)
            return None

    # ─── Model 3: Meta-Label (win probability) ───────────────

    def predict_win_probability(self, checklist_result: Dict[str, Any]) -> Optional[float]:
        """Return P(win) for this specific signal, or None if model unavailable."""
        try:
            bundle = self.models.get("model3")
            if bundle is None:
                return None

            model = bundle["model"]
            feature_cols = bundle["features"]

            if self._trainer is None:
                return None
            flat = self._flatten_checklist(checklist_result)
            X_df = self._trainer.prepare_checklist_features(pd.DataFrame([flat]))

            for col in feature_cols:
                if col not in X_df.columns:
                    X_df[col] = float("nan")
            X_df = X_df[feature_cols]

            proba = model.predict_proba(X_df)[0]
            if len(proba) == 0:
                return None
            win_prob = float(proba[1]) if len(proba) == 2 else float(proba[-1])
            return round(win_prob, 3)

        except Exception as exc:
            logger.error("predict_win_probability failed: %s", exc, exc_info=True)
            return None

    # ─── SHAP explanation ────────────────────────────────────

    def get_shap_explanation(self, checklist_result: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Return top 3 SHAP feature explanations for this prediction.
        If checklist_result is None, return the global SHAP report.
        """
        try:
            if self._shap_explainer is None:
                return None

            bundle = self.models.get("model3")
            if bundle is None:
                return None

            feature_cols = bundle["features"]

            if self._trainer is None:
                return None
            if checklist_result is None:
                return self._trainer.get_shap_report()

            flat = self._flatten_checklist(checklist_result)
            X_df = self._trainer.prepare_checklist_features(pd.DataFrame([flat]))

            for col in feature_cols:
                if col not in X_df.columns:
                    X_df[col] = float("nan")
            X_df = X_df[feature_cols]

            shap_values = self._shap_explainer.shap_values(X_df.values)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            vals = shap_values[0]
            top_idx = np.argsort(np.abs(vals))[::-1][:3]
            explanations = []
            for i in top_idx:
                fname = feature_cols[i] if i < len(feature_cols) else f"feature_{i}"
                explanations.append(f"{fname} ({vals[i]:+.2f})")
            return explanations

        except Exception as exc:
            logger.error("get_shap_explanation failed: %s", exc, exc_info=True)
            return None

    # ─── composite ML enhancement ────────────────────────────

    def get_ml_enhancement(
        self, checklist_result: Dict[str, Any], macro_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run all available predictions and return a consolidated dict.
        ML output NEVER blocks a signal — it only provides informational probability.
        """
        try:
            macro_features = self._extract_macro_features(macro_data)

            day_quality = self.predict_day_quality(macro_features)
            session_bias = self.predict_session_bias(macro_features)
            win_prob = self.predict_win_probability(checklist_result)
            shap_feats = self.get_shap_explanation(checklist_result)

            loaded = [name for name in ["model1", "model2", "model3"] if name in self.models]

            return {
                "day_quality_prob": day_quality,
                "session_bias_direction": session_bias["direction"] if session_bias else None,
                "session_bias_confidence": session_bias["confidence"] if session_bias else None,
                "win_probability": win_prob,
                "shap_top_features": shap_feats,
                "models_loaded": loaded,
                "ml_active": len(loaded) > 0,
            }

        except Exception as exc:
            logger.error("get_ml_enhancement failed: %s", exc, exc_info=True)
            return {
                "day_quality_prob": None, "session_bias_direction": None,
                "session_bias_confidence": None, "win_probability": None,
                "shap_top_features": None, "models_loaded": [], "ml_active": False,
            }

    # ─── internal helpers ────────────────────────────────────

    def _flatten_checklist(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested checklist result into a single-level dict for DataFrame creation."""
        flat: Dict[str, Any] = {}
        for key in ("layer1", "layer2", "layer3", "layer4", "entry"):
            sub = result.get(key)
            if isinstance(sub, dict):
                flat.update(sub)
        for k, v in result.items():
            if k not in ("layer1", "layer2", "layer3", "layer4", "entry", "ml"):
                flat[k] = v
        return flat

    @staticmethod
    def _extract_macro_features(macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a flat dict of macro features compatible with Model 1 / Model 2 inputs.

        Computes as many features as possible from live macro data.  Multi-day
        features (3-day changes, consecutive runs, volatility) require historical
        data we don't cache, so they default to 0 — these are low-importance
        features in the trained models.
        """
        raw = macro_data.get("raw_data", macro_data)
        features: Dict[str, Any] = {}

        vix = raw.get("vix", {})
        vix_pct = vix.get("pct_change", macro_data.get("vix_pct", 0)) or 0
        vix_val = vix.get("value", macro_data.get("vix_value", 0)) or 0
        features["vix_pct"] = vix_pct

        dxy = raw.get("dxy", {})
        dxy_pct = dxy.get("pct_change", 0) or 0
        features["dxy_pct"] = dxy_pct

        us10y = raw.get("us10y", {})
        features["us10y_pct"] = us10y.get("pct_change", 0) or 0

        oil = raw.get("oil", {})
        features["oil_pct"] = oil.get("pct_change", 0) or 0

        rut = raw.get("rut", {})
        rut_pct = rut.get("pct_change", 0) or 0
        sp500 = raw.get("sp500", raw.get("sp500_return", {}))
        sp500_pct = sp500.get("pct_change", 0) if isinstance(sp500, dict) else 0

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        features["day_of_week"] = now.weekday()
        features["is_monday"] = int(now.weekday() == 0)
        features["is_friday"] = int(now.weekday() == 4)
        features["is_tuesday"] = int(now.weekday() == 1)
        features["week_of_month"] = min((now.day - 1) // 7 + 1, 4)

        features["vix_spike_20"] = int(vix_val > 20)
        features["vix_spike_25"] = int(vix_val > 25)
        features["vix_spike_30"] = int(vix_val > 30)
        features["vix_trend"] = 1 if vix_pct > 0 else -1
        features["rut_vs_sp500"] = rut_pct - sp500_pct
        features["rut_leading"] = 1 if (rut_pct - sp500_pct) > 0 else -1
        features["vix_up_dxy_up"] = int(vix_pct > 1 and dxy_pct > 0.2)
        features["vix_down_rut_up"] = int(vix_pct < -1 and (rut_pct - sp500_pct) > 0)
        features["vix_spike_friday"] = int(vix_val > 20 and now.weekday() == 4)
        features["yield_shock"] = int(abs(features["us10y_pct"]) > 0.1)

        for k in ["vix_3day_change", "dxy_3day", "us10y_3day", "oil_3day",
                   "overnight_gap", "prev_day_return", "consecutive_up",
                   "consecutive_down", "volatility_5d"]:
            features.setdefault(k, 0)

        return features
