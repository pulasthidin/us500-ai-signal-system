"""
Model Trainer — trains three ML models:
  1. Day Quality (XGBoost binary) — "Is today a trending day?"
  2. Session Bias (LightGBM binary) — "Which direction?"
  3. Meta-Label (XGBoost multiclass) — "Will this checklist signal win?"

Walk-forward validation, SHAP explanations, auto-retrain scheduling.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    shap = None
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

import yfinance as yf

from config import (
    MODEL_INCREMENTAL_THRESHOLD,
    MODEL_RETRAIN_SIGNAL_THRESHOLD,
    YF_TICKERS,
)

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("trainer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/trainer.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

MODELS_DIR = "models"
DATA_FEAT_DIR = os.path.join("data", "features")


class ModelTrainer:
    """Trains and persists all ML models used by the signal app."""

    def __init__(self, signal_logger=None, alert_bot=None) -> None:
        self._signal_logger = signal_logger
        self._alert_bot = alert_bot
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(DATA_FEAT_DIR, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # PHASE 1 — HISTORICAL FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════

    def fetch_historical_data(self) -> pd.DataFrame:
        """Pull 5 years of daily data and engineer every feature."""
        try:
            tickers_str = " ".join(YF_TICKERS.values())
            raw = yf.download(tickers_str, period="5y", interval="1d", group_by="ticker", progress=False)

            df = pd.DataFrame()

            for key, ticker in YF_TICKERS.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                        sub = raw[ticker].copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
                        continue
                    else:
                        sub = raw.copy()
                    df[f"{key}_close"] = sub["Close"]
                    df[f"{key}_open"] = sub["Open"]
                    df[f"{key}_high"] = sub["High"]
                    df[f"{key}_low"] = sub["Low"]
                    df[f"{key}_volume"] = sub.get("Volume", 0)
                except Exception as e:
                    logger.warning("Could not extract %s: %s", key, e)

            df.dropna(subset=["sp500_close"], inplace=True)

            df["sp500_return"] = df["sp500_close"].pct_change() * 100
            df["sp500_next_return"] = df["sp500_return"].shift(-1)
            df["sp500_direction"] = np.where(df["sp500_next_return"] >= 0, 1, -1)

            vix_close = df["vix_close"]
            df["vix_pct"] = vix_close.pct_change() * 100
            # vix_trend must match inference: sign of pct_change (not EMA crossover)
            # so that training and live prediction use identical semantics.
            df["vix_trend"] = np.where(df["vix_pct"] > 0, 1, -1)
            df["vix_spike_20"] = (vix_close > 20).astype(int)
            df["vix_spike_25"] = (vix_close > 25).astype(int)
            df["vix_spike_30"] = (vix_close > 30).astype(int)
            df["vix_3day_change"] = vix_close.pct_change(3) * 100

            df["dxy_pct"] = df["dxy_close"].pct_change() * 100
            df["dxy_3day"] = df["dxy_close"].pct_change(3) * 100
            df["us10y_pct"] = df["us10y_close"].pct_change() * 100
            df["us10y_3day"] = df["us10y_close"].pct_change(3) * 100
            df["oil_pct"] = df["oil_close"].pct_change() * 100
            df["oil_3day"] = df["oil_close"].pct_change(3) * 100

            rut_ret = df["rut_close"].pct_change() * 100
            df["rut_vs_sp500"] = rut_ret - df["sp500_return"]
            df["rut_leading"] = np.where(df["rut_vs_sp500"] > 0, 1, -1)

            df["day_of_week"] = pd.to_datetime(df.index).dayofweek
            df["is_monday"] = (df["day_of_week"] == 0).astype(int)
            df["is_friday"] = (df["day_of_week"] == 4).astype(int)
            df["is_tuesday"] = (df["day_of_week"] == 1).astype(int)
            df["week_of_month"] = pd.to_datetime(df.index).day.map(lambda d: min((d - 1) // 7 + 1, 4))

            prev_close = df["sp500_close"].shift(1)
            df["overnight_gap"] = np.where(
                prev_close != 0,
                ((df["sp500_open"] - prev_close) / prev_close) * 100,
                0.0,
            )
            df["prev_day_return"] = df["sp500_return"].shift(1)

            up = (df["sp500_return"] > 0).astype(int)
            down = (df["sp500_return"] < 0).astype(int)
            df["consecutive_up"] = up.groupby((up != up.shift()).cumsum()).cumsum()
            df["consecutive_down"] = down.groupby((down != down.shift()).cumsum()).cumsum()
            df["volatility_5d"] = df["sp500_return"].rolling(5).std()

            df["vix_up_dxy_up"] = ((df["vix_pct"] > 1) & (df["dxy_pct"] > 0.2)).astype(int)
            df["vix_down_rut_up"] = ((df["vix_pct"] < -1) & (df["rut_vs_sp500"] > 0)).astype(int)
            df["vix_spike_friday"] = (df["vix_spike_20"] & df["is_friday"]).astype(int)
            df["yield_shock"] = (df["us10y_pct"].abs() > 0.1).astype(int)

            df.dropna(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            path = os.path.join(DATA_FEAT_DIR, "historical_features.parquet")
            df.to_parquet(path)
            logger.info("Historical features saved (%d rows) -> %s", len(df), path)
            return df

        except Exception as exc:
            logger.error("fetch_historical_data failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "historical_data", str(exc))
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════
    # MODEL 1 — DAY QUALITY (XGBoost binary)
    # ══════════════════════════════════════════════════════════

    def _get_feature_cols_m1(self) -> List[str]:
        return [
            "vix_pct", "vix_trend", "vix_spike_20", "vix_spike_25", "vix_spike_30",
            "vix_3day_change", "dxy_pct", "dxy_3day", "us10y_pct", "us10y_3day",
            "oil_pct", "oil_3day", "rut_vs_sp500", "rut_leading",
            "day_of_week", "is_monday", "is_friday", "is_tuesday", "week_of_month",
            "overnight_gap", "prev_day_return", "consecutive_up", "consecutive_down",
            "volatility_5d", "vix_up_dxy_up", "vix_down_rut_up", "vix_spike_friday", "yield_shock",
        ]

    def train_day_quality_model(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        XGBoost binary classifier: is today a "good trade day"?
        Label 1 if absolute return > 0.5% (trending), else 0.
        Walk-forward validation across 5 windows.
        """
        try:
            if df is None:
                path = os.path.join(DATA_FEAT_DIR, "historical_features.parquet")
                df = pd.read_parquet(path) if os.path.exists(path) else self.fetch_historical_data()
            if df.empty:
                logger.warning("No data for Model 1")
                return

            feature_cols = [c for c in self._get_feature_cols_m1() if c in df.columns]
            df["good_day"] = (df["sp500_return"].abs() > 0.5).astype(int)

            X = df[feature_cols].shift(1)
            y = df["good_day"].values
            valid = X.notna().all(axis=1)
            X = X.loc[valid]
            y = y[valid.values]

            tscv = TimeSeriesSplit(n_splits=5)
            accs = []
            for train_idx, test_idx in tscv.split(X):
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=42,
                )
                model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
                preds = model.predict(X.iloc[test_idx])
                acc = accuracy_score(y[test_idx], preds)
                accs.append(acc)

            avg_acc = np.mean(accs)
            logger.info("Model 1 walk-forward accuracy: %.2f%% (%s)", avg_acc * 100, [f"{a:.2f}" for a in accs])

            final_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
            )
            final_model.fit(X, y, verbose=False)

            model_path = os.path.join(MODELS_DIR, "model1_day_quality.pkl")
            with open(model_path, "wb") as f:
                pickle.dump({"model": final_model, "features": feature_cols}, f)
            logger.info("Model 1 saved -> %s", model_path)

            try:
                if shap is None:
                    raise ImportError("shap not installed")
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(X.iloc[:200])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                mean_abs = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(mean_abs)[::-1][:5]
                for i in top_idx:
                    logger.info("  SHAP Model1: %s = %.4f", feature_cols[i], mean_abs[i])
            except Exception as se:
                logger.warning("SHAP for Model 1 failed: %s", se)

            print(f"[Model 1] Day Quality — Avg accuracy: {avg_acc:.2%}")

        except Exception as exc:
            logger.error("train_day_quality_model failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "model1_train", str(exc))

    # ══════════════════════════════════════════════════════════
    # MODEL 2 — SESSION BIAS (LightGBM binary)
    # ══════════════════════════════════════════════════════════

    def _get_feature_cols_m2(self) -> List[str]:
        return self._get_feature_cols_m1()

    def train_session_bias_model(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        LightGBM binary classifier: predict next-day direction (1=up, 0=down).
        Walk-forward validation.
        """
        try:
            if df is None:
                path = os.path.join(DATA_FEAT_DIR, "historical_features.parquet")
                df = pd.read_parquet(path) if os.path.exists(path) else self.fetch_historical_data()
            if df.empty:
                logger.warning("No data for Model 2")
                return

            feature_cols = [c for c in self._get_feature_cols_m2() if c in df.columns]
            y_raw = df["sp500_direction"].values
            y = np.where(y_raw == 1, 1, 0)
            X = df[feature_cols].shift(1)
            valid = X.notna().all(axis=1)
            X = X.loc[valid]
            y = y[valid.values]

            tscv = TimeSeriesSplit(n_splits=5)
            accs = []
            for train_idx, test_idx in tscv.split(X):
                model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
                )
                model.fit(X.iloc[train_idx], y[train_idx])
                preds = model.predict(X.iloc[test_idx])
                acc = accuracy_score(y[test_idx], preds)
                accs.append(acc)

            avg_acc = np.mean(accs)
            logger.info("Model 2 walk-forward accuracy: %.2f%% (%s)", avg_acc * 100, [f"{a:.2f}" for a in accs])

            final_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
            )
            final_model.fit(X, y)

            model_path = os.path.join(MODELS_DIR, "model2_session_bias.pkl")
            with open(model_path, "wb") as f:
                pickle.dump({"model": final_model, "features": feature_cols}, f)
            logger.info("Model 2 saved -> %s", model_path)

            print(f"[Model 2] Session Bias — Avg accuracy: {avg_acc:.2%}")

        except Exception as exc:
            logger.error("train_session_bias_model failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "model2_train", str(exc))

    # ══════════════════════════════════════════════════════════
    # PHASE 2 — META-LABEL MODEL (XGBoost multiclass on live signals)
    # ══════════════════════════════════════════════════════════

    CHECKLIST_FEATURE_MAP = {
        "direction": {"SHORT": -1, "LONG": 1},
        "macro_bias": {"LONG": 1, "SHORT": -1, "MIXED": 0},
        "session": {"Asia": 0, "London": 1, "NY_Open_Killzone": 2, "NY_Session": 3, "Off_Hours": 4},
        "vix_bucket": {"low": 0, "normal": 1, "elevated": 2, "high": 3, "extreme": 4},
        "vix_direction_bias": {"BUY_BIAS": 1, "SELL_BIAS": -1, "NEUTRAL": 0},
        "us10y_direction": {"falling": -1, "flat": 0, "rising": 1},
        "oil_direction": {"falling": -1, "stable": 0, "spiking": 1},
        "dxy_direction": {"falling": -1, "flat": 0, "rising": 1},
        "rut_direction": {"red": -1, "neutral": 0, "green": 1},
        "structure_bias": {"long": 1, "short": -1, "unclear": 0},
        "h4_bos": {"bullish": 1, "bearish": -1},
        "wyckoff": {"markup": 1, "markdown": -1, "unclear": 0},
        "delta_direction": {"buyers": 1, "sellers": -1, "mixed": 0},
        "divergence": {"bullish": 1, "bearish": -1, "none": 0},
        "zone_type": {"round": 1, "pdh": 2, "pdl": 3, "poc": 4, "eqh": 5, "eql": 6, "asian_high": 7, "asian_low": 8},
        "zone_direction": {"support": 1, "resistance": -1, "at_level": 0},
        "groq_sentiment": {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0},
        "grade": {"A": 2, "B": 1},
        "tp_source": {"eqh": 5, "eql": 5, "pdh": 4, "pdl": 4, "round": 3, "poc": 2, "atr": 1},
        "entry_confidence": {"full": 4, "high": 3, "base": 2, "no_sweep": 1},
        "direction_confidence": {"high": 3, "medium": 2, "reduced": 1},
        "range_strength": {"strong": 2, "mild": 1, "none": 0},
        "sweep_side": {"buy_side": -1, "sell_side": 1},
        "sweep_level_type": {"pdh": 5, "pdl": 5, "eqh": 4, "eql": 4, "asian_high": 3, "asian_low": 3, "round": 1},
        "sl_method": {"range": 2, "swing": 1, "fvg": 1, "atr": 0},
    }

    META_FEATURE_COLS = [
        # --- macro / VIX ---
        "vix_level", "vix_pct", "bullish_count", "bearish_count",
        # --- H4 structure ---
        "above_ema200", "above_ema50", "choch_recent",
        "ob_bullish_nearby", "ob_bearish_nearby",
        # --- zones ---
        "at_zone", "zone_distance", "eqh_nearby", "eql_nearby",
        # --- orderflow ---
        "confirms_bias", "vix_spiking_now",
        # --- entry ---
        "fvg_present", "m5_bos", "ustec_agrees", "rr", "atr",
        "sl_points", "tp_points",
        "displacement_valid", "has_liquidity_sweep",
        "fvg_age_bars", "fvg_size_points",
        # --- range / consolidation ---
        "is_ranging", "adx_value", "atr_compressing",
        "range_size_points", "price_in_range_pct",
        # --- AMD / Asian session ---
        "asian_range_size",
        "sweep_bars_ago",
        # --- time / context ---
        "score", "is_monday", "is_friday", "is_news_day",
        "hour_utc", "data_version",
        # --- encoded categoricals ---
        "direction_enc", "macro_bias_enc", "session_enc", "vix_bucket_enc",
        "vix_direction_bias_enc", "us10y_direction_enc", "oil_direction_enc",
        "dxy_direction_enc", "rut_direction_enc", "structure_bias_enc",
        "h4_bos_enc", "wyckoff_enc", "delta_direction_enc", "divergence_enc",
        "zone_type_enc", "zone_direction_enc", "groq_sentiment_enc", "grade_enc",
        "tp_source_enc", "entry_confidence_enc", "direction_confidence_enc",
        "range_strength_enc", "sweep_side_enc", "sweep_level_type_enc", "sl_method_enc",
    ]

    def prepare_checklist_features(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Convert all categorical checklist fields to numeric for ML consumption.

        NaN-aware: XGBoost and LightGBM handle NaN natively, so old signals
        with NULL values in new columns (data_version=1) work without imputation.
        Only boolean columns are filled with 0 (False) when missing.
        """
        try:
            df = signals_df.copy()

            for col, mapping in self.CHECKLIST_FEATURE_MAP.items():
                enc_col = f"{col}_enc"
                if col in df.columns:
                    df[enc_col] = df[col].map(mapping).fillna(0).astype(float)
                else:
                    df[enc_col] = 0.0

            bool_cols = [
                "above_ema200", "above_ema50", "choch_recent", "at_zone",
                "confirms_bias", "vix_spiking_now", "fvg_present", "m5_bos",
                "ustec_agrees", "is_monday", "is_friday", "is_news_day",
                "ob_bullish_nearby", "ob_bearish_nearby", "eqh_nearby", "eql_nearby",
                "displacement_valid", "has_liquidity_sweep",
                "is_ranging", "atr_compressing",
            ]
            for bc in bool_cols:
                if bc in df.columns:
                    df[bc] = df[bc].fillna(0).astype(float)

            numeric_cols = [
                "vix_level", "vix_pct", "bullish_count", "bearish_count",
                "zone_distance", "rr", "atr", "score",
                "sl_points", "tp_points", "tp1_points",
                "adx_value", "fvg_age_bars", "fvg_size_points",
                "range_size_points", "price_in_range_pct",
                "asian_range_size", "sweep_bars_ago",
                "hour_utc", "data_version",
            ]
            for nc in numeric_cols:
                if nc in df.columns:
                    df[nc] = pd.to_numeric(df[nc], errors="coerce")

            available = [c for c in self.META_FEATURE_COLS if c in df.columns]
            missing = [c for c in self.META_FEATURE_COLS if c not in df.columns]
            if missing:
                logger.warning("Meta features missing from data: %s", missing[:10])
            return df[available]

        except Exception as exc:
            logger.error("prepare_checklist_features failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    def train_meta_label_model(self) -> bool:
        """
        XGBoost multiclass on real signal outcomes.
        Requires at least MODEL_RETRAIN_SIGNAL_THRESHOLD labelled signals.
        """
        try:
            if self._signal_logger is None:
                logger.warning("No signal logger — cannot train Model 3")
                return False

            data = self._signal_logger.get_training_data()
            data = data[data["outcome_label"].notna()].copy()
            data = data[~data["outcome"].str.contains("TIMEOUT", na=True)].copy()
            count = len(data)

            if count < MODEL_RETRAIN_SIGNAL_THRESHOLD:
                needed = MODEL_RETRAIN_SIGNAL_THRESHOLD - count
                msg = f"Meta model needs {needed} more signals (have {count})"
                logger.info(msg)
                self._send_system_alert("INFO", "model3_train", msg)
                print(f"[Model 3] {msg}")
                return False

            X_df = self.prepare_checklist_features(data)
            if X_df.empty:
                logger.warning("Feature prep returned empty")
                return False

            # PARTIAL_WIN (label=0, outcome='PARTIAL_WIN') = direction correct = treat as win
            # LOSS (label=-1) = direction wrong = 0
            # WIN (label=1) = full target hit = 1
            y_raw = data["outcome_label"].values
            outcomes = data["outcome"].values
            label_map = {-1: 0, 1: 1}
            y = np.array([
                1 if "PARTIAL_WIN" in str(outcomes[i]) else label_map.get(int(v), 0)
                for i, v in enumerate(y_raw)
            ])
            X = X_df
            feature_cols = list(X_df.columns)

            tscv = TimeSeriesSplit(n_splits=min(5, max(2, count // 50)))
            accs = []
            for train_idx, test_idx in tscv.split(X):
                model = xgb.XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="binary:logistic", eval_metric="logloss",
                    random_state=42,
                )
                model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
                preds = model.predict(X.iloc[test_idx])
                acc = accuracy_score(y[test_idx], preds)
                accs.append(acc)

            avg_acc = np.mean(accs)
            logger.info("Model 3 walk-forward accuracy: %.2f%%", avg_acc * 100)

            final_model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42,
            )
            final_model.fit(X, y, verbose=False)

            model_path = os.path.join(MODELS_DIR, "model3_meta_label.pkl")
            with open(model_path, "wb") as f:
                pickle.dump({"model": final_model, "features": feature_cols}, f)

            try:
                if shap is None:
                    raise ImportError("shap not installed")
                explainer = shap.TreeExplainer(final_model)
                shap_path = os.path.join(MODELS_DIR, "model3_shap.pkl")
                with open(shap_path, "wb") as f:
                    pickle.dump(explainer, f)
            except Exception as se:
                logger.warning("SHAP explainer save failed: %s", se)

            features_path = os.path.join(MODELS_DIR, "model3_features.json")
            with open(features_path, "w") as f:
                json.dump(feature_cols, f)

            self._save_retrain_timestamp()

            msg = f"Model 3 retrained — accuracy: {avg_acc:.2%}, signals: {count}"
            logger.info(msg)
            self._send_system_alert("INFO", "model3_train", msg)
            print(f"[Model 3] Meta-Label — Avg accuracy: {avg_acc:.2%}")
            return True

        except Exception as exc:
            logger.error("train_meta_label_model failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "model3_train", str(exc))
            return False

    # ─── SHAP report ─────────────────────────────────────────

    def get_shap_report(self) -> Optional[List[str]]:
        """Load model3 + SHAP explainer and return top 5 feature importances."""
        try:
            model_path = os.path.join(MODELS_DIR, "model3_meta_label.pkl")
            shap_path = os.path.join(MODELS_DIR, "model3_shap.pkl")

            if not os.path.exists(model_path) or not os.path.exists(shap_path):
                return None

            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            with open(shap_path, "rb") as f:
                explainer = pickle.load(f)

            features = bundle.get("features", [])
            if not features:
                return None

            if self._signal_logger:
                data = self._signal_logger.get_training_data()
                if not data.empty:
                    # Exclude TIMEOUT outcomes for consistent SHAP analysis
                    data = data[~data["outcome"].str.contains("TIMEOUT", na=True)].copy()
                    X_df = self.prepare_checklist_features(data)
                    X = X_df.iloc[:100]
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[-1]
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    top_idx = np.argsort(mean_abs)[::-1][:5]
                    return [f"{features[i]} ({mean_abs[i]:.3f})" for i in top_idx if i < len(features)]
            return None

        except Exception as exc:
            logger.error("get_shap_report failed: %s", exc, exc_info=True)
            return None

    # ─── auto-retrain scheduler ──────────────────────────────

    def check_and_retrain(self) -> None:
        """Called every Sunday — retrain Model 3 if enough new signals, Models 1/2 monthly."""
        try:
            last = self._load_retrain_timestamp()

            if self._signal_logger:
                count = self._signal_logger.get_signal_count()
                last_count = last.get("signal_count", 0)
                new_signals = count - last_count

                if new_signals >= MODEL_INCREMENTAL_THRESHOLD:
                    self._send_system_alert(
                        "INFO", "retrain",
                        f"Model 3 retrain starting — {new_signals} new signals ({count} total)",
                    )
                    logger.info("Retraining Model 3: %d new signals", new_signals)
                    success = self.train_meta_label_model()
                    if success:
                        self._send_system_alert(
                            "INFO", "retrain",
                            f"Model 3 retrain complete — {count} signals used",
                        )
                else:
                    needed = MODEL_INCREMENTAL_THRESHOLD - new_signals
                    msg = (
                        f"Model 3 retrain skipped — only {new_signals} new signals "
                        f"(need {MODEL_INCREMENTAL_THRESHOLD}). "
                        f"Collect {needed} more to trigger retrain."
                    )
                    logger.info(msg)
                    self._send_system_alert("INFO", "retrain", msg)

            last_m1_date = last.get("model1_date", "")
            try:
                days_since_m1 = (datetime.now(timezone.utc) - datetime.fromisoformat(last_m1_date)).days if last_m1_date else 999
            except Exception:
                days_since_m1 = 999

            if days_since_m1 > 30:
                self._send_system_alert(
                    "INFO", "retrain",
                    f"Monthly retrain starting for Models 1 & 2 ({days_since_m1} days since last)",
                )
                logger.info("Monthly retrain for Models 1 & 2")
                df = self.fetch_historical_data()
                if not df.empty:
                    self.train_day_quality_model(df)
                    self.train_session_bias_model(df)
                    self._save_retrain_timestamp(model1=True)
                    self._send_system_alert(
                        "INFO", "retrain",
                        "Models 1 & 2 retrain complete — historical data refreshed",
                    )

        except Exception as exc:
            logger.error("check_and_retrain failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "retrain", str(exc))

    # ─── retrain timestamp persistence ───────────────────────

    def _save_retrain_timestamp(self, model1: bool = False) -> None:
        path = os.path.join(MODELS_DIR, "last_retrain.json")
        try:
            existing = self._load_retrain_timestamp()
            existing["last_retrain"] = datetime.now(timezone.utc).isoformat()
            if self._signal_logger:
                existing["signal_count"] = self._signal_logger.get_signal_count()
            if model1:
                existing["model1_date"] = datetime.now(timezone.utc).isoformat()
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as exc:
            logger.error("_save_retrain_timestamp failed: %s", exc, exc_info=True)

    def _load_retrain_timestamp(self) -> Dict[str, Any]:
        path = os.path.join(MODELS_DIR, "last_retrain.json")
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    # ─── helpers ─────────────────────────────────────────────

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
