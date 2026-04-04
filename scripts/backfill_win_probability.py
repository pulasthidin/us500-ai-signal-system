"""
Backfill win_probability for all existing signals.

Steps:
  1. Retrain Model 3 on current signal data (ensures model exists and is fresh)
  2. Load the retrained model
  3. For each signal with outcome, compute win_probability and write it back

NOTE: These are in-sample predictions for signals the model trained on.
Future out-of-sample signals are the real test of model quality.
This backfill exists so we have a baseline for bucket analysis.

Usage:
  cd C:/Projects/US500
  python scripts/backfill_win_probability.py
"""

import json
import os
import pickle
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.signal_logger import SignalLogger
from src.model_trainer import ModelTrainer

DB_PATH = os.path.join("database", "signals.db")
MODELS_DIR = "models"


def main():
    signal_logger = SignalLogger()
    signal_logger.init_db()

    count = signal_logger.get_signal_count()
    print(f"Labelled signals in DB: {count}")

    # ── Step 1: Retrain Model 3 ──
    print("\n=== Step 1: Retraining Model 3 ===")
    trainer = ModelTrainer(signal_logger=signal_logger)
    success = trainer.train_meta_label_model()
    if not success:
        print("Model 3 training failed or not enough signals. Cannot backfill.")
        return

    # ── Step 2: Load retrained model ──
    print("\n=== Step 2: Loading retrained model ===")
    model_path = os.path.join(MODELS_DIR, "model3_meta_label.pkl")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["features"]
    print(f"Model loaded with {len(feature_cols)} features")

    # ── Step 3: Predict on all signals ──
    print("\n=== Step 3: Computing win_probability for all signals ===")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    all_signals = pd.read_sql_query("SELECT * FROM signals", conn)
    print(f"Total signals in DB: {len(all_signals)}")

    if all_signals.empty:
        print("No signals to backfill.")
        conn.close()
        return

    X_df = trainer.prepare_checklist_features(all_signals)
    if X_df.empty:
        print("Feature preparation returned empty DataFrame.")
        conn.close()
        return

    for col in feature_cols:
        if col not in X_df.columns:
            X_df[col] = float("nan")
    X_df = X_df[feature_cols]

    probas = model.predict_proba(X_df)
    win_probs = probas[:, 1] if probas.shape[1] == 2 else probas[:, -1]

    # ── Step 4: Write back to DB ──
    print(f"\n=== Step 4: Writing {len(win_probs)} predictions to DB ===")
    updated = 0
    cursor = conn.cursor()
    for i, (_, row) in enumerate(all_signals.iterrows()):
        signal_id = int(row["id"])
        wp = round(float(win_probs[i]), 3)
        cursor.execute(
            "UPDATE signals SET win_probability = ? WHERE id = ?",
            (wp, signal_id),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} signals with win_probability")

    # ── Summary stats ──
    print("\n=== Backfill Summary ===")
    wp_arr = np.array([round(float(w), 3) for w in win_probs])
    print(f"  Min:    {wp_arr.min():.3f}")
    print(f"  Max:    {wp_arr.max():.3f}")
    print(f"  Mean:   {wp_arr.mean():.3f}")
    print(f"  Median: {np.median(wp_arr):.3f}")

    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]:
        mask = (wp_arr >= lo) & (wp_arr < hi)
        n = mask.sum()
        label = f"[{lo:.1f}, {hi:.1f})"
        print(f"  {label}: {n} signals ({n/len(wp_arr)*100:.1f}%)")

    print("\nDone. Run the app normally — new signals will have win_probability saved automatically.")


if __name__ == "__main__":
    main()
