"""
Manual retrain for all 3 models.
Safe to run while the app is running — SQLite handles concurrent reads.
App needs restart after to load new models.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.signal_logger import SignalLogger
from src.model_trainer import ModelTrainer

MODELS_DIR = "models"


def main():
    sl = SignalLogger()
    sl.init_db()
    trainer = ModelTrainer(signal_logger=sl)

    count = sl.get_signal_count()
    print(f"Labelled signals in DB: {count}")
    print()

    # ── Model 1: Day Quality ──────────────────────────────────
    print("[M1] Fetching historical data from Yahoo Finance...")
    try:
        df = trainer.fetch_historical_data()
        if df.empty:
            print("[M1] ERROR: No historical data returned. Skipping M1 & M2.")
        else:
            print(f"[M1] Got {len(df)} rows")
            print("[M1] Training Day Quality model...")
            trainer.train_day_quality_model(df)
            print()

            # ── Model 2: Session Bias ─────────────────────────
            print("[M2] Training Session Bias model...")
            trainer.train_session_bias_model(df)
            print()
    except Exception as exc:
        print(f"[M1/M2] ERROR: {exc}")
        print()

    # ── Model 3: Meta-Label ───────────────────────────────────
    print(f"[M3] Training Meta-Label model ({count} signals)...")
    try:
        success = trainer.train_meta_label_model()
        if success:
            print("[M3] OK")
        else:
            print("[M3] Training returned False (need 200+ labelled signals)")
    except Exception as exc:
        print(f"[M3] ERROR: {exc}")

    # ── Verify files ──────────────────────────────────────────
    print()
    print("Model files:")
    all_ok = True
    for name in ["model1_day_quality.pkl", "model2_session_bias.pkl", "model3_meta_label.pkl"]:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {name}: OK ({size:,} bytes)")
        else:
            print(f"  {name}: MISSING")
            all_ok = False

    print()
    if all_ok:
        print("All 3 models saved. Restart the app to load them.")
    else:
        print("Some models missing — check errors above.")


if __name__ == "__main__":
    main()
