"""
One-time fix: Re-evaluate all TIMEOUT signals using cTrader historical data.

The outcome_tracker had a bug where it marked TIMEOUT after checking only
5-6 M5 bars (30 min of data). Many of those signals actually hit TP or SL
later — they just needed more time.

This script:
  1. Finds all TIMEOUT signals
  2. Fetches enough M5 bars from cTrader to cover each signal
  3. Re-runs the triple barrier check with full data
  4. Updates the database with correct WIN/LOSS outcomes
  5. Prints a summary of changes

Run this ONCE when cTrader is connected:
  python fix_timeouts.py
"""

import os
import sys
import sqlite3
from datetime import datetime, timezone, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

DB_PATH = os.path.join("database", "signals.db")
from config import OUTCOME_MIN_M5_BARS_FOR_TIMEOUT, OUTCOME_MIN_H1_BARS_FOR_TIMEOUT


def triple_barrier_check(signal, bars_df):
    """Replicates the fixed triple barrier logic."""
    direction = signal["direction"]
    sl_price = signal["sl_price"]
    tp_price = signal["tp_price"]
    entry_price = signal["entry_price"]

    if not all([direction, sl_price, tp_price, entry_price]):
        return None

    try:
        entry_time = pd.Timestamp(signal["timestamp"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
    except Exception:
        return None

    bars_df = bars_df.copy()
    ts_col = None
    for c in ("timestamp", "time", "datetime"):
        if c in bars_df.columns:
            ts_col = c
            break

    if ts_col:
        bars_df[ts_col] = pd.to_datetime(bars_df[ts_col], utc=True)
        bars_df.sort_values(ts_col, inplace=True)
        bars_df.reset_index(drop=True, inplace=True)
        bars_after = bars_df[bars_df[ts_col] >= entry_time]
    else:
        bars_after = bars_df

    if bars_after.empty:
        return None

    for bar_num, (_, bar) in enumerate(bars_after.iterrows(), start=1):
        if direction == "SHORT":
            sl_hit = bar["high"] >= sl_price
            tp_hit = bar["low"] <= tp_price
        elif direction == "LONG":
            sl_hit = bar["low"] <= sl_price
            tp_hit = bar["high"] >= tp_price
        else:
            continue

        if sl_hit and tp_hit:
            return {"outcome": "LOSS", "label": -1, "bars": bar_num,
                    "pnl": _calc_pnl(signal, "LOSS"),
                    "ts": str(bar.get(ts_col, "")) if ts_col else ""}
        if sl_hit:
            return {"outcome": "LOSS", "label": -1, "bars": bar_num,
                    "pnl": _calc_pnl(signal, "LOSS"),
                    "ts": str(bar.get(ts_col, "")) if ts_col else ""}
        if tp_hit:
            return {"outcome": "WIN", "label": 1, "bars": bar_num,
                    "pnl": _calc_pnl(signal, "WIN"),
                    "ts": str(bar.get(ts_col, "")) if ts_col else ""}

    if len(bars_after) >= OUTCOME_MIN_M5_BARS_FOR_TIMEOUT:
        return {"outcome": "TIMEOUT", "label": 0, "bars": len(bars_after),
                "pnl": 0.0, "ts": datetime.now(timezone.utc).isoformat()}
    return None


def _calc_pnl(signal, outcome):
    entry = signal.get("entry_price", 0)
    sl = signal.get("sl_price", 0)
    tp = signal.get("tp_price", 0)
    d = signal.get("direction", "")
    if outcome == "TIMEOUT":
        return 0.0
    if d == "SHORT":
        return round(entry - tp, 2) if outcome == "WIN" else round(entry - sl, 2)
    else:
        return round(tp - entry, 2) if outcome == "WIN" else round(sl - entry, 2)


def main():
    from dotenv import load_dotenv
    load_dotenv()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    timeouts = conn.execute("""
        SELECT * FROM signals
        WHERE outcome LIKE '%TIMEOUT%'
        ORDER BY timestamp ASC
    """).fetchall()

    print(f"Found {len(timeouts)} TIMEOUT signals to re-evaluate\n")

    if not timeouts:
        conn.close()
        return

    try:
        from src.ctrader_connection import CTraderConnection
        from src.alert_bot import AlertBot

        print("Connecting to cTrader...")
        ctrader = CTraderConnection(alert_bot=AlertBot())
        ctrader.connect()
        ctrader.authenticate_app()
        ctrader.authenticate_account()
        ctrader.refresh_token()
        us500_id = ctrader.get_symbol_id("US500")

        if not us500_id:
            print("ERROR: Could not find US500 symbol")
            conn.close()
            return

        print(f"Connected. US500 symbol ID: {us500_id}")
        print("Fetching M5 bars (500 bars = ~42 hours)...\n")

        bars_df = ctrader.fetch_bars(us500_id, "M5", 500)
        h1_df = ctrader.fetch_bars(us500_id, "H1", 500)

    except Exception as e:
        print(f"cTrader connection failed: {e}")
        print("Cannot re-evaluate without live data.")
        conn.close()
        return

    changes = {"win": 0, "loss": 0, "still_timeout": 0, "no_data": 0}

    for sig in timeouts:
        signal = dict(sig)
        sid = signal["id"]

        result = triple_barrier_check(signal, bars_df)

        if result is None and h1_df is not None:
            result = triple_barrier_check(signal, h1_df)
            if result and result["outcome"] != "TIMEOUT":
                result["outcome"] = f"ESTIMATED_{result['outcome']}"

        if result is None:
            print(f"  #{sid}: No data coverage — keeping TIMEOUT")
            changes["no_data"] += 1
            continue

        old_outcome = signal["outcome"]
        new_outcome = result["outcome"]
        new_pnl = result["pnl"]

        if "WIN" in new_outcome:
            changes["win"] += 1
            marker = " *** FIXED → WIN ***"
        elif "LOSS" in new_outcome:
            changes["loss"] += 1
            marker = " fixed → LOSS"
        else:
            changes["still_timeout"] += 1
            marker = " (confirmed TIMEOUT)"

        print(f"  #{sid}: {old_outcome} → {new_outcome} | pnl={new_pnl:+.1f} | bars={result['bars']}{marker}")

        if new_outcome != old_outcome:
            conn.execute("""
                UPDATE signals
                SET outcome = ?, outcome_label = ?, bars_to_outcome = ?,
                    pnl_points = ?, outcome_timestamp = ?
                WHERE id = ?
            """, (new_outcome, result["label"], result["bars"],
                  new_pnl, result["ts"], sid))

    conn.commit()
    conn.close()

    print(f"\n{'='*60}")
    print(f"Re-evaluation complete:")
    print(f"  Fixed to WIN:    {changes['win']}")
    print(f"  Fixed to LOSS:   {changes['loss']}")
    print(f"  Confirmed TIMEOUT: {changes['still_timeout']}")
    print(f"  No data:         {changes['no_data']}")
    total_fixed = changes['win'] + changes['loss']
    print(f"  Total fixed:     {total_fixed} of {len(timeouts)}")
    print(f"{'='*60}")

    if changes['win'] > 0:
        conn2 = sqlite3.connect(DB_PATH)
        row = conn2.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome IN ('WIN','ESTIMATED_WIN') THEN 1 ELSE 0 END) as wins
            FROM signals WHERE outcome IS NOT NULL
        """).fetchone()
        conn2.close()
        if row[0] > 0:
            print(f"\n  Updated win rate: {row[1]}/{row[0]} = {row[1]/row[0]*100:.1f}%")


if __name__ == "__main__":
    main()
