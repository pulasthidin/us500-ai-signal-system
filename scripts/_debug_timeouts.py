"""Debug TIMEOUT signals — find the root cause."""
import sqlite3
from datetime import datetime, timezone

DB_PATH = r"c:\Projects\US500\database\signals.db"
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# All timeouts
rows = conn.execute("""
    SELECT id, timestamp, direction, entry_price, sl_price, tp_price,
           sl_points, tp_points, rr, atr, bars_to_outcome, outcome,
           outcome_timestamp, session, score, grade
    FROM signals
    WHERE outcome LIKE '%TIMEOUT%'
    ORDER BY timestamp ASC
""").fetchall()

print(f"Total TIMEOUT signals: {len(rows)}\n")
print(f"{'ID':>4} | {'Timestamp':16} | {'Dir':>5} | {'Entry':>7} | {'SL':>7} | {'TP':>7} | "
      f"{'SL pts':>6} | {'TP pts':>6} | {'RR':>4} | {'Bars':>4} | {'Session':>18} | Outcome Time")
print("-" * 140)

for r in rows:
    entry_ts = r['timestamp'][:16] if r['timestamp'] else '?'
    outcome_ts = r['outcome_timestamp'][:16] if r['outcome_timestamp'] else '?'
    print(f"#{r['id']:>3} | {entry_ts} | {r['direction']:>5} | "
          f"{r['entry_price'] or 0:>7.1f} | {r['sl_price'] or 0:>7.1f} | {r['tp_price'] or 0:>7.1f} | "
          f"{r['sl_points'] or 0:>6.1f} | {r['tp_points'] or 0:>6.1f} | {r['rr'] or 0:>4.1f} | "
          f"{r['bars_to_outcome'] or 0:>4} | {r['session'] or '?':>18} | {outcome_ts}")

# Check: how many have bars_to_outcome = 0?
zero_bars = [r for r in rows if (r['bars_to_outcome'] or 0) == 0]
some_bars = [r for r in rows if (r['bars_to_outcome'] or 0) > 0]
print(f"\nBars breakdown:")
print(f"  bars_to_outcome = 0:  {len(zero_bars)} (no bars found after entry → data gap or timestamp issue)")
print(f"  bars_to_outcome > 0:  {len(some_bars)} (price never hit SL or TP within available bars)")

if some_bars:
    bars_counts = [r['bars_to_outcome'] for r in some_bars]
    print(f"  Max bars checked: {max(bars_counts)}")
    print(f"  Min bars checked: {min(bars_counts)}")
    print(f"  Avg bars checked: {sum(bars_counts)/len(bars_counts):.0f}")

# Check the time gap between entry and outcome resolution
print(f"\nTime gap analysis:")
for r in rows[:5]:
    if r['timestamp'] and r['outcome_timestamp']:
        try:
            entry_dt = datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00'))
            outcome_dt = datetime.fromisoformat(r['outcome_timestamp'].replace('Z', '+00:00'))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            if outcome_dt.tzinfo is None:
                outcome_dt = outcome_dt.replace(tzinfo=timezone.utc)
            gap = (outcome_dt - entry_dt).total_seconds() / 3600
            print(f"  #{r['id']}: entry={r['timestamp'][:19]} → resolved={r['outcome_timestamp'][:19]} = {gap:.1f}h gap, bars={r['bars_to_outcome']}")
        except Exception as e:
            print(f"  #{r['id']}: parse error: {e}")

# Check: for 0-bar timeouts, what's the M5 bar coverage?
# 500 M5 bars = 500 * 5min = 2500min = ~41.7 hours
print(f"\nM5 coverage: 500 bars = ~41.7 hours back from 'now'")
print(f"Signals older than ~42h at check time cannot be resolved via M5.")

# Check: which signals were resolved by catchup vs scheduled
print(f"\nLikely cause for bars_to_outcome=0:")
print(f"  - Signal was too old when checked (>42h) and M5 bars didn't cover it")
print(f"  - H1 fallback (200 bars = ~33 days) should have caught it")
print(f"  - If H1 also returned None → cTrader connection issue at check time")

# Check if there are recent timeouts that SHOULD have had data
wed = "2026-03-25T00:00:00"
recent_timeouts = conn.execute("""
    SELECT id, timestamp, entry_price, sl_price, tp_price, direction, bars_to_outcome
    FROM signals
    WHERE outcome LIKE '%TIMEOUT%' AND timestamp >= ?
    ORDER BY timestamp ASC
""", (wed,)).fetchall()

print(f"\nRecent timeouts (since Wed): {len(recent_timeouts)}")
for r in recent_timeouts:
    print(f"  #{r['id']} | {r['timestamp'][:16]} | {r['direction']} | "
          f"entry={r['entry_price']:.1f} SL={r['sl_price']:.1f} TP={r['tp_price']:.1f} | "
          f"bars_checked={r['bars_to_outcome']}")

conn.close()
