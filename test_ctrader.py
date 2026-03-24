"""Test all timeframes on live cTrader."""
import os
os.makedirs("logs", exist_ok=True)
from dotenv import load_dotenv
load_dotenv()

from src.ctrader_connection import CTraderConnection

print("Connecting...")
ct = CTraderConnection()
ct.connect()
ct.authenticate_app()
ct.authenticate_account()

us500 = ct.get_symbol_id("US500")
print(f"US500 ID: {us500}\n")

for tf, count in [("M1", 5), ("M5", 5), ("M15", 5), ("H1", 5), ("H4", 5)]:
    print(f"--- {tf} ({count} bars) ---")
    bars = ct.fetch_bars(us500, tf, count)
    if bars is not None and not bars.empty:
        print(f"  {len(bars)} bars returned")
        print(f"  First: {bars['timestamp'].iloc[0]}  Close: {bars['close'].iloc[0]}")
        print(f"  Last:  {bars['timestamp'].iloc[-1]}  Close: {bars['close'].iloc[-1]}")
    else:
        print("  No bars returned")
    print()

print("DONE")
