# US500 AI Signal System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Quality](https://github.com/pulasthidin/us500-ai-signal-system/actions/workflows/quality.yml/badge.svg)](https://github.com/pulasthidin/us500-ai-signal-system/actions)
[![Tests](https://img.shields.io/badge/tests-443%20passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6.svg)]()

A fully automated intraday trading signal system for US500 (S&P 500) built on ICT/SMC confluence analysis, market regime detection, liquidity sweep confirmation, and self-evolving ML models. Telegram-only alerts with TP1/TP2 partial profit targets.

---

## Architecture

```
                    +------------------+
                    |    Telegram       |
                    |  Trade + System   |
                    +--------+---------+
                             |
                    +--------+---------+
                    |     live.py       |
                    |   Main Loop 60s   |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
  +-------+------+  +-------+------+  +-------+-------+
  | Checklist    |  | Outcome      |  | Health        |
  | Engine       |  | Tracker      |  | Monitor       |
  +-------+------+  | (Quad Barrier|  +---------------+
          |          |  TP1/TP2)    |
  +-------+-------+-+-----+-------+
  |       |       |       |       |
  L1      L2      L3      L4      M5
  Macro   H4+H1   H1      M15     Entry
  VIX     EMA     Zones   Delta   FVG+Disp
  DXY     BOS     PDH/L   Diverg  BOS
  US10Y   ChoCH   POC     VIX     SMT
  OIL     Wyckoff Asian   Spike   Sweep
  RUT     Range   EQH/L           TP1/TP2
          ADX     Sweep           R:R
```

## The 4-Layer Confluence System

Each signal must pass through 4 independent layers before firing. Only setups scoring 3/4 or 4/4 generate alerts.

| Layer | Timeframe | What It Checks | Data Source |
|-------|-----------|----------------|-------------|
| **L1 Macro** | Daily | VIX level/direction, US10Y, Oil, DXY, RUT | yfinance |
| **L2 Structure** | H4 + H1 | EMA 200/50, BOS/ChoCH, Wyckoff, **range detection (ADX + ATR compression)** | cTrader + smartmoneyconcepts |
| **L3 Zones** | H1 | PDH/PDL, round numbers, POC, EQH/EQL, **Asian session range, liquidity sweep detection** | cTrader |
| **L4 Order Flow** | M15 | Volume delta (M1 intrabar), delta divergence, VIX spike gate | cTrader |
| **M5 Entry** | M5 | **Displacement-validated FVG**, BOS, USTEC SMT, **sweep confirmation**, **TP1/TP2**, R:R >= 1.6 | cTrader + smartmoneyconcepts |

## Quality Filters (ICT/SMC)

These filters prevent entering low-probability setups:

| Filter | What It Does | When Active |
|--------|-------------|-------------|
| **Range Detection** | ADX < 20 + ATR compression = market is consolidating | Always checked |
| **Liquidity Sweep** | Requires price to sweep a key level (PDH/PDL/EQH/EQL/Asian H-L) before entry | Required in ranging, bonus in trending |
| **Displacement Validation** | FVG impulse candle must have body > 50% of range AND range > 0.6x ATR | Always (filters noise FVGs) |
| **Asian Session Range** | Tracks 00:00-09:00 UTC range for AMD (Accumulation-Manipulation-Distribution) model | Always tracked |
| **Range-aware SL** | Uses H1 range boundaries for SL in ranging markets (capped at 3x ATR) | When ranging |

### How Filters Interact

```
Market Trending (ADX > 20)?
  YES --> Displacement validated FVG required
          Sweep = confidence bonus (A vs B grade)
          Swing SL (5-bar M5)

  NO (Ranging) --> Displacement validated FVG required
                   Sweep REQUIRED or entry blocked
                   Range boundary SL (if < 3x ATR)
```

### Scoring & Decisions

| Score | Entry Ready | Decision | Action |
|-------|-------------|----------|--------|
| 4/4 | Yes + sweep | **FULL SEND** (Grade A) | Full VIX-sized position |
| 4/4 | Yes, no sweep | **HALF SIZE** (Grade B) | Downgraded, still fires |
| 3/4 | Yes | **HALF SIZE** (Grade B) | Half position |
| 3/4 | No | **WAIT** | Heads-up alert only |
| 0-2/4 | -- | **NO TRADE** | Silent |
| VIX > 30 | -- | **HARD STOP** | All signals blocked |

### Signal Format (Telegram)

```
SHORT -- HALF SIZE
━━━━━━━━━━━━━━━━━━━━
Entry:  6,574.00
SL:     6,582.00  (+8.0 pts)
TP1:    6,566.00  (-8.0 pts) = 1R partial
TP2:    6,550.00  (-24.0 pts) = full target
R:R:    1:2.50
━━━━━━━━━━━━━━━━━━━━
Score:  4/4
Size:   HALF (VIX elevated)
━━━━━━━━━━━━━━━━━━━━
L1: SHORT bias | VIX +2.5% rising
L2: Below EMA50 | BOS bearish
L3: At PDL 6,567 | 3.0pts away
L4: Sellers | Divergence: bearish
━━━━━━━━━━━━━━━━━━━━
SWEEP: sell_side PDL confirmed
RANGE: Trending (ADX 22.4)
14:43 SL | Thursday
```

## TP1 Partial Profit System

The system provides two take-profit levels:

| Level | Distance | Purpose |
|-------|----------|---------|
| **TP1** | 1R (equal to SL distance) | Take partial profit, move SL to breakeven |
| **TP2** | Structure level or ATR x 2.5 | Full target for remaining position |

### Outcome Tracking (Quad Barrier)

```
Signal created --> outcome = NULL
    |
Phase 1: Check TP1, TP2, SL
    |-- SL hit           --> LOSS (-1)              [terminal]
    |-- TP2 hit directly --> WIN (+1)               [terminal]
    |-- TP1 hit          --> PARTIAL_WIN (0)        [continues]
    |
Phase 2: Check TP2 vs breakeven (entry price)
    |-- TP2 hit          --> upgrade to WIN (+1)    [terminal]
    |-- Breakeven hit    --> PARTIAL_WIN_FINAL (0)  [terminal]
    |-- Timeout          --> PARTIAL_WIN_FINAL (0)  [terminal]
```

## Self-Evolving ML System

The system automatically evolves through 4 stages as it collects trading data:

| Stage | Signals | What Activates | Benefit |
|-------|---------|----------------|---------|
| **0** | < 200 | Nothing -- collecting data | Rule-based only |
| **1** | 200+ | Model 3 (meta-label) trains | Win probability on alerts |
| **2** | 500+ | Pattern scanner activates | Early warning alerts |
| **3** | 1000+ | Daily retraining at midnight | Maximum accuracy |

### Three ML Models

| Model | Type | Target | Purpose |
|-------|------|--------|---------|
| **Model 1** | XGBoost | Good trade day? | Day quality filter |
| **Model 2** | LightGBM | Direction up/down | Session bias prediction |
| **Model 3** | XGBoost | WIN/PARTIAL_WIN/LOSS | Signal win probability |

### 64 ML Training Features

Every signal logs 64 features including:
- Macro: VIX level/bucket, DXY, US10Y, Oil, RUT
- Structure: EMA position, BOS direction, Wyckoff phase
- Zones: distance to zone, zone type, EQH/EQL nearby
- Order flow: delta direction, divergence, VIX spike
- Entry: FVG present, displacement valid, R:R, ATR, SL/TP distances
- **Range: ADX value, is_ranging, ATR compression, range size, price position in range**
- **Sweep: has_liquidity_sweep, sweep side, sweep level type, bars ago**
- **Session: Asian range size, hour_utc, data_version**
- Confidence: entry/direction confidence, grade, tp_source

All models use walk-forward validation (no data leakage) and SHAP explainability.

## VIX-Based Position Sizing

| VIX Level | Size | Restriction |
|-----------|------|-------------|
| < 15 | Full | -- |
| 15-20 | Normal | -- |
| 20-25 | Half | -- |
| 25-30 | Quarter | Short only |
| > 30 | **NO TRADE** | Hard stop |

## Reliability Features

- **Crash Recovery**: Catches up ALL missed outcomes on restart (even weeks old)
- **Minimum Bars Check**: Won't mark TIMEOUT until 72+ M5 bars (6 hours) checked
- **Atomic DB Writes**: SQLite transactions -- full row or nothing
- **Signal Save Order**: Database FIRST, Telegram SECOND (never lose a signal)
- **Checkpoint File**: Resumes interrupted outcome checks after crash
- **Heartbeat**: Saves every 5 minutes, calculates offline duration on restart
- **10-Point Self-Test**: Verifies all dependencies before starting main loop
- **Exponential Reconnect**: cTrader reconnects with [5,10,20,40,80]s backoff
- **Health Monitor**: 6-component checks every 15 min, alerts on state change only
- **Duplicate Prevention**: Same signal not fired twice within 5 minutes
- **News Filter**: Blocks signals 30 min before high-impact USD events
- **DB Migration**: Auto-adds new columns on startup, preserves old data

## Quick Start

```bash
# 1. Clone
git clone https://github.com/pulasthidin/us500-ai-signal-system.git
cd us500-ai-signal-system

# 2. First-time setup (Windows)
install.bat

# 3. Configure API keys
# Edit .env with your real keys (see .env.example for guide)

# 4. Run
run.bat
```

### Manual Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
# Edit .env with your keys
python live.py
```

## Project Structure

```
us500-ai-signal-system/
├── live.py                      # Main entry point + startup self-tests
├── config.py                    # All constants and thresholds
├── .env                         # API keys (not in repo)
├── requirements.txt             # Python dependencies
├── fix_timeouts.py              # One-time script to re-evaluate TIMEOUT signals
│
├── src/
│   ├── ctrader_connection.py    # cTrader Open API + reconnect
│   ├── macro_checker.py         # Layer 1: VIX, DXY, US10Y, Oil, RUT
│   ├── structure_analyzer.py    # Layer 2: EMA, BOS, ChoCH, Wyckoff, range detection
│   ├── zone_calculator.py       # Layer 3: PDH/PDL, POC, Asian range, liquidity sweeps
│   ├── orderflow_analyzer.py    # Layer 4: volume delta, divergence
│   ├── entry_checker.py         # M5: FVG + displacement, sweep, TP1/TP2, R:R
│   ├── checklist_engine.py      # 4-layer orchestrator + range/sweep filters
│   ├── signal_logger.py         # SQLite database + 64 ML features + crash recovery
│   ├── outcome_tracker.py       # Quad barrier (TP1/TP2) + catch-up
│   ├── calendar_checker.py      # News calendar from Forex Factory (free)
│   ├── alert_bot.py             # Two Telegram bots (trade + system)
│   ├── health_monitor.py        # 6-component health checks
│   ├── model_trainer.py         # XGBoost, LightGBM, SHAP, 64 features
│   ├── model_predictor.py       # Real-time ML predictions
│   ├── evolution_manager.py     # 4-stage auto-evolution
│   └── pattern_scanner.py       # Early warning pattern matching
│
├── tests/                       # 443 unit tests (18 test files)
├── notebooks/
│   └── 01_data_fetcher.ipynb    # Data download + feature engineering
│
├── run.bat                      # Double-click to start
├── stop.bat                     # Double-click to stop
├── restart.bat                  # Double-click to restart
├── install.bat                  # First-time setup
├── check_status.bat             # Check if running
└── view_logs.bat                # View recent logs
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Broker API | cTrader Open API (Twisted) |
| Market Data | yfinance |
| SMC Analysis | smartmoneyconcepts |
| Technical Analysis | pandas-ta |
| ML Models | XGBoost, LightGBM |
| Explainability | SHAP |
| Database | SQLite |
| Alerts | python-telegram-bot |
| Sentiment | Groq (LLaMA 3) |
| Calendar | faireconomy.media (Forex Factory, free) |
| Scheduler | schedule |

## Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Trading financial instruments carries significant risk of loss. Past performance does not guarantee future results. The author is not responsible for any financial losses incurred through the use of this software. Always do your own research and consult with a licensed financial advisor before trading.

## License

[MIT License](LICENSE) - Copyright (c) 2026 Pulasthi Ranathunga
