# US500 AI Signal System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Quality](https://github.com/pulasthidin/us500-ai-signal-system/actions/workflows/quality.yml/badge.svg)](https://github.com/pulasthidin/us500-ai-signal-system/actions)
[![Tests](https://img.shields.io/badge/tests-217%20passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6.svg)]()

A fully automated intraday trading signal system for US500 (S&P 500) built on a 4-layer SMC confluence checklist, self-evolving ML models, and Telegram-only alerts. No UI. No dashboards. Just signals to your phone.

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
  +-------+------+  +--------------+  +---------------+
          |
  +-------+-------+-------+-------+
  |       |       |       |       |
  L1      L2      L3      L4      M5
  Macro   H4      H1      M15     Entry
  VIX     EMA     Zones   Delta   FVG
  DXY     BOS     PDH/L   Diverg  BOS
  US10Y   ChoCH   POC     VIX     SMT
  OIL     Wyckoff Round   Spike   R:R
  RUT
```

## The 4-Layer Confluence System

Each signal must pass through 4 independent layers before firing. Only setups scoring 3/4 or 4/4 generate alerts.

| Layer | Timeframe | What It Checks | Data Source |
|-------|-----------|----------------|-------------|
| **L1 Macro** | Daily | VIX level/direction, US10Y, Oil, DXY, RUT | yfinance |
| **L2 Structure** | H4 | EMA 200/50 position, BOS direction, Wyckoff phase | cTrader + smartmoneyconcepts |
| **L3 Zones** | H1 | PDH/PDL, round numbers (every 50pts), volume POC | cTrader |
| **L4 Order Flow** | M15 | Volume delta direction, delta divergence, VIX spike | cTrader |
| **M5 Entry** | M5 | FVG present, BOS confirmed, USTEC SMT, R:R >= target | cTrader + smartmoneyconcepts |

### Scoring & Decisions

| Score | Entry Ready | Decision | Action |
|-------|-------------|----------|--------|
| 4/4 | Yes | **FULL SEND** (Grade A) | Full VIX-sized position |
| 3/4 | Yes | **HALF SIZE** (Grade B) | Half position |
| 3/4 | No | **WAIT** | Heads-up alert only |
| 0-2/4 | — | **NO TRADE** | Silent |
| VIX > 30 | — | **HARD STOP** | All signals blocked |

### Signal Format (Telegram)

```
🟢 LONG — FULL SEND
━━━━━━━━━━━━━━━━━━━━
Entry:  6,544.50
SL:     6,532.00  (-12.5 pts)
TP:     6,565.00  (+20.5 pts)
R:R:    1:1.67
━━━━━━━━━━━━━━━━━━━━
Score:  4/4 ✅
Size:   FULL (VIX normal)
━━━━━━━━━━━━━━━━━━━━
L1: LONG bias | VIX -3.2% falling
L2: Above EMA200 | BOS bullish
L3: At PDL 6,544 | 1.5pts away
L4: Buyers | No divergence
M5: FVG ✅ | BOS ✅ | SMT ✅
━━━━━━━━━━━━━━━━━━━━
📍 London killzone
14:23 SL | Tuesday
[🧠 ML: 74% win probability]
```

## Self-Evolving ML System

The system automatically evolves through 4 stages as it collects your trading data:

| Stage | Signals | What Activates | Benefit |
|-------|---------|----------------|---------|
| **0** | < 200 | Nothing — collecting data | Rule-based only |
| **1** | 200+ | Model 3 (meta-label) trains | Win probability on alerts |
| **2** | 500+ | Pattern scanner activates | Early warning alerts |
| **3** | 1000+ | Daily retraining at midnight | Maximum accuracy |

### Three ML Models

| Model | Type | Target | Purpose |
|-------|------|--------|---------|
| **Model 1** | XGBoost | Good trade day? | Day quality filter |
| **Model 2** | LightGBM | Direction up/down | Session bias prediction |
| **Model 3** | XGBoost multiclass | WIN/LOSS/TIMEOUT | Signal win probability |

All models use walk-forward validation (no data leakage) and SHAP explainability.

## VIX-Based Position Sizing

| VIX Level | Size | Restriction |
|-----------|------|-------------|
| < 15 | Full | — |
| 15–20 | Normal | — |
| 20–25 | Half | — |
| 25–30 | Quarter | Short only |
| > 30 | **NO TRADE** | Hard stop |

## Reliability Features

- **Crash Recovery**: Catches up ALL missed outcomes on restart (even weeks old)
- **Atomic DB Writes**: SQLite transactions — full row or nothing
- **Signal Save Order**: Database FIRST, Telegram SECOND (never lose a signal)
- **Checkpoint File**: Resumes interrupted outcome checks after crash
- **Heartbeat**: Saves every 5 minutes, calculates offline duration on restart
- **10-Point Self-Test**: Verifies all dependencies before starting main loop
- **Exponential Reconnect**: cTrader reconnects with [5,10,20,40,80]s backoff, never gives up
- **Health Monitor**: 6-component checks every 15 min, alerts on state change only
- **Duplicate Prevention**: Same signal not fired twice within 5 minutes
- **News Filter**: Blocks signals 30 min before high-impact USD events

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
│
├── src/
│   ├── ctrader_connection.py    # cTrader Open API + reconnect
│   ├── macro_checker.py         # Layer 1: VIX, DXY, US10Y, Oil, RUT
│   ├── structure_analyzer.py    # Layer 2: EMA, BOS, ChoCH, Wyckoff
│   ├── zone_calculator.py       # Layer 3: PDH/PDL, round levels, POC
│   ├── orderflow_analyzer.py    # Layer 4: volume delta, divergence
│   ├── entry_checker.py         # M5: FVG, BOS, SMT, ATR, R:R
│   ├── checklist_engine.py      # 4-layer orchestrator + filters
│   ├── signal_logger.py         # SQLite database + crash recovery
│   ├── outcome_tracker.py       # Triple barrier + catch-up
│   ├── calendar_checker.py      # News calendar from Forex Factory (free)
│   ├── alert_bot.py             # Two Telegram bots (trade + system)
│   ├── health_monitor.py        # 6-component health checks
│   ├── model_trainer.py         # XGBoost, LightGBM, SHAP
│   ├── model_predictor.py       # Real-time ML predictions
│   ├── evolution_manager.py     # 4-stage auto-evolution
│   └── pattern_scanner.py       # Early warning pattern matching
│
├── tests/                       # 217 unit tests
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

## Repository

This repository is public read-only. Branch protection is enabled on `main`. Only the owner can push changes. Clone or fork to use with your own API keys.

## License

[MIT License](LICENSE) - Copyright (c) 2026 Pulasthi Ranathunga
