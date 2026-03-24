"""
Central configuration for the US500 intraday trading signal app.
All constants and thresholds live here — no magic numbers in src/ modules.
"""

# ──────────────────────────────────────────────
# SESSION WINDOWS (UTC)
# ──────────────────────────────────────────────
SESSIONS_UTC = {
    "Asia":              {"start": "00:00", "end": "09:00"},
    "London":            {"start": "07:00", "end": "16:00"},
    "NY_Open_Killzone":  {"start": "13:00", "end": "16:00"},
    "NY_Session":        {"start": "16:00", "end": "21:00"},
}

SL_UTC_OFFSET = 5.5  # Sri Lanka is UTC+5:30

TRADING_START_SL = "05:30"
TRADING_END_SL = "23:00"

# ──────────────────────────────────────────────
# SYMBOLS
# ──────────────────────────────────────────────
US500_SYMBOL = "US500"
USTEC_SYMBOL = "USTEC"

US500_SYMBOL_ALIASES = ["US500", "SPX500", "SP500"]
USTEC_SYMBOL_ALIASES = ["USTEC", "NAS100", "NASDAQ100"]

# ──────────────────────────────────────────────
# ZONE / LEVEL SETTINGS
# ──────────────────────────────────────────────
ROUND_LEVEL_INTERVAL = 50
ZONE_THRESHOLD_POINTS = 15
ZONE_DEDUP_DISTANCE = 5  # merge levels within 5 points

# ──────────────────────────────────────────────
# ENTRY / EXIT
# ──────────────────────────────────────────────
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 2.5
MIN_RR = 1.6  # TP/SL = 2.5/1.5 = 1.67; threshold set just below to allow entries

# ──────────────────────────────────────────────
# VIX BUCKETS — position sizing & trade gating
# ──────────────────────────────────────────────
VIX_BUCKETS = {
    "low":      {"range": (0, 15),    "size": "full",               "allowed": True,  "short_only": False},
    "normal":   {"range": (15, 20),   "size": "normal",             "allowed": True,  "short_only": False},
    "elevated": {"range": (20, 25),   "size": "half",               "allowed": True,  "short_only": False},
    "high":     {"range": (25, 30),   "size": "quarter_short_only", "allowed": True,  "short_only": True},
    "extreme":  {"range": (30, 999),  "size": "no_trade",           "allowed": False, "short_only": False},
}

VIX_DIRECTION_THRESHOLD = 1.5  # % change to trigger directional bias

# ──────────────────────────────────────────────
# MACRO DIRECTION THRESHOLDS
# ──────────────────────────────────────────────
MACRO_THRESHOLDS = {
    "us10y": {"rising": 0.05, "falling": -0.05},
    "oil":   {"spiking": 0.5, "falling": -0.5},
    "dxy":   {"rising": 0.2, "falling": -0.2},
    "rut":   {"green": 0.3, "red": -0.3},
}

# ──────────────────────────────────────────────
# NEWS CALENDAR
# ──────────────────────────────────────────────
NEWS_KEYWORDS = [
    "NFP", "CPI", "FOMC", "GDP", "Fed", "Powell",
    "Interest Rate", "Non-Farm", "Unemployment",
    "Retail Sales", "PPI", "PCE",
]
NEWS_BLOCK_MINUTES_BEFORE = 30
NEWS_CAUTION_MINUTES_AFTER = 30

# ──────────────────────────────────────────────
# SIGNAL DEDUP & OUTCOME
# ──────────────────────────────────────────────
SIGNAL_DEDUP_MINUTES = 5
OUTCOME_CHECK_DELAY_SECONDS = 7200  # 2 hours
MAX_BARS_MEMORY = 500

# ──────────────────────────────────────────────
# ML MODEL RETRAINING
# ──────────────────────────────────────────────
MODEL_RETRAIN_SIGNAL_THRESHOLD = 200
MODEL_INCREMENTAL_THRESHOLD = 50
MODEL_RETRAIN_DAY = "sunday"
MODEL_RETRAIN_TIME_SL = "23:00"

# ──────────────────────────────────────────────
# SCHEDULING
# ──────────────────────────────────────────────
MORNING_BRIEF_TIME_UTC = "00:00"  # = 05:30 SL
HEALTH_CHECK_INTERVAL_MINUTES = 15
OUTCOME_CHECK_INTERVAL_MINUTES = 30
SIGNAL_CHECK_INTERVAL_SECONDS = 60
MACRO_CACHE_SECONDS = 300

# ──────────────────────────────────────────────
# ALERTING
# ──────────────────────────────────────────────
NO_SIGNAL_ALERT_HOURS = 4
SYSTEM_ALERT_DEDUP_MINUTES = 10

# ──────────────────────────────────────────────
# CTRADER CONNECTION
# ──────────────────────────────────────────────
CTRADER_RECONNECT_DELAYS = [5, 10, 20, 40, 80]  # exponential backoff seconds
CTRADER_MAX_RETRIES = 5
CTRADER_TOKEN_WARN_DAYS = 10  # warn this many days before expiry

# ──────────────────────────────────────────────
# GRADING
# ──────────────────────────────────────────────
GRADE_A_SCORE = 4
GRADE_B_SCORE = 3

# ──────────────────────────────────────────────
# H4 STRUCTURE DETECTION
# ──────────────────────────────────────────────
H4_BOS_SWING_LENGTH = 10  # swing lookback for BOS/ChoCH (50 was ~8 days, 10 is ~2 days)

# ──────────────────────────────────────────────
# VIX SPIKE INTRA-BAR
# ──────────────────────────────────────────────
VIX_SPIKE_THRESHOLD_PCT = 3.0  # real-time spike gate on M15

# ──────────────────────────────────────────────
# WEEKLY REPORT
# ──────────────────────────────────────────────
WEEKLY_REPORT_TIME_UTC = "17:30"

# ──────────────────────────────────────────────
# YFINANCE TICKERS
# ──────────────────────────────────────────────
YF_TICKERS = {
    "vix":   "^VIX",
    "dxy":   "DX-Y.NYB",
    "us10y": "^TNX",
    "oil":   "CL=F",
    "rut":   "^RUT",
    "sp500": "^GSPC",
}
