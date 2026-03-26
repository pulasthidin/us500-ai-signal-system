"""Tests for config.py — validate all constants are present and typed correctly."""

import config


class TestConfig:
    def test_sessions_utc_has_all_sessions(self):
        assert "Asia" in config.SESSIONS_UTC
        assert "London" in config.SESSIONS_UTC
        assert "NY_Open_Killzone" in config.SESSIONS_UTC
        assert "NY_Session" in config.SESSIONS_UTC

    def test_each_session_has_start_end(self):
        for name, window in config.SESSIONS_UTC.items():
            assert "start" in window, f"{name} missing start"
            assert "end" in window, f"{name} missing end"

    def test_vix_buckets_cover_full_range(self):
        ranges = [v["range"] for v in config.VIX_BUCKETS.values()]
        assert ranges[0][0] == 0
        assert ranges[-1][1] >= 100

    def test_vix_buckets_have_required_keys(self):
        for name, bucket in config.VIX_BUCKETS.items():
            assert "range" in bucket
            assert "size" in bucket
            assert "allowed" in bucket
            assert "short_only" in bucket

    def test_atr_multipliers_produce_valid_rr(self):
        rr = config.TP_ATR_MULTIPLIER / config.SL_ATR_MULTIPLIER
        assert rr >= config.MIN_RR

    def test_news_keywords_not_empty(self):
        assert len(config.NEWS_KEYWORDS) > 0

    def test_sl_utc_offset(self):
        assert config.SL_UTC_OFFSET == 5.5

    def test_grade_scores(self):
        assert config.GRADE_A_SCORE == 4
        assert config.GRADE_B_SCORE == 3

    def test_yf_tickers_present(self):
        assert "vix" in config.YF_TICKERS
        assert "sp500" in config.YF_TICKERS

    def test_sl_min_atr_multiplier_exists(self):
        assert hasattr(config, "SL_MIN_ATR_MULTIPLIER")
        assert config.SL_MIN_ATR_MULTIPLIER > 0
        assert config.SL_MIN_ATR_MULTIPLIER <= config.SL_ATR_MULTIPLIER

    def test_outcome_check_delay_is_1h(self):
        assert config.OUTCOME_CHECK_DELAY_SECONDS == 3600

    def test_outcome_min_bars_m5_and_h1(self):
        assert config.OUTCOME_MIN_M5_BARS_FOR_TIMEOUT >= 48
        assert config.OUTCOME_MIN_H1_BARS_FOR_TIMEOUT >= 6
        assert config.OUTCOME_MIN_M5_BARS_FOR_TIMEOUT > config.OUTCOME_MIN_H1_BARS_FOR_TIMEOUT

    def test_vix_buckets_are_contiguous_and_non_overlapping(self):
        """Every VIX value must fall in exactly one bucket — no gaps, no overlaps."""
        sorted_buckets = sorted(config.VIX_BUCKETS.values(), key=lambda b: b["range"][0])
        for i in range(len(sorted_buckets) - 1):
            current_end = sorted_buckets[i]["range"][1]
            next_start = sorted_buckets[i + 1]["range"][0]
            assert current_end == next_start, (
                f"Gap or overlap between buckets at boundary {current_end} / {next_start}"
            )

    def test_model_retrain_day_is_valid_schedule_day(self):
        valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        assert config.MODEL_RETRAIN_DAY.lower() in valid_days, (
            f"MODEL_RETRAIN_DAY='{config.MODEL_RETRAIN_DAY}' is not a valid schedule day"
        )

    def test_eqh_eql_nearby_points_exists(self):
        assert hasattr(config, "EQH_EQL_NEARBY_POINTS")
        assert config.EQH_EQL_NEARBY_POINTS > 0

    # ─── Range / consolidation detection constants ─────────

    def test_adx_constants_present_and_valid(self):
        assert hasattr(config, "ADX_PERIOD")
        assert config.ADX_PERIOD > 0
        assert hasattr(config, "ADX_RANGE_THRESHOLD")
        assert 10 <= config.ADX_RANGE_THRESHOLD <= 30

    def test_atr_compression_constants(self):
        assert hasattr(config, "ATR_COMPRESSION_LOOKBACK")
        assert config.ATR_COMPRESSION_LOOKBACK >= 10
        assert hasattr(config, "ATR_COMPRESSION_RATIO")
        assert 0 < config.ATR_COMPRESSION_RATIO < 1.0

    def test_range_lookback_bars(self):
        assert hasattr(config, "RANGE_LOOKBACK_BARS")
        assert config.RANGE_LOOKBACK_BARS >= 10

    # ─── Displacement candle constants ─────────────────────

    def test_displacement_body_ratio(self):
        assert hasattr(config, "DISPLACEMENT_BODY_RATIO")
        assert 0.3 <= config.DISPLACEMENT_BODY_RATIO <= 0.8

    def test_displacement_atr_ratio(self):
        assert hasattr(config, "DISPLACEMENT_ATR_RATIO")
        assert 0.3 <= config.DISPLACEMENT_ATR_RATIO <= 1.5

    # ─── Liquidity sweep constants ─────────────────────────

    def test_sweep_constants_present(self):
        assert hasattr(config, "SWEEP_LOOKBACK_BARS")
        assert config.SWEEP_LOOKBACK_BARS >= 10
        assert hasattr(config, "SWEEP_WICK_MIN_POINTS")
        assert config.SWEEP_WICK_MIN_POINTS >= 0

    # ─── Asian session constants ───────────────────────────

    def test_asian_session_times(self):
        assert hasattr(config, "ASIAN_SESSION_START_UTC")
        assert hasattr(config, "ASIAN_SESSION_END_UTC")
        start_h = int(config.ASIAN_SESSION_START_UTC.split(":")[0])
        end_h = int(config.ASIAN_SESSION_END_UTC.split(":")[0])
        assert end_h > start_h
