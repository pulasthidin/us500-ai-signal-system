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

    def test_outcome_check_delay_is_30m(self):
        assert config.OUTCOME_CHECK_DELAY_SECONDS == 1800

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
