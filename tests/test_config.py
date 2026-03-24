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
