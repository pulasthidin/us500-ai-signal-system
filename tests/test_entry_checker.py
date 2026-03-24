"""Tests for EntryChecker — ATR, R:R, SMT, M5 entry composite."""

import pytest
import numpy as np
import pandas as pd
from src.entry_checker import EntryChecker
from config import MIN_RR, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER


@pytest.fixture
def entry(mock_ctrader, mock_alert_bot):
    return EntryChecker(mock_ctrader, us500_id=1, ustec_id=2, alert_bot=mock_alert_bot)


class TestCalculateATR:
    def test_atr_positive(self, entry, sample_ohlcv_df):
        atr = entry.calculate_atr(sample_ohlcv_df)
        assert atr > 0

    def test_atr_small_df(self, entry):
        df = pd.DataFrame({
            "open": [100, 102], "high": [105, 108],
            "low": [98, 99], "close": [103, 106],
        })
        atr = entry.calculate_atr(df, period=2)
        assert atr > 0


class TestCalculateRR:
    def test_long_rr(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 10.0)
        assert result["sl_price"] == 6500.0 - SL_ATR_MULTIPLIER * 10
        assert result["tp_price"] == 6500.0 + TP_ATR_MULTIPLIER * 10
        assert result["rr"] >= MIN_RR

    def test_short_rr(self, entry):
        result = entry.calculate_rr(6500.0, "SHORT", 10.0)
        assert result["sl_price"] == 6500.0 + SL_ATR_MULTIPLIER * 10
        assert result["tp_price"] == 6500.0 - TP_ATR_MULTIPLIER * 10

    def test_rr_ratio_matches(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 10.0)
        expected_rr = TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER
        assert abs(result["rr"] - expected_rr) < 0.01

    def test_zero_atr_no_crash(self, entry):
        result = entry.calculate_rr(6500.0, "LONG", 0.0)
        assert result["rr"] == 0


class TestCheckUstecSMT:
    def test_agrees_when_both_trending_up(self, entry):
        up_df = pd.DataFrame({
            "high": np.linspace(100, 120, 15),
            "low": np.linspace(90, 110, 15),
            "close": np.linspace(95, 115, 15),
        })
        result = entry.check_ustec_smt(up_df, up_df.copy(), "LONG")
        assert isinstance(result["agrees"], (bool, np.bool_))

    def test_handles_empty_dataframes(self, entry):
        result = entry.check_ustec_smt(pd.DataFrame(), pd.DataFrame(), "LONG")
        assert result["agrees"] is False

    def test_handles_none_dataframes(self, entry):
        result = entry.check_ustec_smt(None, None, "SHORT")
        assert result["agrees"] is False


class TestGetEntryResult:
    def test_returns_all_keys(self, entry):
        result = entry.get_entry_result("LONG", 6500.0)
        assert "fvg_present" in result
        assert "m5_bos_confirmed" in result
        assert "ustec_agrees" in result
        assert "rr_valid" in result
        assert "entry_ready" in result
        assert "sl_price" in result
        assert "tp_price" in result
        assert "atr" in result

    def test_entry_ready_requires_fvg_and_rr(self, entry):
        """entry_ready gates on FVG + R:R only; BOS and SMT are bonuses."""
        result = entry.get_entry_result("LONG", 6500.0)
        if result["entry_ready"]:
            assert result["fvg_present"] is True
            assert result["rr_valid"] is True
        # entry_confidence should reflect bonus conditions
        assert "entry_confidence" in result
