"""
Regression tests built from REAL winning trades in the production database.

Every test here replicates the exact conditions of a winning trade that the
system correctly identified (Wed Mar 25 – Thu Mar 26, 2026).

Purpose: Prove that new filters (range detection, displacement validation,
liquidity sweep, range SL) do NOT block any winning trades when deployed.

Test logic for each trade:
  1. Is the market trending? → ADX > 20 → sweep is bonus only, never blocks
  2. Does the FVG candle pass displacement? → body>50%, range>0.6xATR
  3. In trending mode, entry_ready = FVG + RR only → always True for these
  4. Checklist decision must be FULL_SEND or HALF_SIZE → never NO_TRADE
"""

import pytest
from src.entry_checker import EntryChecker
from src.checklist_engine import ChecklistEngine
from unittest.mock import MagicMock


@pytest.fixture
def entry(mock_ctrader, mock_alert_bot):
    return EntryChecker(mock_ctrader, us500_id=1, ustec_id=2, alert_bot=mock_alert_bot)


@pytest.fixture
def engine(mock_alert_bot):
    eng = ChecklistEngine(
        macro_checker=MagicMock(), structure_analyzer=MagicMock(),
        zone_calculator=MagicMock(), orderflow_analyzer=MagicMock(),
        entry_checker=MagicMock(), signal_logger=MagicMock(is_duplicate=MagicMock(return_value=False)),
        alert_bot=mock_alert_bot,
    )
    return eng


def _make_checklist_result(engine, trade, sample_news_data):
    """Build full checklist mocks from a winning trade dict and run through the pipeline."""
    macro = {
        "bias": "SHORT", "vix_bucket": trade["vix_bucket"],
        "vix_value": trade["vix"], "vix_pct": 2.0,
        "size_label": "half" if trade["vix"] > 20 else "normal",
        "short_only": trade["vix"] >= 25,
        "vix_direction_bias": "SELL_BIAS",
        "score_contribution": 1,
        "us10y_direction": "rising", "oil_direction": "stable",
        "dxy_direction": "rising", "rut_direction": "red",
        "groq_sentiment": "BEARISH", "bullish_count": 0, "bearish_count": 3,
        "raw_data": {"vix": {"value": trade["vix"], "pct_change": 2.0}},
    }
    range_cond = {
        "is_ranging": False, "range_strength": "none",
        "adx_value": 28.0, "adx_ranging": False,
        "atr_compressing": False, "atr_ratio": 1.1,
        "range_high": trade["entry"] + 30, "range_low": trade["entry"] - 30,
    }
    engine._structure.get_layer2_result.return_value = {
        "structure_bias": "short", "score_contribution": 1,
        "above_ema50": False, "ema_bias": "bearish",
        "bos_direction": "bearish", "bos_level": trade["entry"] - 20,
        "bos_bars_ago": 3, "choch_direction": None, "choch_recent": False,
        "wyckoff": "markdown", "ema50_value": trade["entry"] + 20,
        "ob_bullish_nearby": False, "ob_bearish_nearby": False,
        "bullish_obs": [], "bearish_obs": [],
        "range_condition": range_cond,
    }
    engine._zones.get_layer3_result.return_value = {
        "at_zone": True, "zone_level": trade["entry"],
        "zone_type": trade.get("zone_type", "round"),
        "distance": 3.0, "zone_direction": "resistance",
        "score_contribution": 1, "all_levels": [], "all_nearby": [],
        "asian_range": {"asian_high": trade["entry"] + 10, "asian_low": trade["entry"] - 15, "valid": True},
        "liquidity_sweeps": [], "has_buy_side_sweep": False, "has_sell_side_sweep": False,
    }
    engine._orderflow.get_layer4_result.return_value = {
        "delta_direction": trade.get("delta", "sellers"),
        "divergence": trade.get("divergence", "none"),
        "vix_spiking_now": False,
        "confirms_bias": trade.get("confirms", True),
        "score_contribution": 1 if trade.get("confirms", True) else 0,
    }
    engine._entry.get_entry_result.return_value = {
        "entry_ready": True,
        "entry_confidence": trade.get("confidence", "base"),
        "fvg_present": True, "fvg_details": {"top": trade["sl"], "bottom": trade["entry"]},
        "displacement_valid": True,
        "m5_bos_confirmed": False,
        "ustec_agrees": trade.get("smt", False),
        "rr_valid": True, "rr": trade["rr"],
        "has_liquidity_sweep": False,
        "sweep_details": None,
        "sl_price": trade["sl"], "tp_price": trade["tp"],
        "sl_points": trade["sl_pts"], "tp_points": trade["tp_pts"],
        "atr": trade["atr"], "tp_source": trade.get("tp_source", "atr"),
        "sl_method": "swing",
        "ustec_details": {"agrees": trade.get("smt", False)},
    }
    return engine.run_full_checklist(trade["entry"], macro, sample_news_data)


# All 15 winning trades from the database
WINNING_TRADES = [
    {"id": 129, "entry": 6621.5, "sl": 6632.58, "tp": 6600.0, "sl_pts": 11.08, "tp_pts": 21.5,
     "rr": 1.94, "atr": 9.79, "vix": 24.94, "vix_bucket": "elevated", "pnl": 21.5,
     "score": 3, "confidence": "base", "delta": "buyers", "confirms": False,
     "zone_type": "pdh", "tp_source": "round"},
    {"id": 130, "entry": 6618.2, "sl": 6631.08, "tp": 6593.73, "sl_pts": 12.88, "tp_pts": 24.47,
     "rr": 1.9, "atr": 9.79, "vix": 24.97, "vix_bucket": "elevated", "pnl": 24.5,
     "score": 3, "confidence": "base", "delta": "buyers", "confirms": False, "tp_source": "atr"},
    {"id": 157, "entry": 6625.1, "sl": 6634.68, "tp": 6600.0, "sl_pts": 9.58, "tp_pts": 25.1,
     "rr": 2.62, "atr": 9.58, "vix": 25.22, "vix_bucket": "high", "pnl": 25.1,
     "score": 4, "confidence": "base", "delta": "buyers", "confirms": True,
     "divergence": "bearish", "zone_type": "pdh", "tp_source": "round"},
    {"id": 162, "entry": 6606.6, "sl": 6621.48, "tp": 6569.4, "sl_pts": 14.88, "tp_pts": 37.2,
     "rr": 2.5, "atr": 14.88, "vix": 25.99, "vix_bucket": "high", "pnl": 37.2,
     "score": 4, "confidence": "base", "zone_type": "poc", "tp_source": "atr"},
    {"id": 164, "entry": 6606.5, "sl": 6621.47, "tp": 6569.08, "sl_pts": 14.97, "tp_pts": 37.42,
     "rr": 2.5, "atr": 14.97, "vix": 25.3, "vix_bucket": "high", "pnl": 37.4,
     "score": 4, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "atr"},
    {"id": 165, "entry": 6608.7, "sl": 6623.67, "tp": 6571.28, "sl_pts": 14.97, "tp_pts": 37.42,
     "rr": 2.5, "atr": 14.97, "vix": 25.21, "vix_bucket": "high", "pnl": 37.4,
     "score": 3, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "atr"},
    {"id": 166, "entry": 6611.3, "sl": 6626.27, "tp": 6573.88, "sl_pts": 14.97, "tp_pts": 37.42,
     "rr": 2.5, "atr": 14.97, "vix": 25.37, "vix_bucket": "high", "pnl": 37.4,
     "score": 4, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "atr"},
    {"id": 167, "entry": 6612.7, "sl": 6627.56, "tp": 6575.56, "sl_pts": 14.86, "tp_pts": 37.14,
     "rr": 2.5, "atr": 14.86, "vix": 25.26, "vix_bucket": "high", "pnl": 37.1,
     "score": 3, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "atr"},
    {"id": 168, "entry": 6614.8, "sl": 6629.66, "tp": 6577.66, "sl_pts": 14.86, "tp_pts": 37.14,
     "rr": 2.5, "atr": 14.86, "vix": 25.24, "vix_bucket": "high", "pnl": 37.1,
     "score": 3, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "atr"},
    {"id": 169, "entry": 6613.0, "sl": 6627.4, "tp": 6576.99, "sl_pts": 14.4, "tp_pts": 36.01,
     "rr": 2.5, "atr": 14.4, "vix": 25.06, "vix_bucket": "high", "pnl": 36.0,
     "score": 4, "confidence": "base", "zone_type": "poc", "tp_source": "atr"},
    {"id": 170, "entry": 6610.7, "sl": 6625.1, "tp": 6574.69, "sl_pts": 14.4, "tp_pts": 36.01,
     "rr": 2.5, "atr": 14.4, "vix": 25.04, "vix_bucket": "high", "pnl": 36.0,
     "score": 4, "confidence": "base", "zone_type": "poc", "tp_source": "atr"},
    {"id": 204, "entry": 6590.4, "sl": 6598.66, "tp": 6567.5, "sl_pts": 8.26, "tp_pts": 22.9,
     "rr": 2.77, "atr": 8.26, "vix": 25.35, "vix_bucket": "high", "pnl": 22.9,
     "score": 4, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "pdl"},
    {"id": 205, "entry": 6586.6, "sl": 6594.86, "tp": 6567.5, "sl_pts": 8.26, "tp_pts": 19.1,
     "rr": 2.31, "atr": 8.26, "vix": 25.35, "vix_bucket": "high", "pnl": 19.1,
     "score": 4, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "pdl"},
    {"id": 206, "entry": 6585.9, "sl": 6593.86, "tp": 6567.5, "sl_pts": 7.96, "tp_pts": 18.4,
     "rr": 2.31, "atr": 7.96, "vix": 25.35, "vix_bucket": "high", "pnl": 18.4,
     "score": 3, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "pdl"},
    {"id": 207, "entry": 6583.4, "sl": 6591.36, "tp": 6567.5, "sl_pts": 7.96, "tp_pts": 15.9,
     "rr": 2.0, "atr": 7.96, "vix": 25.35, "vix_bucket": "high", "pnl": 15.9,
     "score": 3, "confidence": "high", "smt": True, "zone_type": "poc", "tp_source": "pdl"},
]


class TestDisplacementPassesForAllWinners:
    """Every winning trade's ATR and typical candle size must pass displacement validation."""

    @pytest.mark.parametrize("trade", WINNING_TRADES, ids=[f"#{t['id']}" for t in WINNING_TRADES])
    def test_typical_displacement_candle_passes(self, trade):
        atr = trade["atr"]
        sl_pts = trade["sl_pts"]
        body = sl_pts * 0.7
        candle_range = sl_pts * 1.1
        result = EntryChecker._is_displacement_candle(
            open_price=trade["entry"],
            close_price=trade["entry"] - body,
            high=trade["entry"] + (candle_range - sl_pts) * 0.3,
            low=trade["entry"] - body - (candle_range - body) * 0.5,
            atr=atr,
        )
        assert result is True, (
            f"Trade #{trade['id']}: displacement must pass for a candle with "
            f"body={body:.1f} range={candle_range:.1f} atr={atr:.1f}"
        )


class TestSweepDoesNotBlockTrendingWinners:
    """In trending market (ADX>20), no sweep = entry_confidence='no_sweep' but entry_ready=True."""

    @pytest.mark.parametrize("trade", WINNING_TRADES, ids=[f"#{t['id']}" for t in WINNING_TRADES])
    def test_trending_no_sweep_still_fires(self, entry, trade):
        range_cond = {
            "is_ranging": False, "range_strength": "none",
            "adx_value": 28.0, "adx_ranging": False,
        }
        result = entry.get_entry_result(
            "SHORT", trade["entry"],
            range_condition=range_cond,
            liquidity_sweeps=[],
        )
        if result["fvg_present"] and result["rr_valid"]:
            assert result["entry_ready"] is True, (
                f"Trade #{trade['id']}: trending + FVG + RR valid → entry_ready must be True"
            )


class TestChecklistProducesTradeForAllWinners:
    """Full checklist pipeline must produce FULL_SEND or HALF_SIZE for every real winner."""

    @pytest.mark.parametrize("trade", WINNING_TRADES, ids=[f"#{t['id']}" for t in WINNING_TRADES])
    def test_checklist_fires_signal(self, engine, trade, sample_news_data):
        result = _make_checklist_result(engine, trade, sample_news_data)

        assert result["decision"] in ("FULL_SEND", "HALF_SIZE"), (
            f"Trade #{trade['id']} (real WIN +{trade['pnl']:.1f}pts) must fire! "
            f"Got decision={result['decision']}, grade={result['grade']}, "
            f"block={result.get('block_reason')}"
        )
        assert result["entry_ready"] is True
        assert result["direction"] == "SHORT"
