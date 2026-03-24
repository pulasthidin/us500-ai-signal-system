"""
Telegram Alert Bot — two independent bots for TRADE alerts and SYSTEM health.
All message formatting lives here so the rest of the app stays format-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import os
import telegram

from config import SL_UTC_OFFSET, SESSIONS_UTC, SYSTEM_ALERT_DEDUP_MINUTES

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("alert_bot")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/alerts.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)


class AlertBot:
    """Formats and dispatches messages to two Telegram channels."""

    def __init__(self) -> None:
        self._trade_token = os.getenv("TELEGRAM_TRADE_TOKEN", "")
        self._trade_chat_id = os.getenv("TELEGRAM_TRADE_CHAT_ID", "")
        self._system_token = os.getenv("TELEGRAM_SYSTEM_TOKEN", "")
        self._system_chat_id = os.getenv("TELEGRAM_SYSTEM_CHAT_ID", "")

        self._trade_bot: Optional[telegram.Bot] = None
        self._system_bot: Optional[telegram.Bot] = None

        if self._trade_token:
            self._trade_bot = telegram.Bot(token=self._trade_token)
        if self._system_token:
            self._system_bot = telegram.Bot(token=self._system_token)

        self._system_alert_cache: Dict[str, float] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ─── SL time helper ──────────────────────────────────────

    @staticmethod
    def _sl_time_str() -> str:
        sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
        return datetime.now(sl_tz).strftime("%H:%M SL")

    @staticmethod
    def _sl_date_str() -> str:
        sl_tz = timezone(timedelta(hours=SL_UTC_OFFSET))
        return datetime.now(sl_tz).strftime("%A %b %d").upper()

    # ──────────────────────────────────────────────────────────
    # TRADE ALERT FORMATTING
    # ──────────────────────────────────────────────────────────

    def format_trade_alert(self, result: Dict[str, Any], ml_result: Optional[Dict] = None) -> str:
        """Build the Telegram trade-alert string for FULL_SEND / HALF_SIZE / WAIT / HARD_STOP."""
        try:
            decision = result.get("decision", "NO_TRADE")
            direction = result.get("direction", "")
            entry = result.get("entry") or {}

            if decision == "HARD_STOP":
                return self._format_hard_stop(result)

            if decision == "WAIT":
                return self._format_wait(result)

            arrow = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
            size_tag = "FULL SEND" if decision == "FULL_SEND" else "HALF SIZE"

            price = result.get("current_price") or 0
            sl = entry.get("sl_price") or 0
            tp = entry.get("tp_price") or 0
            sl_pts = entry.get("sl_points") or 0
            tp_pts = entry.get("tp_points") or 0
            rr = entry.get("rr") or 0
            score = result.get("score") or 0
            layer1 = result.get("layer1") or {}
            layer2 = result.get("layer2") or {}
            layer3 = result.get("layer3") or {}
            layer4 = result.get("layer4") or {}
            vix_pct = layer1.get("vix_pct") or 0
            vix_bucket = result.get("vix_bucket", "")

            lines = [
                f"{arrow} {direction} — {size_tag}",
                "\u2501" * 20,
                f"Entry:  {price:,.2f}",
                f"SL:     {sl:,.2f}  ({'-' if direction == 'LONG' else '+'}{sl_pts:.1f} pts)",
                f"TP:     {tp:,.2f}  ({'+' if direction == 'LONG' else '-'}{tp_pts:.1f} pts)",
                f"R:R:    1:{rr:.2f}",
                "\u2501" * 20,
                f"Score:  {score}/4 \u2705",
                f"Size:   {result.get('size_label', '').upper()} (VIX {vix_bucket})",
                "\u2501" * 20,
                f"L1: {layer1.get('bias', '?')} bias | VIX {vix_pct:+.1f}% {'falling' if vix_pct < 0 else 'rising'}",
                f"L2: {'Above' if layer2.get('above_ema200') else 'Below'} EMA200 | BOS {layer2.get('bos_direction', '?')}"
                f"{' | OB nearby' if layer2.get('ob_bullish_nearby') or layer2.get('ob_bearish_nearby') else ''}",
                f"L3: {'At' if layer3.get('at_zone') else 'No'} {layer3.get('zone_type', 'zone')} {layer3.get('zone_level', '')}"
                f" | {(layer3.get('distance') or 0):.1f}pts away",
                f"L4: {layer4.get('delta_direction', '?').title()} | "
                f"{'Divergence: ' + layer4.get('divergence', 'none') if layer4.get('divergence', 'none') != 'none' else 'No divergence'}",
            ]

            fvg_ok = "\u2705" if entry.get("fvg_present") else "\u274c"
            bos_ok = "\u2705" if entry.get("m5_bos_confirmed") else "\u274c"
            smt_ok = "\u2705" if entry.get("ustec_agrees") else "\u274c"
            conf_label = (entry.get("entry_confidence") or "?").upper()
            lines.append(f"M5: FVG {fvg_ok} | BOS {bos_ok} | SMT {smt_ok} [{conf_label}]")
            lines.append("\u2501" * 20)

            session = result.get("session", "")
            sl_time = result.get("sl_time", self._sl_time_str())
            day = result.get("day_name", "")
            lines.append(f"\U0001f4cd {session}")
            lines.append(f"{sl_time} | {day}")

            if ml_result and ml_result.get("win_probability") is not None:
                wp = ml_result["win_probability"]
                lines.append(f"[\U0001f9e0 ML: {wp:.0%} win probability]")
                if ml_result.get("shap_top_features"):
                    feats = ", ".join(ml_result["shap_top_features"][:3])
                    lines.append(f"[Top factors: {feats}]")

            pattern_early = result.get("pattern_called_minutes_ago")
            if pattern_early is not None:
                lines.append(f"\u26a1 Pattern scanner called this {pattern_early:.0f} min ago")

            cautions = result.get("caution_flags", [])
            for c in cautions:
                lines.append(f"\u26a0\ufe0f {c}")

            return "\n".join(lines)

        except Exception as exc:
            logger.error("format_trade_alert failed: %s", exc, exc_info=True)
            return f"Signal alert (formatting error): {result.get('decision')}"

    def _format_hard_stop(self, result: Dict[str, Any]) -> str:
        vix_val = result.get("layer1", {}).get("vix_value", 0)
        return (
            "\U0001f6ab HARD STOP\n"
            f"VIX at {vix_val:.1f} — panic mode\n"
            "All signals blocked\n"
            "Close charts, come back tomorrow."
        )

    def _format_wait(self, result: Dict[str, Any]) -> str:
        layer3 = result.get("layer3", {})
        entry = result.get("entry") or {}
        missing = []
        if not entry.get("fvg_present"):
            missing.append("M5 FVG not formed")
        if not entry.get("rr_valid", True):
            missing.append("R:R below minimum")
        # BOS/SMT are bonuses, not blockers — show as info, not "missing"
        bonus_info = []
        if not entry.get("m5_bos_confirmed"):
            bonus_info.append("BOS pending")
        if not entry.get("ustec_agrees"):
            bonus_info.append("SMT pending")
        missing_str = ", ".join(missing) if missing else "entry conditions pending"
        bonus_str = " | Bonus: " + ", ".join(bonus_info) if bonus_info else ""

        return (
            "\u23f3 WAIT — setup building\n"
            f"Score {result.get('score', 0)}/4 — entry not ready yet\n"
            f"Watching: {layer3.get('zone_level', '?')} zone\n"
            f"Missing: {missing_str}{bonus_str}\n"
            f"Session: {result.get('session', '?')} | {result.get('sl_time', self._sl_time_str())}"
        )

    # ──────────────────────────────────────────────────────────
    # MORNING BRIEF
    # ──────────────────────────────────────────────────────────

    def format_morning_brief(
        self,
        macro_data: Dict[str, Any],
        news_data: Dict[str, Any],
        zone_data: List[Dict[str, Any]],
        ml_stats: Optional[Dict] = None,
    ) -> str:
        """Daily morning briefing sent before London open."""
        try:
            date_str = self._sl_date_str()
            vix = macro_data.get("vix", macro_data.get("raw_data", {}).get("vix", {}))
            vix_val = vix.get("value", 0) or macro_data.get("vix_value", 0)
            vix_pct = vix.get("pct_change", 0) or macro_data.get("vix_pct", 0)
            vix_bucket = macro_data.get("vix_bucket", "?")

            dxy = macro_data.get("dxy", macro_data.get("raw_data", {}).get("dxy", {}))
            us10y = macro_data.get("us10y", macro_data.get("raw_data", {}).get("us10y", {}))
            oil = macro_data.get("oil", macro_data.get("raw_data", {}).get("oil", {}))
            rut = macro_data.get("rut", macro_data.get("raw_data", {}).get("rut", {}))

            bias = macro_data.get("bias", "MIXED")
            arrow_map = lambda p: "\u2191" if p and p > 0 else ("\u2193" if p and p < 0 else "\u2194")

            lines = [
                f"\U0001f4c5 {date_str} — LONDON + NY",
                "",
                f"\U0001f4ca DAY BIAS: {bias}",
                f"VIX  {vix_val or '?'}  {vix_pct:+.2f}% {arrow_map(vix_pct)}  [{vix_bucket}]",
                f"DXY  {dxy.get('value', '?')}  {dxy.get('pct_change', 0):+.2f}% {arrow_map(dxy.get('pct_change'))}",
                f"US10Y {us10y.get('value', '?')}  {us10y.get('pct_change', 0):+.2f}% {arrow_map(us10y.get('pct_change'))}",
                f"OIL  {oil.get('value', '?')}  {oil.get('pct_change', 0):+.2f}% {arrow_map(oil.get('pct_change'))}",
                f"RUT  {rut.get('pct_change', 0):+.2f}% {arrow_map(rut.get('pct_change'))}",
            ]

            lines.append("")
            lines.append("\U0001f4cd KEY LEVELS TODAY")
            valid_zones = [z for z in zone_data if z.get("price", 0) > 0]
            mid = len(valid_zones) // 2
            resistance = valid_zones[mid:][:3]
            support = valid_zones[:mid][-3:]
            res_str = " | ".join(
                f"{z.get('price', 0):,.0f}" + (f"({z.get('type', 'round').upper()})" if z.get("type", "round") != "round" else "")
                for z in reversed(resistance)
            )
            sup_str = " | ".join(
                f"{z.get('price', 0):,.0f}" + (f"({z.get('type', 'round').upper()})" if z.get("type", "round") != "round" else "")
                for z in support
            )
            lines.append(f"Res: {res_str}" if res_str else "Res: —")
            lines.append(f"Sup: {sup_str}" if sup_str else "Sup: —")

            lines.append("")
            lines.append("\U0001f4f0 HIGH IMPACT NEWS")
            if news_data.get("calendar_unavailable"):
                lines.append("News calendar unavailable today")
            else:
                events = news_data.get("events_today", [])
                if events:
                    for e in events[:5]:
                        title = e.get("title", e.get("event", "?"))
                        evt_time = e.get("date", e.get("time", "?"))
                        lines.append(f"  \u2022 {title} @ {evt_time}")
                else:
                    lines.append("None today — clear to trade")

            lines.append("")
            lines.append("\u23f0 YOUR SESSIONS (SL TIME)")
            lines.append("London:   12:30 – 21:30")
            lines.append("NY Open:  18:30 – 21:30")
            lines.append("NY:       21:30 – 02:30")

            if ml_stats:
                cnt = ml_stats.get("signal_count", 0)
                wr = ml_stats.get("win_rate", 0)
                lines.append("")
                lines.append(f"[\U0001f4c8 {cnt} signals | {wr:.0f}% win rate]")

            return "\n".join(lines)

        except Exception as exc:
            logger.error("format_morning_brief failed: %s", exc, exc_info=True)
            return f"\U0001f4c5 Morning Brief (formatting error)\n{exc}"

    # ──────────────────────────────────────────────────────────
    # SYSTEM ALERTS
    # ──────────────────────────────────────────────────────────

    def format_system_alert(self, level: str, component: str, message: str, extra: Optional[str] = None) -> str:
        """Format a system-health alert for the SYSTEM channel."""
        icons = {"WARNING": "\u26a0\ufe0f WARNING", "CRITICAL": "\U0001f534 CRITICAL", "INFO": "\u2705 INFO"}
        header = icons.get(level, level)
        sl_time = self._sl_time_str()
        parts = [f"{header} — {component}", message]
        if level == "CRITICAL":
            parts.append("Manual check needed")
        if extra:
            parts.append(extra)
        parts.append(f"Time: {sl_time}")
        return "\n".join(parts)

    # ──────────────────────────────────────────────────────────
    # WEEKLY REPORT
    # ──────────────────────────────────────────────────────────

    def format_weekly_report(self, stats: Dict[str, Any]) -> str:
        """Format the Sunday weekly performance summary."""
        try:
            wr = stats.get("win_rate", 0)
            by_session = stats.get("by_session", {})
            by_grade = stats.get("by_grade", {})

            lines = [
                "\U0001f4ca WEEKLY REPORT",
                "\u2501" * 20,
                f"Win rate: {wr:.1f}%",
                "",
                "BY SESSION:",
            ]
            for sess, data in by_session.items():
                lines.append(f"  {sess}: {data.get('win_rate', 0):.0f}% ({data.get('total', 0)} signals)")

            lines.append("")
            lines.append("BY GRADE:")
            for g, data in by_grade.items():
                lines.append(f"  Grade {g}: {data.get('win_rate', 0):.0f}% ({data.get('total', 0)} signals)")

            health = stats.get("system_health", {})
            if health:
                lines.append("")
                lines.append("SYSTEM HEALTH:")
                for comp, status in health.items():
                    ok = status if isinstance(status, bool) else status.get("ok", True) if isinstance(status, dict) else True
                    icon = "\u2705" if ok else "\u274c"
                    lines.append(f"  {comp}: {icon}")

            shap_report = stats.get("shap_report")
            if shap_report:
                lines.append("")
                lines.append("TOP ML FEATURES:")
                for feat in shap_report[:5]:
                    lines.append(f"  {feat}")

            evo = stats.get("evolution")
            if evo:
                lines.append("")
                lines.append("EVOLUTION:")
                lines.append(f"  Stage: {evo.get('stage', 0)}")
                lines.append(f"  Signals: {evo.get('signal_count', 0)}")

            pa_stats = stats.get("pattern_accuracy")
            if pa_stats and pa_stats.get("total_alerts", 0) > 0:
                lines.append("")
                lines.append("PATTERN SCANNER:")
                lines.append(f"  Alerts fired: {pa_stats['total_alerts']}")
                lines.append(f"  Led to signal: {pa_stats['match_rate']:.0f}%")
                lines.append(f"  Signal won: {pa_stats['win_rate_after_match']:.0f}%")

            return "\n".join(lines)

        except Exception as exc:
            logger.error("format_weekly_report failed: %s", exc, exc_info=True)
            return f"\U0001f4ca Weekly Report (formatting error)\n{exc}"

    # ──────────────────────────────────────────────────────────
    # SENDERS
    # ──────────────────────────────────────────────────────────

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return a persistent event loop for Telegram calls (created once, reused)."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro):
        """Run an async coroutine from sync context, safe with Twisted reactor."""
        try:
            loop = self._get_loop()
            loop.run_until_complete(coro)
        except Exception as exc:
            logger.error("_run_async failed: %s", exc, exc_info=True)
            if self._loop is not None:
                try:
                    self._loop.close()
                except Exception:
                    pass
            self._loop = None

    def send_trade_message(self, message: str) -> None:
        """Send an arbitrary text message to the TRADE channel."""
        try:
            if self._trade_bot and self._trade_chat_id:
                self._run_async(self._trade_bot.send_message(chat_id=self._trade_chat_id, text=message))
                logger.info("Trade message sent (%d chars)", len(message))
        except Exception as exc:
            logger.error("send_trade_message failed: %s", exc, exc_info=True)

    def send_trade_alert(self, result: Dict[str, Any], ml_result: Optional[Dict] = None) -> None:
        """Format and send a trade alert to the TRADE channel."""
        try:
            msg = self.format_trade_alert(result, ml_result)
            self.send_trade_message(msg)
            logger.info("Trade alert sent: %s", result.get("decision"))
        except Exception as exc:
            logger.error("send_trade_alert failed: %s", exc, exc_info=True)

    def send_morning_brief(
        self,
        macro_data: Dict[str, Any],
        news_data: Dict[str, Any],
        zone_data: List[Dict[str, Any]],
        ml_stats: Optional[Dict] = None,
    ) -> None:
        """Send the daily morning brief to the TRADE channel."""
        try:
            msg = self.format_morning_brief(macro_data, news_data, zone_data, ml_stats)
            if self._trade_bot and self._trade_chat_id:
                self._run_async(self._trade_bot.send_message(chat_id=self._trade_chat_id, text=msg))
            logger.info("Morning brief sent")
        except Exception as exc:
            logger.error("send_morning_brief failed: %s", exc, exc_info=True)

    def send_system_alert(self, level: str, component: str, message: str) -> None:
        """Send a system alert to the SYSTEM channel with dedup."""
        try:
            cache_key = f"{level}:{component}:{message[:50]}"
            now = time.time()
            last_sent = self._system_alert_cache.get(cache_key, 0)
            if now - last_sent < SYSTEM_ALERT_DEDUP_MINUTES * 60:
                return

            if len(self._system_alert_cache) > 500:
                cutoff = now - SYSTEM_ALERT_DEDUP_MINUTES * 60
                self._system_alert_cache = {k: v for k, v in self._system_alert_cache.items() if v > cutoff}

            msg = self.format_system_alert(level, component, message)
            if self._system_bot and self._system_chat_id:
                self._run_async(self._system_bot.send_message(chat_id=self._system_chat_id, text=msg))
            self._system_alert_cache[cache_key] = now
            logger.info("System alert sent: [%s] %s — %s", level, component, message[:80])
        except Exception as exc:
            logger.error("send_system_alert failed: %s", exc, exc_info=True)

    def send_weekly_report(self, stats: Dict[str, Any]) -> None:
        """Send the weekly performance report to the SYSTEM channel."""
        try:
            msg = self.format_weekly_report(stats)
            if self._system_bot and self._system_chat_id:
                self._run_async(self._system_bot.send_message(chat_id=self._system_chat_id, text=msg))
            logger.info("Weekly report sent")
        except Exception as exc:
            logger.error("send_weekly_report failed: %s", exc, exc_info=True)
