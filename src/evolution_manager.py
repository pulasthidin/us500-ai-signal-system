"""
Evolution Manager — monitors signal count and automatically upgrades the
system through 4 stages as data accumulates.

Stage 0: < 200 signals  — rule-based only, collecting data silently
Stage 1: 200+ signals   — Model 3 trained, win probability on alerts
Stage 2: 500+ signals   — pattern_scanner activated (early warnings)
Stage 3: 1000+ signals  — daily retraining at midnight SL, max accuracy
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logger = logging.getLogger("evolution")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _fh = logging.FileHandler("logs/evolution.log", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)

STAGE_FILE = os.path.join("models", "stage.json")

STAGE_THRESHOLDS = {
    1: 200,
    2: 500,
    3: 1000,
}


class EvolutionManager:
    """Reads signal count and promotes the system through evolution stages."""

    def __init__(self, signal_logger, model_trainer, alert_bot=None) -> None:
        self._signal_logger = signal_logger
        self._model_trainer = model_trainer
        self._alert_bot = alert_bot

    # ─── stage persistence ───────────────────────────────────

    def _load_stage(self) -> Dict[str, Any]:
        try:
            if os.path.exists(STAGE_FILE):
                with open(STAGE_FILE) as f:
                    return json.load(f)
        except Exception as exc:
            logger.error("Failed to load stage file: %s", exc, exc_info=True)
        return {"stage": 0, "pattern_scanner_active": False, "daily_training_active": False}

    def _save_stage(self, data: Dict[str, Any]) -> None:
        """Write atomically: dump to .tmp then os.replace so a crash can never truncate."""
        tmp_path = STAGE_FILE + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, STAGE_FILE)
            logger.info("Stage file saved: %s", data)
        except Exception as exc:
            logger.error("Failed to save stage file: %s", exc, exc_info=True)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def get_current_stage(self) -> int:
        """Return the current evolution stage number (0-3)."""
        return self._load_stage().get("stage", 0)

    def get_stage_info(self) -> Dict[str, Any]:
        """Full stage metadata for reporting."""
        return self._load_stage()

    # ─── main check ──────────────────────────────────────────

    def check_evolution_stage(self) -> Dict[str, Any]:
        """
        Compare signal count against thresholds and activate the next stage
        if a threshold is crossed.  Never downgrades.
        """
        try:
            count = self._signal_logger.get_signal_count()
            state = self._load_stage()
            current_stage = state.get("stage", 0)

            if count >= STAGE_THRESHOLDS[1] and current_stage < 1:
                self._activate_stage_1(state, count)
                state = self._load_stage()
                current_stage = state.get("stage", 0)

            if count >= STAGE_THRESHOLDS[2] and current_stage < 2:
                self._activate_stage_2(state, count)
                state = self._load_stage()
                current_stage = state.get("stage", 0)

            if count >= STAGE_THRESHOLDS[3] and current_stage < 3:
                self._activate_stage_3(state, count)

            state = self._load_stage()
            state["signal_count"] = count
            return state

        except Exception as exc:
            logger.error("check_evolution_stage failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "evolution", str(exc))
            return {"stage": 0, "signal_count": 0}

    # ─── stage activations ───────────────────────────────────

    def _activate_stage_1(self, state: Dict, count: int) -> None:
        logger.info("Activating STAGE 1 — %d signals", count)

        try:
            success = self._model_trainer.train_meta_label_model()
        except Exception as exc:
            logger.error("Stage 1 model training failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "evolution", f"Stage 1 training error: {exc}")
            return

        if not success:
            logger.warning("Stage 1 training returned False — not activating")
            self._send_system_alert("WARNING", "evolution", "Stage 1 training did not produce a model")
            return

        state["stage"] = 1
        self._save_stage(state)

        self._send_system_alert("INFO", "evolution", (
            "\U0001f680 STAGE 1 ACTIVE\n"
            "200+ signals reached\n"
            "Model 3 trained successfully\n"
            "Win probability added to alerts\n"
            "Weekly retraining every Sunday"
        ))

    def _activate_stage_2(self, state: Dict, count: int) -> None:
        logger.info("Activating STAGE 2 — %d signals", count)
        state["stage"] = 2
        state["pattern_scanner_active"] = True
        self._save_stage(state)

        self._send_system_alert("INFO", "evolution", (
            "\U0001f680 STAGE 2 ACTIVE\n"
            "500+ signals reached\n"
            "Pattern scanner now running\n"
            "Finding YOUR patterns proactively\n"
            "Early warning alerts enabled"
        ))

    def _activate_stage_3(self, state: Dict, count: int) -> None:
        logger.info("Activating STAGE 3 — %d signals", count)
        state["stage"] = 3
        state["pattern_scanner_active"] = True
        state["daily_training_active"] = True
        self._save_stage(state)

        self._send_system_alert("INFO", "evolution", (
            "\U0001f680 STAGE 3 ACTIVE\n"
            "1000+ signals reached\n"
            "Switching to daily retraining\n"
            "Every night midnight SL\n"
            "Maximum accuracy mode enabled"
        ))

    # ─── Sunday evolution report ─────────────────────────────

    def run_evolution_check(self) -> Dict[str, Any]:
        """
        Called every Sunday after the weekly retrain.
        Checks for stage promotion and sends a status update.
        """
        try:
            result = self.check_evolution_stage()
            stage = result.get("stage", 0)
            count = result.get("signal_count", 0)

            next_threshold = None
            needed = 0
            for s in (1, 2, 3):
                if stage < s:
                    next_threshold = STAGE_THRESHOLDS[s]
                    needed = max(0, next_threshold - count)
                    break

            status_lines = [
                "\U0001f4ca EVOLUTION STATUS",
                f"Current stage: {stage}",
                f"Signals: {count}",
            ]
            if next_threshold:
                status_lines.append(f"Next stage at: {next_threshold} signals")
                status_lines.append(f"Need: {needed} more signals")
            else:
                status_lines.append("Maximum stage reached \u2705")

            self._send_system_alert("INFO", "evolution", "\n".join(status_lines))
            logger.info("Evolution check: stage=%d count=%d", stage, count)
            return result

        except Exception as exc:
            logger.error("run_evolution_check failed: %s", exc, exc_info=True)
            self._send_system_alert("WARNING", "evolution", str(exc))
            return {"stage": 0}

    # ─── helpers ─────────────────────────────────────────────

    def _send_system_alert(self, level: str, component: str, message: str) -> None:
        if self._alert_bot:
            try:
                self._alert_bot.send_system_alert(level, component, message)
            except Exception:
                logger.error("Failed to send system alert", exc_info=True)
