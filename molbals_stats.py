import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

import psutil
import torch

import comfy.model_management
from comfy.comfy_types.node_typing import IO
from comfy_api.latest._caching import CacheProvider
from comfy_execution.cache_provider import register_cache_provider
from comfy_execution.utils import get_executing_context


LOGGER = logging.getLogger(__name__)
SAMPLE_INTERVAL_SECONDS = 0.05
MAX_FINISHED_PROMPTS = 64
BYTES_PER_MIB = 1024 * 1024


def _bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / BYTES_PER_MIB


def _round_optional(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _process_ram_bytes(process: psutil.Process) -> int:
    return int(process.memory_info().rss)


def _torch_device():
    try:
        return comfy.model_management.get_torch_device()
    except Exception:
        LOGGER.debug("Could not resolve ComfyUI torch device", exc_info=True)
        return None


def _device_name(device) -> str:
    if device is None:
        return "unavailable"

    device_type = getattr(device, "type", None)
    if device_type == "cuda" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(device)
        except Exception:
            return str(device)

    return str(device)


def _device_vram_used_bytes(device) -> Optional[int]:
    if device is None:
        return None

    device_type = getattr(device, "type", None)
    if device_type in (None, "cpu", "mps"):
        return None

    if device_type == "cuda" and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            return max(0, int(total_bytes - free_bytes))
        except Exception:
            LOGGER.debug("Could not read CUDA memory info", exc_info=True)

    try:
        total_bytes = int(comfy.model_management.get_total_memory(device))
        free_bytes = int(comfy.model_management.get_free_memory(device))
        return max(0, total_bytes - free_bytes)
    except Exception:
        LOGGER.debug("Could not read device memory info", exc_info=True)
        return None


class _PromptStatsSession:
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        self.started_at = time.perf_counter()
        self.finished_at: Optional[float] = None
        self.process = psutil.Process(os.getpid())
        self.device = _torch_device()
        self.device_name = _device_name(self.device)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        ram = _process_ram_bytes(self.process)
        vram = _device_vram_used_bytes(self.device)

        self.baseline_ram_bytes = ram
        self.peak_ram_bytes = ram
        self.baseline_vram_bytes = vram
        self.peak_vram_bytes = vram
        self.sample_count = 1

        self.thread = threading.Thread(
            target=self._run,
            name=f"molbals-stats-{prompt_id[:8]}",
            daemon=True,
        )
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(SAMPLE_INTERVAL_SECONDS):
            self.sample()

    def sample(self):
        ram = _process_ram_bytes(self.process)
        vram = _device_vram_used_bytes(self.device)

        with self.lock:
            self.sample_count += 1
            self.peak_ram_bytes = max(self.peak_ram_bytes, ram)
            if vram is not None:
                if self.peak_vram_bytes is None:
                    self.peak_vram_bytes = vram
                else:
                    self.peak_vram_bytes = max(self.peak_vram_bytes, vram)

    def finish(self):
        self.sample()
        with self.lock:
            if self.finished_at is None:
                self.finished_at = time.perf_counter()
        self.stop_event.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=0.25)

    def snapshot(self) -> dict:
        self.sample()
        with self.lock:
            end = self.finished_at or time.perf_counter()
            peak_ram = self.peak_ram_bytes
            peak_vram = self.peak_vram_bytes
            baseline_ram = self.baseline_ram_bytes
            baseline_vram = self.baseline_vram_bytes
            sample_count = self.sample_count
            finished = self.finished_at is not None

        vram_delta = None
        if peak_vram is not None and baseline_vram is not None:
            vram_delta = max(0, peak_vram - baseline_vram)

        return {
            "prompt_id": self.prompt_id,
            "device": self.device_name,
            "complete": finished,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "sample_count": sample_count,
            "total_seconds": round(end - self.started_at, 3),
            "peak_vram_mb": _round_optional(_bytes_to_mib(peak_vram)),
            "peak_vram_delta_mb": _round_optional(_bytes_to_mib(vram_delta)),
            "peak_ram_mb": _round_optional(_bytes_to_mib(peak_ram)),
            "peak_ram_delta_mb": _round_optional(_bytes_to_mib(max(0, peak_ram - baseline_ram))),
        }


_lock = threading.RLock()
_active_sessions: dict[str, _PromptStatsSession] = {}
_finished_sessions: OrderedDict[str, dict] = OrderedDict()


def _remember_finished(prompt_id: str, snapshot: dict):
    _finished_sessions[prompt_id] = snapshot
    _finished_sessions.move_to_end(prompt_id)
    while len(_finished_sessions) > MAX_FINISHED_PROMPTS:
        _finished_sessions.popitem(last=False)


def _get_snapshot(prompt_id: str) -> Optional[dict]:
    with _lock:
        session = _active_sessions.get(prompt_id)
        if session is not None:
            return session.snapshot()
        return _finished_sessions.get(prompt_id)


class _StatsLifecycleProvider(CacheProvider):
    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        return None

    def should_cache(self, context, value=None) -> bool:
        return False

    def on_prompt_start(self, prompt_id: str) -> None:
        with _lock:
            old_session = _active_sessions.pop(prompt_id, None)
            if old_session is not None:
                old_session.finish()
                _remember_finished(prompt_id, old_session.snapshot())
            _active_sessions[prompt_id] = _PromptStatsSession(prompt_id)

    def on_prompt_end(self, prompt_id: str) -> None:
        with _lock:
            session = _active_sessions.pop(prompt_id, None)
            if session is None:
                return
            session.finish()
            _remember_finished(prompt_id, session.snapshot())


class MolbalsStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (
                    IO.ANY,
                    {
                        "tooltip": "Connect this to the final generated value you want to wait for before reading stats.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("stats_json", "peak_vram_mb", "peak_ram_mb", "total_seconds")
    FUNCTION = "stats"
    OUTPUT_NODE = True
    CATEGORY = "utils/diagnostics"
    DESCRIPTION = "Reports elapsed generation time plus sampled peak process RAM and device VRAM for the current prompt."

    @classmethod
    def IS_CHANGED(cls, trigger=None):
        return float("NaN")

    def stats(self, trigger=None):
        context = get_executing_context()
        prompt_id = context.prompt_id if context is not None else None
        node_id = context.node_id if context is not None else None

        if prompt_id is None:
            payload = {
                "error": "No ComfyUI execution context is available for this node.",
                "peak_vram_mb": None,
                "peak_ram_mb": None,
                "total_seconds": None,
            }
        else:
            payload = _get_snapshot(prompt_id)
            if payload is None:
                payload = {
                    "prompt_id": prompt_id,
                    "node_id": node_id,
                    "error": "No active molbals-stats monitor was found for this prompt. Restart ComfyUI after installing the node.",
                    "peak_vram_mb": None,
                    "peak_ram_mb": None,
                    "total_seconds": None,
                }
            else:
                payload = dict(payload)
                payload["node_id"] = node_id

        stats_json = json.dumps(payload, indent=2, sort_keys=True)
        peak_vram = payload.get("peak_vram_mb")
        peak_ram = payload.get("peak_ram_mb")
        total_seconds = payload.get("total_seconds")

        return {
            "ui": {"text": (stats_json,)},
            "result": (
                stats_json,
                float(peak_vram) if peak_vram is not None else -1.0,
                float(peak_ram) if peak_ram is not None else -1.0,
                float(total_seconds) if total_seconds is not None else -1.0,
            ),
        }


_provider = _StatsLifecycleProvider()
register_cache_provider(_provider)

NODE_CLASS_MAPPINGS = {
    "MolbalsStats": MolbalsStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MolbalsStats": "molbals-stats",
}
