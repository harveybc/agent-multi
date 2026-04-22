"""Reusable callbacks shared by ioin plugins."""
from __future__ import annotations
import gc
import os
import time
from dataclasses import dataclass
from typing import Any
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
import tensorflow.keras.backend as K

class ReduceLROnPlateauWithCounter(ReduceLROnPlateau):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        if self.verbose:
            print(f"ReduceLROnPlateau patience counter: {self.patience_counter}")

class EarlyStoppingWithPatienceCounter(EarlyStopping):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patience_counter = 0
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.patience_counter = self.wait if self.wait > 0 else 0
        if self.verbose:
            print(f"EarlyStopping patience counter: {self.patience_counter}")

class ClearMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()


class MemoryUsageLogger(Callback):
    """Append RSS/VmHWM to a log file each epoch.

    Enabled by passing a path via config key `memory_log_file`.
    """
    def __init__(self, file_path: str, flush_every: int = 1, tag: str | None = None):
        super().__init__()
        self.file_path = file_path
        self.flush_every = max(1, int(flush_every))
        self.tag = tag or ""
        self._epoch_count = 0
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("ts,epoch,tag,VmRSS_kB,VmHWM_kB\n")

    @staticmethod
    def _read_status_kb(key: str) -> int | None:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(key + ":"):
                        parts = line.split()
                        return int(parts[1])  # kB
        except Exception:
            return None
        return None

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        rss = self._read_status_kb("VmRSS")
        hwm = self._read_status_kb("VmHWM")
        ts = time.time()
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(f"{ts:.3f},{epoch},{self.tag},{rss if rss is not None else ''},{hwm if hwm is not None else ''}\n")
            if (self._epoch_count % self.flush_every) == 0:
                f.flush()


@dataclass(frozen=True)
class ResourceSnapshot:
    ts: float
    rss_kb: int | None
    hwm_kb: int | None
    gpu_current_bytes: int | None
    gpu_peak_bytes: int | None
    gc_counts: tuple[int, int, int] | None


def _read_proc_status_kb(key: str) -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    return int(parts[1])  # kB
    except Exception:
        return None
    return None


def _read_gpu_mem_bytes() -> tuple[int | None, int | None]:
    """Best-effort GPU memory snapshot.

    Uses TF's get_memory_info when available; returns (current, peak) in bytes.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return (None, None)
        info = tf.config.experimental.get_memory_info("GPU:0")  # type: ignore[attr-defined]
        cur = int(info.get("current")) if isinstance(info, dict) and info.get("current") is not None else None
        peak = int(info.get("peak")) if isinstance(info, dict) and info.get("peak") is not None else None
        return (cur, peak)
    except Exception:
        return (None, None)


def capture_resource_snapshot(*, include_gpu: bool = True, include_gc: bool = True) -> ResourceSnapshot:
    ts = time.time()
    rss_kb = _read_proc_status_kb("VmRSS")
    hwm_kb = _read_proc_status_kb("VmHWM")
    gpu_current_bytes = None
    gpu_peak_bytes = None
    if include_gpu:
        gpu_current_bytes, gpu_peak_bytes = _read_gpu_mem_bytes()
    gc_counts = None
    if include_gc:
        try:
            gc_counts = gc.get_count()
        except Exception:
            gc_counts = None
    return ResourceSnapshot(
        ts=ts,
        rss_kb=rss_kb,
        hwm_kb=hwm_kb,
        gpu_current_bytes=gpu_current_bytes,
        gpu_peak_bytes=gpu_peak_bytes,
        gc_counts=gc_counts,
    )


class ResourceUsageLogger(Callback):
    """Append RSS/HWM (+ optional GPU + optional GC counts) each epoch.

    Designed for long GA runs where the kernel OOM killer provides no Python traceback.
    """
    def __init__(
        self,
        file_path: str,
        *,
        tag: str | None = None,
        flush_every: int = 1,
        include_gpu: bool = True,
        include_gc: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.tag = tag or ""
        self.flush_every = max(1, int(flush_every))
        self.include_gpu = bool(include_gpu)
        self.include_gc = bool(include_gc)
        self._epoch_count = 0
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    "ts,epoch,tag,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,gc0,gc1,gc2\n"
                )

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        snap = capture_resource_snapshot(include_gpu=self.include_gpu, include_gc=self.include_gc)
        gc0 = gc1 = gc2 = ""
        if snap.gc_counts is not None:
            gc0, gc1, gc2 = snap.gc_counts
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{snap.ts:.3f},{epoch},{self.tag},"
                f"{snap.rss_kb if snap.rss_kb is not None else ''},"
                f"{snap.hwm_kb if snap.hwm_kb is not None else ''},"
                f"{snap.gpu_current_bytes if snap.gpu_current_bytes is not None else ''},"
                f"{snap.gpu_peak_bytes if snap.gpu_peak_bytes is not None else ''},"
                f"{gc0},{gc1},{gc2}\n"
            )
            if (self._epoch_count % self.flush_every) == 0:
                f.flush()


class BatchResourceUsageLogger(Callback):
    """Append RSS/HWM (+ optional GPU) every N batches.

    This is the most reliable way to pinpoint *where inside model.fit* memory climbs
    immediately before a Linux OOM kill (the last written row is close to the death).
    """

    def __init__(
        self,
        file_path: str,
        *,
        tag: str | None = None,
        every_n_batches: int = 50,
        flush_every: int = 1,
        include_gpu: bool = True,
        include_gc: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.tag = tag or ""
        self.every_n_batches = max(1, int(every_n_batches))
        self.flush_every = max(1, int(flush_every))
        self.include_gpu = bool(include_gpu)
        self.include_gc = bool(include_gc)
        self._row_count = 0
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    "ts,epoch,batch,tag,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,gc0,gc1,gc2\n"
                )

    def on_train_batch_end(self, batch, logs=None):
        # Keras gives batch index within epoch.
        if ((int(batch) + 1) % self.every_n_batches) != 0:
            return
        self._row_count += 1
        snap = capture_resource_snapshot(include_gpu=self.include_gpu, include_gc=self.include_gc)
        gc0 = gc1 = gc2 = ""
        if snap.gc_counts is not None:
            gc0, gc1, gc2 = snap.gc_counts
        epoch = getattr(self, "_current_epoch", "")
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{snap.ts:.3f},{epoch},{int(batch)},{self.tag},"
                f"{snap.rss_kb if snap.rss_kb is not None else ''},"
                f"{snap.hwm_kb if snap.hwm_kb is not None else ''},"
                f"{snap.gpu_current_bytes if snap.gpu_current_bytes is not None else ''},"
                f"{snap.gpu_peak_bytes if snap.gpu_peak_bytes is not None else ''},"
                f"{gc0},{gc1},{gc2}\n"
            )
            if (self._row_count % self.flush_every) == 0:
                f.flush()

    def on_epoch_begin(self, epoch, logs=None):
        # Track epoch for batch-level rows.
        self._current_epoch = int(epoch)


class ResourceGuard(Callback):
    """Abort training before the OS kills the process.

    Set `max_rss_gb` (decimal GB) or `max_rss_mb` (decimal MB) in config to enable.

    Notes on units:
      - Linux `/proc/self/status` reports VmRSS/VmHWM in "kB" but the values are KiB.
      - `max_rss_gb` is interpreted as decimal GB (1 GB = 1e9 bytes) to match typical user expectations.
      - If you want binary units, use `max_rss_gib` / `max_rss_mib`.
    """
    def __init__(
        self,
        *,
        max_rss_mb: int | None = None,
        max_rss_gb: float | None = None,
        max_rss_mib: int | None = None,
        max_rss_gib: float | None = None,
        include_gpu: bool = True,
        print_every: int = 1,
        check_every_batches: int | None = None,
    ):
        super().__init__()

        if all(v is None for v in (max_rss_mb, max_rss_gb, max_rss_mib, max_rss_gib)):
            raise ValueError(
                "ResourceGuard requires one of max_rss_mb/max_rss_gb/max_rss_mib/max_rss_gib"
            )

        # /proc reports KiB. Convert the user-provided limit into KiB.
        # Prefer explicit binary units when provided.
        if max_rss_mib is not None:
            self.max_rss_kib = int(max_rss_mib) * 1024
            self._limit_label = f"{max_rss_mib} MiB"
        elif max_rss_gib is not None:
            self.max_rss_kib = int(float(max_rss_gib) * 1024 * 1024)
            self._limit_label = f"{max_rss_gib} GiB"
        elif max_rss_mb is not None:
            # decimal MB -> bytes -> KiB
            self.max_rss_kib = int((float(max_rss_mb) * 1_000_000) / 1024)
            self._limit_label = f"{max_rss_mb} MB"
        else:
            # decimal GB -> bytes -> KiB
            self.max_rss_kib = int((float(max_rss_gb) * 1_000_000_000) / 1024)
            self._limit_label = f"{max_rss_gb} GB"

        self.include_gpu = bool(include_gpu)
        self.print_every = max(1, int(print_every))
        self.check_every_batches = None if check_every_batches is None else max(1, int(check_every_batches))
        self._epoch_count = 0
        self._batch_count = 0

    @staticmethod
    def _fmt_gib_from_kib(kib: int | None) -> str:
        if kib is None:
            return ""
        return f"{(kib / 1024 / 1024):.2f} GiB"

    def _check_and_maybe_abort(self, *, where: str, epoch: int | None = None, batch: int | None = None) -> None:
        rss_kib = _read_proc_status_kb("VmRSS")
        if rss_kib is None:
            return
        if rss_kib >= self.max_rss_kib:
            raise RuntimeError(
                "ResourceGuard abort: "
                f"VmRSS={rss_kib} KiB ({self._fmt_gib_from_kib(rss_kib)}) "
                f">= limit={self.max_rss_kib} KiB ({self._fmt_gib_from_kib(self.max_rss_kib)}) "
                f"(configured as {self._limit_label}) at {where}"
                + (f" epoch={epoch}" if epoch is not None else "")
                + (f" batch={batch}" if batch is not None else "")
            )

    def on_train_batch_end(self, batch, logs=None):
        if self.check_every_batches is None:
            return
        self._batch_count += 1
        # Use batch index cadence for deterministic sampling within each epoch.
        if ((int(batch) + 1) % self.check_every_batches) != 0:
            return
        self._check_and_maybe_abort(where="batch_end", batch=int(batch))

    def on_train_begin(self, logs=None):
        # Catch cases where we already start above the limit.
        self._check_and_maybe_abort(where="train_begin")

    def on_epoch_begin(self, epoch, logs=None):
        # Catch large jumps that occur between epochs (e.g., caching / graph build effects).
        self._check_and_maybe_abort(where="epoch_begin", epoch=int(epoch))

    def on_train_batch_begin(self, batch, logs=None):
        # Critical: batch 0 often triggers big internal allocations (workspace/autotune).
        if self.check_every_batches is None:
            if int(batch) == 0:
                self._check_and_maybe_abort(where="batch_begin", batch=int(batch))
            return
        if int(batch) == 0 or ((int(batch) + 1) % self.check_every_batches) == 0:
            self._check_and_maybe_abort(where="batch_begin", batch=int(batch))

    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        rss_kib = _read_proc_status_kb("VmRSS")
        if (self._epoch_count % self.print_every) == 0:
            gpu_cur, gpu_peak = _read_gpu_mem_bytes() if self.include_gpu else (None, None)
            print(
                f"[RESOURCE] epoch={epoch} VmRSS_kB={rss_kib} limit_kB={self.max_rss_kib} "
                f"VmRSS_GiB={self._fmt_gib_from_kib(rss_kib)} limit_GiB={self._fmt_gib_from_kib(self.max_rss_kib)} "
                f"limit_config={self._limit_label} "
                f"gpu_current_B={gpu_cur} gpu_peak_B={gpu_peak}"
            )
        self._check_and_maybe_abort(where="epoch_end", epoch=int(epoch))

__all__ = [
    'ReduceLROnPlateauWithCounter',
    'EarlyStoppingWithPatienceCounter',
    'ClearMemoryCallback',
    'MemoryUsageLogger',
    'ResourceUsageLogger',
    'BatchResourceUsageLogger',
    'ResourceGuard',
    'capture_resource_snapshot',
    'ResourceSnapshot',
]
