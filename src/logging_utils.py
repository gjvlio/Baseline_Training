"""
Rich terminal logging for training: run headers, live per-batch progress with
throughput + ETA + GPU memory, epoch summary lines, per-class accuracy, and a
file mirror so every run is also written to checkpoints/<run>.log.
"""
import sys
import time
from datetime import datetime, timedelta

import torch


# ---------------------------------------------------------------------------
# basic ANSI (no external deps); disabled if not a tty
# ---------------------------------------------------------------------------
_TTY = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s):  return _c("1", s)
def dim(s):   return _c("2", s)
def green(s): return _c("32", s)
def yellow(s):return _c("33", s)
def cyan(s):  return _c("36", s)
def red(s):   return _c("31", s)


class Logger:
    """stdout + file mirror with timestamps."""

    def __init__(self, log_path=None):
        self.fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def log(self, msg=""):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{dim(ts)}  {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(f"{ts}  {_strip(msg)}\n")
            self.fh.flush()

    def raw(self, msg=""):
        print(msg, flush=True)
        if self.fh:
            self.fh.write(_strip(msg) + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def _strip(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", str(s))


# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------
def gpu_mem_str():
    if not torch.cuda.is_available():
        return "cpu"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.1f}/{reserved:.1f}/{total:.1f}GB used/resv/total"


def run_header(logger, title, cfg_dict, extra=None):
    bar = "=" * 72
    logger.raw(bold(cyan(bar)))
    logger.raw(bold(cyan(f"  {title}")))
    logger.raw(bold(cyan(bar)))
    logger.log(f"start    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        logger.log(f"device   : {torch.cuda.get_device_name(0)}  ({gpu_mem_str()})")
    else:
        logger.log("device   : CPU")
    logger.log("config   :")
    for k, v in cfg_dict.items():
        logger.raw(f"             {k:<18}= {v}")
    if extra:
        for line in extra:
            logger.log(line)
    logger.raw("")


# ---------------------------------------------------------------------------
# live progress meter
# ---------------------------------------------------------------------------
class ProgressMeter:
    """In-place batch progress with throughput + ETA + running loss/acc."""

    def __init__(self, total_batches, epoch, max_epochs, phase="train", every=1):
        self.total = total_batches
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.phase = phase
        self.every = every
        self.t0 = time.time()
        self.n = 0

    def update(self, batch_idx, loss=None, acc=None, extra=""):
        self.n = batch_idx + 1
        if self.n % self.every and self.n != self.total:
            return
        elapsed = time.time() - self.t0
        rate = self.n / max(elapsed, 1e-6)
        remaining = (self.total - self.n) / max(rate, 1e-6)
        pct = self.n / self.total
        barw = 24
        fill = int(barw * pct)
        bar = "#" * fill + "-" * (barw - fill)
        msg = (f"\r  [{self.epoch:>3}/{self.max_epochs}] {self.phase:<5} "
               f"|{bar}| {self.n:>4}/{self.total} "
               f"{rate:4.1f}b/s ETA {_fmt(remaining)}")
        if loss is not None:
            msg += f"  loss {loss:.4f}"
        if acc is not None:
            msg += f"  acc {acc:.3f}"
        if extra:
            msg += f"  {extra}"
        if torch.cuda.is_available():
            msg += f"  mem {torch.cuda.memory_allocated()/1024**3:.1f}G"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        return time.time() - self.t0


def _fmt(sec):
    return str(timedelta(seconds=int(sec)))


# ---------------------------------------------------------------------------
# per-class accuracy table
# ---------------------------------------------------------------------------
def per_class_report(logger, confusion, class_names):
    """confusion: [C,C] tensor (rows=true, cols=pred)."""
    logger.raw(dim("    per-class accuracy:"))
    diag = confusion.diag()
    row_sum = confusion.sum(1).clamp(min=1)
    for i, name in enumerate(class_names):
        acc = (diag[i] / row_sum[i]).item()
        n = int(row_sum[i].item())
        logger.raw(f"      {name:<10} {acc:5.3f}  (n={n})")
