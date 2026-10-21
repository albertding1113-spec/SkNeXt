"""Timestamped console messages and reporting intervals for workflows."""

from datetime import datetime


def time_str() -> str:
    """Return local time in the shared [YYYY.MM.DD||HH:MM:SS] format."""
    return datetime.now().strftime("[%Y.%m.%d||%H:%M:%S]")


def log_message(message, *, flush: bool = True) -> None:
    """Print a timestamp on every line, flushing progress immediately by default."""
    timestamp = time_str()
    for line in str(message).splitlines() or [""]:
        print(f"{timestamp} {line}", flush=flush)


def loss_log_steps(total_steps: int, updates: int = 20) -> set[int]:
    """Return evenly spaced batch numbers, including the final batch.

    Short epochs report every batch. Longer epochs report exactly ``updates``
    times, even when the batch count is not divisible by that number.
    """
    if total_steps <= 0:
        return set()
    count = min(total_steps, updates)
    if count <= 0:
        raise ValueError("updates must be positive")
    return {(i * total_steps + count - 1) // count for i in range(1, count + 1)}
