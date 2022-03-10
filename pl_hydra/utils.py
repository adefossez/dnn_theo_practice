"""Utility functions.
"""
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import os


def averager(beta: float = 1):
    """
    Returns a single function that can be called to repeatidly obtain
    a running average from a dictionary of metrics.
    The callback will return the new averaged dict of metrics.

    `beta` is the decay parameter. If `beta == 1`, a regular running
    average is performed. If `beta < 1`, an exponential moving average
    is performed instead.
    """
    count = defaultdict(float)
    total = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, count
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            count[key] = count[key] * beta + weight
        return {key: tot / count[key] for key, tot in total.items()}
    return _update


@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp"):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)
