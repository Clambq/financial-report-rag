"""Latency aggregation with an explicit percentile definition."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """Return linear P50 and P95 latency values in milliseconds."""

    if not samples_ms:
        raise ValueError("latency samples must not be empty")
    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
    }
