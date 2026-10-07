"""Reporting and CSV/JSON output generation utilities."""

import csv
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple


class ReportGenerator:
    """Exports benchmark and evaluation outputs into CSV and JSON tables."""

    @staticmethod
    def write_csv(filepath: str, fieldnames: List[str], rows: List[Dict[str, Any]]):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    @staticmethod
    def write_json(filepath: str, data: Any):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def fit_heaps_law(token_counts: List[int], vocab_sizes: List[int]) -> Tuple[float, float]:
        """Fit Heaps' Law: V = k * T^b using linear regression in log-log space: log(V) = log(k) + b*log(T)."""
        if len(token_counts) < 2:
            return 30.0, 0.5  # Typical defaults

        x = [math.log(max(1, t)) for t in token_counts]
        y = [math.log(max(1, v)) for v in vocab_sizes]
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den = sum((xi - x_mean) ** 2 for xi in x)
        b = num / den if den != 0 else 0.5
        log_k = y_mean - b * x_mean
        k = math.exp(log_k)
        return round(k, 2), round(b, 4)

    @staticmethod
    def fit_zipfs_law(ranks: List[int], frequencies: List[int]) -> Tuple[float, float]:
        """Fit Zipf's Law: f = c / r^alpha using log-log regression: log(f) = log(c) - alpha * log(r)."""
        if len(ranks) < 2:
            return 1000.0, 1.0

        x = [math.log(r) for r in ranks]
        y = [math.log(f) for f in frequencies]
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den = sum((xi - x_mean) ** 2 for xi in x)
        slope = num / den if den != 0 else -1.0
        alpha = -slope
        c = math.exp(y_mean - slope * x_mean)
        return round(c, 2), round(alpha, 4)
