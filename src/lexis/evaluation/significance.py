"""Paired statistical significance testing for ablation experiments."""

import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SignificanceResult:
    full_map: float
    ablated_map: float
    delta_map: float
    t_statistic: float
    p_value: float
    is_significant_05: bool


class SignificanceTester:
    """Performs paired t-test between per-query Average Precision scores."""

    @staticmethod
    def paired_t_test(
        full_scores: Dict[str, float],
        ablated_scores: Dict[str, float],
    ) -> SignificanceResult:
        """Compare per-query AP values using paired Student's t-test."""
        common_qids = sorted(set(full_scores.keys()).intersection(ablated_scores.keys()))
        if not common_qids:
            return SignificanceResult(0.0, 0.0, 0.0, 0.0, 1.0, False)

        x = [full_scores[q] for q in common_qids]
        y = [ablated_scores[q] for q in common_qids]
        n = len(x)

        full_map = sum(x) / float(n)
        ablated_map = sum(y) / float(n)
        delta_map = ablated_map - full_map

        if HAS_SCIPY:
            t_stat, p_val = stats.ttest_rel(x, y)
            if math.isnan(p_val):
                p_val = 1.0
                t_stat = 0.0
        else:
            # Fallback pure python paired t-test
            diffs = [a - b for a, b in zip(x, y)]
            mean_diff = sum(diffs) / float(n)
            var_diff = sum((d - mean_diff) ** 2 for d in diffs) / float(max(1, n - 1))
            se_diff = math.sqrt(var_diff / float(n)) if var_diff > 0 else 0.0
            t_stat = (mean_diff / se_diff) if se_diff > 0 else 0.0
            # Approximate two-tailed p-value
            df = n - 1
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

        return SignificanceResult(
            full_map=round(full_map, 4),
            ablated_map=round(ablated_map, 4),
            delta_map=round(delta_map, 4),
            t_statistic=round(float(t_stat), 4),
            p_value=round(float(p_val), 5),
            is_significant_05=bool(p_val < 0.05),
        )
