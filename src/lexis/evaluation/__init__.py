"""Evaluation and metrics module for Lexis."""

from lexis.evaluation.ranking_metrics import RankingMetrics
from lexis.evaluation.precision_recall import PrecisionRecallCurve
from lexis.evaluation.significance import SignificanceTester, SignificanceResult
from lexis.evaluation.boolean_eval import BooleanEvaluatorRunner, BooleanTestResult
from lexis.evaluation.reports import ReportGenerator

__all__ = [
    "RankingMetrics",
    "PrecisionRecallCurve",
    "SignificanceTester",
    "SignificanceResult",
    "BooleanEvaluatorRunner",
    "BooleanTestResult",
    "ReportGenerator",
]
