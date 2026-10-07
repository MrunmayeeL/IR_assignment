"""Unit tests for ranked retrieval, TF-IDF, zones, pruning, and metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.ranking.scorer import RankedScorer
from lexis.evaluation.ranking_metrics import RankingMetrics


def create_ranked_index():
    index = PositionalIndex()
    reg = index.document_registry
    d1 = reg.register("doc_01", doc_length=20, title_length=5, body_length=15)
    d2 = reg.register("doc_02", doc_length=50, title_length=5, body_length=45)
    d3 = reg.register("doc_03", doc_length=10, title_length=2, body_length=8)

    # doc_01: oil (in title and body)
    index.add_term_occurrence("oil", d1, 0, "title")
    index.add_term_occurrence("oil", d1, 5, "body")
    index.add_term_occurrence("prices", d1, 6, "body")

    # doc_02: oil (in body only, multiple times)
    index.add_term_occurrence("oil", d2, 10, "body")
    index.add_term_occurrence("oil", d2, 20, "body")

    # doc_03: wheat, corn
    index.add_term_occurrence("wheat", d3, 0, "body")
    index.add_term_occurrence("corn", d3, 1, "body")

    return index


def test_ranked_scorer_top_results():
    index = create_ranked_index()
    tok = Tokenizer()
    scorer = RankedScorer(index, tok, title_weight=0.5, body_weight=0.5)

    results, scored_count, _ = scorer.score_query("oil prices", top_k=2)
    assert len(results) > 0
    # doc_01 has both 'oil' and 'prices', plus 'oil' in title
    assert results[0].doc_name == "doc_01"
    assert scored_count >= 2


def test_ranking_metrics():
    retrieved = ["doc_01", "doc_02", "doc_03"]
    relevant = {"doc_01", "doc_03"}
    grades = {"doc_01": 2, "doc_02": 0, "doc_03": 1}

    # Precision@2
    p2 = RankingMetrics.precision_at_k(retrieved, relevant, k=2)
    assert p2 == 0.5

    # Average Precision
    # Rank 1: doc_01 (rel) -> P@1 = 1/1 = 1.0
    # Rank 2: doc_02 (not rel)
    # Rank 3: doc_03 (rel) -> P@3 = 2/3 = 0.6667
    # AP = (1.0 + 2/3) / 2 = 5/6 = 0.8333
    ap = RankingMetrics.average_precision(retrieved, relevant)
    assert round(ap, 3) == 0.833

    # NDCG@2
    ndcg2 = RankingMetrics.ndcg_at_k(retrieved, grades, k=2)
    assert ndcg2 > 0.0
