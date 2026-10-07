"""TF-IDF weighting formulations."""

import math
from typing import Dict, List


class TFIDFFactory:
    """Computes TF, IDF, and combined TF-IDF weights."""

    @staticmethod
    def log_tf(tf: int) -> float:
        """Log-frequency weighting: 1 + log10(tf) if tf > 0 else 0."""
        return 1.0 + math.log10(tf) if tf > 0 else 0.0

    @staticmethod
    def raw_tf(tf: int) -> float:
        """Raw term frequency."""
        return float(tf)

    @staticmethod
    def idf(total_docs: int, doc_freq: int) -> float:
        """Inverse document frequency: log10(N / df) if df > 0 else 0."""
        if doc_freq <= 0 or total_docs <= 0:
            return 0.0
        return math.log10(total_docs / doc_freq)

    @staticmethod
    def compute_query_vector(
        query_terms: List[str],
        total_docs: int,
        doc_freqs: Dict[str, int],
        use_idf: bool = True,
        use_log_tf: bool = True,
    ) -> Dict[str, float]:
        """Compute query term weight vector."""
        q_tf: Dict[str, int] = {}
        for t in query_terms:
            q_tf[t] = q_tf.get(t, 0) + 1

        q_vec: Dict[str, float] = {}
        for term, tf in q_tf.items():
            tf_val = TFIDFFactory.log_tf(tf) if use_log_tf else TFIDFFactory.raw_tf(tf)
            idf_val = TFIDFFactory.idf(total_docs, doc_freqs.get(term, 0)) if use_idf else 1.0
            q_vec[term] = tf_val * idf_val

        return q_vec
