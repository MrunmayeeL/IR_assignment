"""Vector and document length normalization."""

import math
from typing import Dict


class LengthNormalizer:
    """Computes document and vector norms for cosine similarity."""

    @staticmethod
    def vector_norm(vec: Dict[str, float]) -> float:
        """Euclidean L2 norm of a sparse vector."""
        sum_sq = sum(v * v for v in vec.values())
        return math.sqrt(sum_sq) if sum_sq > 0 else 1.0

    @staticmethod
    def doc_length_norm(doc_len: int, enabled: bool = True) -> float:
        """Document length normalization factor."""
        if not enabled or doc_len <= 0:
            return 1.0
        return math.sqrt(doc_len)
