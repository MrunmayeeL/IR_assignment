"""Zone weighting calculations for title and body fields."""

from dataclasses import dataclass
from lexis.ranking.tfidf import TFIDFFactory


@dataclass
class ZoneWeights:
    title_weight: float = 0.3
    body_weight: float = 0.7
    enabled: bool = True

    def compute_weighted_tf(
        self,
        title_tf: int,
        body_tf: int,
        total_tf: int,
        use_log_tf: bool = True,
    ) -> float:
        """Compute zone-weighted term frequency."""
        if not self.enabled:
            return TFIDFFactory.log_tf(total_tf) if use_log_tf else TFIDFFactory.raw_tf(total_tf)

        tf_func = TFIDFFactory.log_tf if use_log_tf else TFIDFFactory.raw_tf
        w_title = self.title_weight * tf_func(title_tf) if title_tf > 0 else 0.0
        w_body = self.body_weight * tf_func(body_tf) if body_tf > 0 else 0.0
        return w_title + w_body
