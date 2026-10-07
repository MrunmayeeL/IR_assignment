"""Text normalizer for Lexis.

Applies case-folding and character normalization consistently to both
documents and queries.
"""

from typing import Optional


class Normalizer:
    def __init__(self, case_folding: bool = True):
        self.case_folding = case_folding

    def normalize(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        if self.case_folding:
            text = text.lower()
        return text.strip()
