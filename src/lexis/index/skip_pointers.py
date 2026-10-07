"""Skip pointer implementation and instrumentation for postings lists.

Supports configurable skip spacing and detailed instrumentation counters
to evaluate comparison efficiency during Boolean intersection.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SkipStats:
    comparisons: int = 0
    skips_taken: int = 0
    postings_visited: int = 0


@dataclass
class SkipPointer:
    target_idx: int
    target_doc_id: int


class SkipList:
    def __init__(self, doc_ids: List[int], skip_spacing: Optional[int] = None):
        """Construct a postings list with skip pointers.
        
        If skip_spacing is None, default to floor(sqrt(len(doc_ids))).
        """
        self.doc_ids = doc_ids
        n = len(doc_ids)
        if skip_spacing is None:
            self.spacing = max(1, int(math.floor(math.sqrt(n))))
        else:
            self.spacing = max(1, skip_spacing)

        # Build skip pointers: map from index -> SkipPointer
        self.skips: dict[int, SkipPointer] = {}
        if self.spacing > 1 and n > self.spacing:
            for i in range(0, n - self.spacing, self.spacing):
                target_idx = i + self.spacing
                if target_idx < n:
                    self.skips[i] = SkipPointer(target_idx=target_idx, target_doc_id=doc_ids[target_idx])

    def has_skip(self, idx: int) -> bool:
        return idx in self.skips

    def get_skip(self, idx: int) -> Optional[SkipPointer]:
        return self.skips.get(idx)


def intersect_without_skips(p1: List[int], p2: List[int]) -> Tuple[List[int], SkipStats]:
    """Standard merge intersection of two sorted postings lists without skips."""
    result: List[int] = []
    stats = SkipStats()
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        stats.comparisons += 1
        stats.postings_visited += 2
        d1 = p1[i]
        d2 = p2[j]
        if d1 == d2:
            result.append(d1)
            i += 1
            j += 1
        elif d1 < d2:
            i += 1
        else:
            j += 1

    return result, stats


def intersect_with_skips(
    p1: List[int],
    p2: List[int],
    spacing1: Optional[int] = None,
    spacing2: Optional[int] = None,
) -> Tuple[List[int], SkipStats]:
    """Merge intersection using skip pointers (Manning et al., Figure 2.10)."""
    skip_list1 = SkipList(p1, skip_spacing=spacing1)
    skip_list2 = SkipList(p2, skip_spacing=spacing2)

    result: List[int] = []
    stats = SkipStats()
    i = 0
    j = 0
    len1 = len(p1)
    len2 = len(p2)

    while i < len1 and j < len2:
        stats.comparisons += 1
        stats.postings_visited += 2
        d1 = p1[i]
        d2 = p2[j]

        if d1 == d2:
            result.append(d1)
            i += 1
            j += 1
        elif d1 < d2:
            # Check if p1 has a skip pointer and target doc_id <= d2
            skip = skip_list1.get_skip(i)
            if skip is not None and skip.target_doc_id <= d2:
                stats.skips_taken += 1
                while skip is not None and skip.target_doc_id <= d2:
                    i = skip.target_idx
                    skip = skip_list1.get_skip(i)
            else:
                i += 1
        else:
            # Check if p2 has a skip pointer and target doc_id <= d1
            skip = skip_list2.get_skip(j)
            if skip is not None and skip.target_doc_id <= d1:
                stats.skips_taken += 1
                while skip is not None and skip.target_doc_id <= d1:
                    j = skip.target_idx
                    skip = skip_list2.get_skip(j)
            else:
                j += 1

    return result, stats
