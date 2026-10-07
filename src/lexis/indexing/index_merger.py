"""Multi-way index merger for partial index blocks."""

import heapq
import json
import zlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList, Posting
from lexis.index.dictionary import DocumentRegistry


class IndexMerger:
    """Merges multiple partial index blocks into a unified final PositionalIndex."""

    @classmethod
    def merge_spimi_blocks(
        cls,
        block_files: List[Path],
        output_dir: Path,
        registry: DocumentRegistry,
    ) -> PositionalIndex:
        """Merge multiple SPIMI block files into a final PositionalIndex."""
        output_dir.mkdir(parents=True, exist_ok=True)
        final_index = PositionalIndex(str(output_dir))
        final_index.document_registry = registry

        if not block_files:
            final_index.save(str(output_dir))
            return final_index

        # For efficient k-way merge without high memory usage, load term dictionaries
        # Each block is a dict of term -> PostingsList
        from lexis.indexing.block_writer import BlockWriter

        # Load blocks into memory or term queues
        blocks_data = [BlockWriter.read_spimi_block(bf) for bf in block_files]
        all_terms = sorted(set().union(*[b.keys() for b in blocks_data]))

        postings_path = output_dir / "postings.dat"
        offset = 0

        with open(postings_path, "wb") as pf:
            for term in all_terms:
                merged_postings: List[Posting] = []
                for b in blocks_data:
                    if term in b:
                        merged_postings.extend(b[term].postings)

                # Ensure doc_ids are strictly sorted
                merged_postings.sort(key=lambda p: p.doc_id)

                merged_plist = PostingsList(term=term, doc_frequency=len(merged_postings), postings=merged_postings)
                serialized = json.dumps(merged_plist.to_dict()).encode("utf-8")
                compressed = zlib.compress(serialized, level=6)
                length = len(compressed)

                pf.write(compressed)

                total_tf = sum(p.tf for p in merged_plist.postings)
                final_index.dictionary.add_or_update(
                    term=term,
                    doc_frequency=len(merged_postings),
                    offset=offset,
                    length=length,
                    total_tf=total_tf,
                )
                offset += length

        # Save metadata
        meta = {
            "dictionary": final_index.dictionary.to_dict(),
            "registry": final_index.document_registry.to_dict(),
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return final_index
