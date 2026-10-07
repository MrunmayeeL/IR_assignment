"""Block writer for intermediate index blocks."""

import json
import zlib
from pathlib import Path
from typing import Dict, Any, List
from lexis.index.postings import PostingsList


class BlockWriter:
    """Writes sorted intermediate index blocks to disk during SPIMI / BSBI."""

    @staticmethod
    def write_spimi_block(block_path: Path, partial_index: Dict[str, PostingsList]) -> int:
        """Serialize and write a SPIMI partial index block to disk.
        
        Returns total bytes written.
        """
        block_path.parent.mkdir(parents=True, exist_ok=True)
        # Store sorted by term
        serialized_block = {
            term: partial_index[term].to_dict()
            for term in sorted(partial_index.keys())
        }
        raw_json = json.dumps(serialized_block).encode("utf-8")
        compressed = zlib.compress(raw_json, level=4)
        with open(block_path, "wb") as f:
            f.write(compressed)
        return len(compressed)

    @staticmethod
    def read_spimi_block(block_path: Path) -> Dict[str, PostingsList]:
        """Read and deserialize a SPIMI partial index block from disk."""
        with open(block_path, "rb") as f:
            compressed = f.read()
        raw_json = zlib.decompress(compressed).decode("utf-8")
        data = json.loads(raw_json)
        return {
            term: PostingsList.from_dict(plist_data)
            for term, plist_data in data.items()
        }

    @staticmethod
    def write_bsbi_block(block_path: Path, term_doc_pairs: List[tuple]) -> int:
        """Write sorted (term_id, doc_id, position, zone) tuples for BSBI block."""
        block_path.parent.mkdir(parents=True, exist_ok=True)
        raw_json = json.dumps(term_doc_pairs).encode("utf-8")
        compressed = zlib.compress(raw_json, level=4)
        with open(block_path, "wb") as f:
            f.write(compressed)
        return len(compressed)

    @staticmethod
    def read_bsbi_block(block_path: Path) -> List[list]:
        with open(block_path, "rb") as f:
            compressed = f.read()
        raw_json = zlib.decompress(compressed).decode("utf-8")
        return json.loads(raw_json)
