"""Single-Pass In-Memory Indexing (SPIMI) implementation."""

import time
import psutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lexis.indexing.document_reader import DocumentReader
from lexis.indexing.block_writer import BlockWriter
from lexis.indexing.index_merger import IndexMerger
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList
from lexis.index.dictionary import DocumentRegistry
from lexis.preprocessing.tokenizer import Tokenizer


@dataclass
class IndexingMetrics:
    method: str
    num_blocks: int
    bytes_written: int
    peak_memory_mb: float
    build_time_seconds: float
    total_docs: int
    vocab_size: int


class SPIMIIndexer:
    """SPIMI indexer supporting configurable artificial memory limit and metrics collection."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        memory_limit_tokens: int = 50000,
        temp_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.memory_limit_tokens = memory_limit_tokens
        self.temp_dir = Path(temp_dir) if temp_dir else None

    def build_index(
        self,
        corpus_dir: str,
        output_dir: str,
        max_docs: Optional[int] = None,
    ) -> Tuple[PositionalIndex, IndexingMetrics]:
        """Build the inverted index using SPIMI and collect performance metrics."""
        start_time = time.time()
        process = psutil.Process()
        initial_mem = process.memory_info().rss / (1024 * 1024)
        peak_mem = initial_mem

        out_path = Path(output_dir)
        blocks_dir = (self.temp_dir or out_path) / "spimi_blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)

        partial_index: Dict[str, PostingsList] = {}
        block_files: List[Path] = []
        bytes_written = 0
        tokens_in_block = 0
        block_idx = 0

        registry = DocumentRegistry()
        doc_count = 0

        for doc in DocumentReader.stream_corpus(corpus_dir):
            if max_docs is not None and doc_count >= max_docs:
                break

            tokens = self.tokenizer.tokenize_document(title=doc.title, body=doc.body)
            title_len = sum(1 for t in tokens if t.zone == "title")
            body_len = len(tokens) - title_len

            doc_id = registry.register(
                doc_name=doc.doc_name,
                doc_length=len(tokens),
                title_length=title_len,
                body_length=body_len,
            )
            doc_count += 1

            for tok in tokens:
                if tok.term not in partial_index:
                    partial_index[tok.term] = PostingsList(term=tok.term)
                partial_index[tok.term].add_occurrence(
                    doc_id=doc_id,
                    position=tok.position,
                    zone=tok.zone,
                )
                tokens_in_block += 1

            # Check peak memory
            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_mem:
                peak_mem = current_mem

            # Flush block when memory limit reached
            if tokens_in_block >= self.memory_limit_tokens:
                block_file = blocks_dir / f"block_{block_idx:04d}.dat"
                written = BlockWriter.write_spimi_block(block_file, partial_index)
                bytes_written += written
                block_files.append(block_file)
                block_idx += 1
                partial_index.clear()
                tokens_in_block = 0

        # Flush remaining block if any
        if partial_index:
            block_file = blocks_dir / f"block_{block_idx:04d}.dat"
            written = BlockWriter.write_spimi_block(block_file, partial_index)
            bytes_written += written
            block_files.append(block_file)
            block_idx += 1
            partial_index.clear()

        # Merge blocks into final index
        final_index = IndexMerger.merge_spimi_blocks(
            block_files=block_files,
            output_dir=out_path,
            registry=registry,
        )

        build_time = time.time() - start_time
        current_mem = process.memory_info().rss / (1024 * 1024)
        if current_mem > peak_mem:
            peak_mem = current_mem

        metrics = IndexingMetrics(
            method="SPIMI",
            num_blocks=len(block_files),
            bytes_written=bytes_written,
            peak_memory_mb=round(peak_mem, 2),
            build_time_seconds=round(build_time, 3),
            total_docs=doc_count,
            vocab_size=final_index.dictionary.size(),
        )

        return final_index, metrics
