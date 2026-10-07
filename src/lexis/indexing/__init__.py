"""Indexing module for Lexis."""

from lexis.indexing.document_reader import DocumentReader, Document
from lexis.indexing.block_writer import BlockWriter
from lexis.indexing.index_merger import IndexMerger
from lexis.indexing.spimi import SPIMIIndexer, IndexingMetrics
from lexis.indexing.bsbi import BSBIIndexer
from lexis.indexing.shard_builder import ShardBuilder
from lexis.indexing.incremental import IncrementalIndexer
from lexis.indexing.deletion import DeletionManager

__all__ = [
    "DocumentReader",
    "Document",
    "BlockWriter",
    "IndexMerger",
    "SPIMIIndexer",
    "IndexingMetrics",
    "BSBIIndexer",
    "ShardBuilder",
    "IncrementalIndexer",
    "DeletionManager",
]
