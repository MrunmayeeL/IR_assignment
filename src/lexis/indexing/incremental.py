"""Incremental index updates without full rebuild."""

import json
import zlib
from pathlib import Path
from typing import List, Optional
from lexis.indexing.document_reader import Document
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList, Posting
from lexis.preprocessing.tokenizer import Tokenizer


class IncrementalIndexer:
    """Supports adding new documents dynamically to an existing index."""

    def __init__(self, index: PositionalIndex, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer

    def add_document(self, doc: Document) -> int:
        """Add a single document to the index without a full rebuild.
        
        Returns the registered integer doc_id.
        """
        tokens = self.tokenizer.tokenize_document(title=doc.title, body=doc.body)
        title_len = sum(1 for t in tokens if t.zone == "title")
        body_len = len(tokens) - title_len

        doc_id = self.index.document_registry.register(
            doc_name=doc.doc_name,
            doc_length=len(tokens),
            title_length=title_len,
            body_length=body_len,
        )

        for tok in tokens:
            plist = self.index.get_postings(tok.term)
            if plist is None:
                plist = PostingsList(term=tok.term)
            plist.add_occurrence(doc_id=doc_id, position=tok.position, zone=tok.zone)
            self.index._postings_cache[tok.term] = plist

        return doc_id

    def add_documents(self, docs: List[Document]) -> List[int]:
        """Add multiple documents in batch."""
        return [self.add_document(d) for d in docs]

    def commit(self, output_dir: Optional[str] = None):
        """Persist incremental updates to disk."""
        target_dir = output_dir or str(self.index.index_dir)
        # Pull any disk-only terms into cache before saving
        for term in self.index.dictionary.vocabulary():
            if term not in self.index._postings_cache:
                p = self.index.get_postings(term)
                if p:
                    self.index._postings_cache[term] = p
        self.index.save(target_dir)
