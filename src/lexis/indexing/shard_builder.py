"""Parallel shard builder for multi-core index construction."""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict
from lexis.indexing.document_reader import DocumentReader, Document
from lexis.index.positional_index import PositionalIndex
from lexis.index.postings import PostingsList, Posting
from lexis.index.dictionary import DocumentRegistry
from lexis.preprocessing.tokenizer import Tokenizer


def _index_shard_worker(args) -> Tuple[int, Dict[str, dict], dict]:
    """Worker function to build partial postings for a list of document files."""
    shard_id, doc_paths_str, case_folding, stemming, stopword_removal = args
    from lexis.preprocessing.normalizer import Normalizer
    from lexis.preprocessing.stemmer import PorterStemmer
    from lexis.preprocessing.stopwords import StopwordFilter
    from lexis.preprocessing.tokenizer import Tokenizer
    from lexis.indexing.document_reader import DocumentReader

    tok = Tokenizer(
        normalizer=Normalizer(case_folding=case_folding),
        stemmer=PorterStemmer(enabled=stemming),
        stopword_filter=StopwordFilter(enabled=stopword_removal),
    )

    shard_postings: Dict[str, PostingsList] = {}
    shard_registry = DocumentRegistry()

    for p_str in doc_paths_str:
        p = Path(p_str)
        doc = DocumentReader.parse_file(p)
        tokens = tok.tokenize_document(title=doc.title, body=doc.body)
        title_len = sum(1 for t in tokens if t.zone == "title")
        body_len = len(tokens) - title_len

        doc_id = shard_registry.register(
            doc_name=doc.doc_name,
            doc_length=len(tokens),
            title_length=title_len,
            body_length=body_len,
        )

        for t in tokens:
            if t.term not in shard_postings:
                shard_postings[t.term] = PostingsList(term=t.term)
            shard_postings[t.term].add_occurrence(doc_id=doc_id, position=t.position, zone=t.zone)

    # Return serialized dictionaries
    serialized_postings = {term: plist.to_dict() for term, plist in shard_postings.items()}
    return shard_id, serialized_postings, shard_registry.to_dict()


class ShardBuilder:
    """Manages parallel index construction across multiple shards."""

    def __init__(self, tokenizer: Tokenizer, num_shards: int = 4):
        self.tokenizer = tokenizer
        self.num_shards = num_shards

    def build_sharded_index(self, corpus_dir: str, output_dir: str) -> Tuple[PositionalIndex, float]:
        """Build index across exactly num_shards worker processes in parallel."""
        start_time = time.time()
        files = DocumentReader.get_document_files(corpus_dir)
        total_files = len(files)

        if total_files == 0:
            final_index = PositionalIndex(output_dir)
            final_index.save(output_dir)
            return final_index, 0.0

        num_shards = min(self.num_shards, total_files)
        # Partition files round-robin or chunk
        shards_files: List[List[str]] = [[] for _ in range(num_shards)]
        for i, f in enumerate(files):
            shards_files[i % num_shards].append(str(f))

        worker_args = [
            (
                s_id,
                shards_files[s_id],
                self.tokenizer.normalizer.case_folding,
                self.tokenizer.stemmer.enabled,
                self.tokenizer.stopword_filter.enabled,
            )
            for s_id in range(num_shards)
        ]

        with ProcessPoolExecutor(max_workers=num_shards) as executor:
            shard_results = list(executor.map(_index_shard_worker, worker_args))

        # Merge shards into final unified index with consistent global doc_ids
        final_index = PositionalIndex(output_dir)
        global_registry = DocumentRegistry()
        global_postings: Dict[str, PostingsList] = {}

        for shard_id, serialized_postings, reg_dict in shard_results:
            shard_reg = DocumentRegistry.from_dict(reg_dict)
            # Map local shard doc_id -> global doc_id
            local_to_global: Dict[int, int] = {}
            for local_id, doc_name in shard_reg.doc_id_to_name.items():
                d_len = shard_reg.doc_lengths.get(local_id, 0)
                t_len = shard_reg.title_lengths.get(local_id, 0)
                b_len = shard_reg.body_lengths.get(local_id, 0)
                g_id = global_registry.register(doc_name, doc_length=d_len, title_length=t_len, body_length=b_len)
                local_to_global[local_id] = g_id

            # Remap postings to global doc_ids
            for term, plist_dict in serialized_postings.items():
                if term not in global_postings:
                    global_postings[term] = PostingsList(term=term)

                target_plist = global_postings[term]
                for p_data in plist_dict.get("postings", []):
                    old_doc_id = p_data["doc_id"]
                    new_doc_id = local_to_global[old_doc_id]
                    p = Posting(
                        doc_id=new_doc_id,
                        tf=p_data["tf"],
                        positions=p_data.get("positions", []),
                        title_tf=p_data.get("title_tf", 0),
                        body_tf=p_data.get("body_tf", 0),
                    )
                    target_plist.postings.append(p)

        # Sort all postings by global doc_id
        for term, plist in global_postings.items():
            plist.postings.sort(key=lambda p: p.doc_id)
            plist.doc_frequency = len(plist.postings)

        final_index._postings_cache = global_postings
        final_index.document_registry = global_registry
        final_index.save(output_dir)

        total_time = time.time() - start_time
        return final_index, total_time
