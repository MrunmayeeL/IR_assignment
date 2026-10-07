#!/usr/bin/env python3
"""Execute a search query over the Lexis index."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexis.index.positional_index import PositionalIndex
from lexis.preprocessing.tokenizer import Tokenizer
from lexis.query.query_engine import QueryEngine
from lexis.ranking.scorer import RankedScorer


def main():
    parser = argparse.ArgumentParser(description="Query Lexis Search Engine")
    parser.add_argument("query", nargs="+", help="Query string")
    parser.add_argument("--index", default="outputs/index", help="Index directory")
    parser.add_argument("--type", choices=["ranked", "boolean"], default="ranked", help="Query type")
    parser.add_argument("--method", choices=["exhaustive", "champion_lists", "impact_ordering"], default="exhaustive", help="Candidate pruning method")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    query_str = " ".join(args.query)

    index = PositionalIndex(args.index)
    index.load(args.index)
    tokenizer = Tokenizer()

    if args.type == "boolean":
        engine = QueryEngine(index, tokenizer)
        results = engine.search_exact(query_str)
        print(f"[*] Exact Query: {query_str}")
        print(f"    Total matches: {len(results)}")
        for r in results[:args.top_k]:
            print(f"    - {r}")
        if len(results) > args.top_k:
            print(f"    ... and {len(results) - args.top_k} more.")
    else:
        scorer = RankedScorer(index, tokenizer)
        results, scored, lat_ms = scorer.score_query(query_str, top_k=args.top_k, method=args.method)
        print(f"[*] Ranked Query ({args.method}): {query_str}")
        print(f"    Candidates scored: {scored} in {lat_ms:.2f} ms")
        print(f"    Top {len(results)} results:")
        for rank, r in enumerate(results, start=1):
            print(f"    {rank:2d}. {r.doc_name} (score: {r.score:.6f})")


if __name__ == "__main__":
    main()
