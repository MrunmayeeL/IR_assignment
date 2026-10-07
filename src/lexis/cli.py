"""Unified Command-Line Interface (CLI) for Lexis."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="lexis",
        description="Lexis Search Engine CLI (Winter 2026)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # build-index
    build_p = subparsers.add_parser("build-index", help="Build inverted index over corpus")
    build_p.add_argument("--corpus", default="corpus", help="Path to corpus directory")
    build_p.add_argument("--output", default="outputs/index", help="Index output directory")
    build_p.add_argument("--method", choices=["spimi", "bsbi"], default="spimi", help="Indexing algorithm")
    build_p.add_argument("--memory-limit", type=int, default=50000, help="Memory limit in tokens")
    build_p.add_argument("--shards", type=int, default=1, help="Number of parallel shards (1 or 4)")
    build_p.add_argument("--stemming", action="store_true", default=True, help="Enable Porter stemming")
    build_p.add_argument("--no-stemming", dest="stemming", action="store_false", help="Disable Porter stemming")
    build_p.add_argument("--stopwords", action="store_true", default=True, help="Enable stopword removal")
    build_p.add_argument("--no-stopwords", dest="stopwords", action="store_false", help="Disable stopword removal")

    # query
    query_p = subparsers.add_parser("query", help="Execute search query")
    query_p.add_argument("query", nargs="+", help="Query string")
    query_p.add_argument("--index", default="outputs/index", help="Path to index directory")
    query_p.add_argument("--type", choices=["ranked", "boolean"], default="ranked", help="Query type")
    query_p.add_argument("--method", choices=["exhaustive", "champion_lists", "impact_ordering"], default="exhaustive")
    query_p.add_argument("--top-k", type=int, default=10, help="Number of results to return")

    # boolean-tests
    bool_p = subparsers.add_parser("boolean-tests", help="Validate Boolean tests against expected results")
    bool_p.add_argument("--index", default="outputs/index", help="Path to index directory")
    bool_p.add_argument("--tests", default="boolean_tests.tsv", help="Path to boolean_tests.tsv")
    bool_p.add_argument("--output", default="outputs/evaluation/boolean_results.csv", help="CSV report path")

    # misspelling-tests
    misp_p = subparsers.add_parser("misspelling-tests", help="Validate spelling correction")
    misp_p.add_argument("--index", default="outputs/index", help="Path to index directory")
    misp_p.add_argument("--tests", default="misspellings.tsv", help="Path to misspellings.tsv")
    misp_p.add_argument("--output", default="outputs/evaluation/misspellings_results.csv", help="CSV report path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Route subcommands
    if args.command == "build-index":
        from lexis.preprocessing.normalizer import Normalizer
        from lexis.preprocessing.stemmer import PorterStemmer
        from lexis.preprocessing.stopwords import StopwordFilter
        from lexis.preprocessing.tokenizer import Tokenizer
        from lexis.indexing.spimi import SPIMIIndexer
        from lexis.indexing.bsbi import BSBIIndexer
        from lexis.indexing.shard_builder import ShardBuilder

        tok = Tokenizer(
            normalizer=Normalizer(),
            stemmer=PorterStemmer(enabled=args.stemming),
            stopword_filter=StopwordFilter(enabled=args.stopwords),
        )

        if args.shards > 1:
            builder = ShardBuilder(tok, num_shards=args.shards)
            _, build_time = builder.build_sharded_index(args.corpus, args.output)
            print(f"Parallel index ({args.shards} shards) built in {build_time:.2f}s -> {args.output}")
        elif args.method == "spimi":
            indexer = SPIMIIndexer(tok, memory_limit_tokens=args.memory_limit)
            _, metrics = indexer.build_index(args.corpus, args.output)
            print(f"SPIMI index built: {metrics.total_docs} docs, {metrics.num_blocks} blocks, {metrics.build_time_seconds}s")
        else:
            indexer = BSBIIndexer(tok, memory_limit_tokens=args.memory_limit)
            _, metrics = indexer.build_index(args.corpus, args.output)
            print(f"BSBI index built: {metrics.total_docs} docs, {metrics.num_blocks} blocks, {metrics.build_time_seconds}s")

    elif args.command == "query":
        from lexis.index.positional_index import PositionalIndex
        from lexis.preprocessing.tokenizer import Tokenizer
        from lexis.query.query_engine import QueryEngine
        from lexis.ranking.scorer import RankedScorer

        idx = PositionalIndex(args.index)
        idx.load(args.index)
        tok = Tokenizer()

        if args.type == "boolean":
            qe = QueryEngine(idx, tok)
            docs = qe.search_exact(args.query)
            print(f"Matched {len(docs)} documents:")
            for d in docs[:args.top_k]:
                print(f"  {d}")
        else:
            scorer = RankedScorer(idx, tok)
            results, scored, lat = scorer.score_query(args.query, top_k=args.top_k, method=args.method)
            print(f"Scored {scored} candidates in {lat:.2f}ms. Top {len(results)}:")
            for r in results:
                print(f"  {r.doc_name} : {r.score:.4f}")

    elif args.command == "boolean-tests":
        from lexis.index.positional_index import PositionalIndex
        from lexis.preprocessing.tokenizer import Tokenizer
        from lexis.query.query_engine import QueryEngine
        from lexis.evaluation.boolean_eval import BooleanEvaluatorRunner
        from lexis.evaluation.reports import ReportGenerator

        idx = PositionalIndex(args.index)
        idx.load(args.index)
        tok = Tokenizer()
        qe = QueryEngine(idx, tok)
        runner = BooleanEvaluatorRunner(qe)
        results = runner.run_tests_from_file(args.tests)

        rows = []
        passed = 0
        for r in results:
            if r.exact_match:
                passed += 1
            rows.append({
                "query_id": r.query_id,
                "query": r.query,
                "type": r.test_type,
                "expected": r.expected_count,
                "actual": r.actual_count,
                "exact_match": r.exact_match,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
            })
        ReportGenerator.write_csv(args.output, list(rows[0].keys()) if rows else [], rows)
        print(f"Boolean Tests: {passed}/{len(results)} exact matches. Saved to {args.output}")

    elif args.command == "misspelling-tests":
        from lexis.index.positional_index import PositionalIndex
        from lexis.query.spelling import SpellingCorrector
        from lexis.evaluation.reports import ReportGenerator

        idx = PositionalIndex(args.index)
        idx.load(args.index)
        corrector = SpellingCorrector(idx)

        rows = []
        hits = 0
        with open(args.tests, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for l in lines[1:]:
            parts = l.split("\t")
            if len(parts) < 2:
                continue
            mis = parts[0]
            expected = parts[1]
            sug = corrector.suggest_correction(mis)
            correct = (sug == expected)
            if correct:
                hits += 1
            rows.append({"misspelled": mis, "expected": expected, "suggested": sug, "correct": correct})

        ReportGenerator.write_csv(args.output, ["misspelled", "expected", "suggested", "correct"], rows)
        print(f"Misspelling Tests: {hits}/{len(rows)} correct suggestions. Saved to {args.output}")


if __name__ == "__main__":
    main()
