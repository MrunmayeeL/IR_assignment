# Lexis — Information Retrieval Search Engine (Winter 2026)

Lexis is a high-performance, modular information retrieval search engine built for the Reuters-21578 news archive (10,788 documents). It implements scalable index construction (SPIMI, BSBI, 4-shard parallelization), compact positional indexing with compression, exact retrieval (Boolean logic, quoted phrases, proximity windows, permuterm wildcards), spelling correction (Levenshtein distance with frequency ranking), and free-text ranked retrieval with Chapter 7 pruning (Champion lists, Impact ordering) and statistical ablation evaluation.

---

## Table of Contents
1. [Architecture & Directory Structure](#architecture--directory-structure)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Data Placement & Corpus Extraction](#data-placement--corpus-extraction)
4. [Step-by-Step Running Instructions](#step-by-step-running-instructions)
5. [Validation Commands & Expected Results](#validation-commands--expected-results)
   - [A. Exact Retrieval & Boolean Tests](#a-exact-retrieval--boolean-tests)
   - [B. Spelling Correction Tests](#b-spelling-correction-tests)
   - [C. Indexing Benchmarks (SPIMI vs BSBI vs Shards)](#c-indexing-benchmarks-spimi-vs-bsbi-vs-shards)
   - [D. Compression Experiments](#d-compression-experiments)
   - [E. Skip-Pointer Intersection Experiments](#e-skip-pointer-intersection-experiments)
   - [F. Ranked Retrieval Tuning & Evaluation (Train Queries)](#f-ranked-retrieval-tuning--evaluation-train-queries)
   - [G. Final Test Evaluation (Test Queries)](#g-final-test-evaluation-test-queries)
   - [H. Pruning & Efficiency Experiments (Chapter 7)](#h-pruning--efficiency-experiments-chapter-7)
   - [I. Ablation Study & Paired Significance Tests](#i-ablation-study--paired-significance-tests)
6. [Design Choices & Mathematical Formulations](#design-choices--mathematical-formulations)
7. [Unit Testing Suite](#unit-testing-suite)

---

## Architecture & Directory Structure

```text
lexis/
├── README.md                           # Master instructions and system documentation
├── requirements.txt                    # Python library dependencies
├── setup.py                            # Package installation config
├── .gitignore                          # Git ignore definitions
├── config/
│   └── config.yaml                     # Configurable thresholds and engine hyperparameters
├── src/
│   └── lexis/
│       ├── __init__.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── normalizer.py           # Case folding and character normalization
│       │   ├── stemmer.py              # Pure-Python deterministic Porter stemmer (1980)
│       │   ├── stopwords.py            # Standard IR stopwords list with toggle
│       │   └── tokenizer.py            # Unified alphanumeric tokenizer with zone & position tracking
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── document_reader.py      # Streaming document parser (headline, blank line, body)
│       │   ├── block_writer.py         # Intermediate sorted block serialization
│       │   ├── index_merger.py         # Multi-way partial block merge
│       │   ├── spimi.py                # Single-Pass In-Memory Indexing with memory thresholds
│       │   ├── bsbi.py                 # Block Sort-Based Indexing with memory thresholds
│       │   ├── shard_builder.py        # 4-shard parallel indexing via ProcessPoolExecutor
│       │   ├── incremental.py          # Dynamic document addition without rebuild
│       │   └── deletion.py             # O(1) tombstone deletion & physical vacuum compaction
│       ├── index/
│       │   ├── __init__.py
│       │   ├── dictionary.py           # Term dictionary and DocumentRegistry mapping
│       │   ├── postings.py             # Positional postings lists with zone frequencies
│       │   ├── positional_index.py     # On-disk binary index with on-demand term retrieval
│       │   ├── skip_pointers.py        # Skip pointers and instrumented comparison counters
│       │   └── compression.py          # Gap, Variable Byte, Elias Gamma, and Dictionary compression
│       ├── query/
│       │   ├── __init__.py
│       │   ├── lexer.py                # Tokenizer for Boolean, Phrase, Proximity, and Wildcards
│       │   ├── parser.py               # Precedence-aware recursive descent AST parser
│       │   ├── boolean.py              # AST tree evaluator (AND, OR, NOT, parentheses)
│       │   ├── phrase.py               # Consecutive positional phrase evaluator
│       │   ├── proximity.py            # Bidirectional positional window evaluator (/N)
│       │   ├── wildcard.py             # Permuterm index vocabulary expansion for '*'
│       │   ├── spelling.py             # Levenshtein distance OOV correction with frequency ranking
│       │   └── query_engine.py         # Unified search engine interface
│       ├── ranking/
│       │   ├── __init__.py
│       │   ├── tfidf.py                # Log-TF, Raw-TF, and IDF vector computations
│       │   ├── zones.py                # Separate title/body zone weighting
│       │   ├── normalization.py        # Cosine vector and document length normalizations
│       │   ├── champion_lists.py       # Chapter 7 precomputed top-r champion lists
│       │   ├── impact_ordering.py      # Chapter 7 impact-ordered postings with early termination
│       │   └── scorer.py               # Decoupled candidate generator and ranked cosine scorer
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ranking_metrics.py      # P@10, MAP, and NDCG@10 implementations
│       │   ├── precision_recall.py     # 11-point interpolated precision-recall curve
│       │   ├── significance.py         # Paired Student's t-test for ablation p-values
│       │   ├── boolean_eval.py         # Automated validator against boolean_tests.tsv
│       │   └── reports.py              # CSV/JSON exporters and Zipf/Heaps fitting
│       └── cli.py                      # Unified CLI entry point
├── scripts/
│   ├── build_index.py                  # Primary index construction script
│   ├── run_query.py                    # Interactive/single query runner
│   ├── run_boolean_tests.py            # Validates 25 queries in boolean_tests.tsv
│   ├── run_misspelling_tests.py        # Validates 60 queries in misspellings.tsv
│   ├── benchmark_indexing.py           # Evaluates 4 memory limits, SPIMI vs BSBI, 4 shards
│   ├── benchmark_compression.py        # Compares uncompressed, gaps, VB, gamma on top-100 & rare
│   ├── benchmark_skips.py              # Measures comparison savings across 8 pairs & 3 spacings
│   ├── evaluate_train.py               # Tunes baselines & full system on queries_train.tsv
│   ├── evaluate_test.py                # Runs final evaluation once on queries_test.tsv
│   ├── evaluate_pruning.py             # Compares exhaustive vs champion lists vs impact ordering
│   └── run_ablations.py                # Runs one-at-a-time ablations with paired t-test p-values
└── tests/
    ├── test_tokenizer.py               # Tests tokens, zones, positions, stemmer, stopwords
    ├── test_index.py                   # Tests index persistence, skips, incremental, deletion
    ├── test_boolean.py                 # Tests AND, OR, NOT, parentheses, operator precedence
    ├── test_phrase.py                  # Tests consecutive positional phrase matching
    ├── test_proximity.py               # Tests bidirectional proximity window queries
    ├── test_wildcard.py                # Tests prefix, suffix, infix permuterm expansions
    ├── test_spelling.py                # Tests Levenshtein distance and frequency ranking
    ├── test_ranking.py                 # Tests TF-IDF scoring, zones, and ranking metrics
    └── test_compression.py             # Tests Gap, Variable Byte, Gamma, and Dictionary codecs
```

---

## Prerequisites & Installation

### Requirements
- Python 3.8+ (Python 3.12 verified)
- Standard scientific Python libraries: `pyyaml`, `numpy`, `scipy`, `pandas`, `psutil`, `matplotlib`

### Installation Command
```powershell
pip install -r requirements.txt
```
*(Optional editable install)*:
```powershell
pip install -e .
```

*Note*: All standalone scripts in `scripts/` configure `sys.path` dynamically, so they can be run immediately without installing the package.

---

## Data Placement & Corpus Extraction

The workspace includes `ProgrammingAssignmentIR_Winter2026.rar`, containing the 10,788 Reuters articles.
To extract the corpus into the expected `./corpus` directory, run:

```powershell
# Using tar (built into Windows 10/11)
tar -xf ProgrammingAssignmentIR_Winter2026.rar ProgrammingAssignmentIR_Winter2026/lexis_dataset/corpus
# Move corpus folder to root
Move-Item -Path "ProgrammingAssignmentIR_Winter2026/lexis_dataset/corpus" -Destination "corpus"
# Clean up temporary extracted folder
Remove-Item -Recurse -Force "ProgrammingAssignmentIR_Winter2026"
```
Ensure the following files are in the root directory:
- `corpus/` (containing `doc_00001.txt` ... `doc_10788.txt`)
- `boolean_tests.tsv`
- `misspellings.tsv`
- `queries_train.tsv`
- `queries_test.tsv`
- `qrels.tsv`

---

## Step-by-Step Running Instructions

Follow this recommended sequence to execute the full experimental pipeline:

### 1. Build the Main Index
Construct the primary on-disk index using SPIMI with stemming and stopword removal enabled:
```powershell
python scripts/build_index.py --corpus corpus --output outputs/index --method spimi --memory-limit 50000 --shards 1
```
*Expected Output*:
- Parses all 10,788 documents.
- Flushes sorted intermediate blocks when memory reaches 50,000 tokens.
- Merges blocks into `outputs/index/metadata.json` and `outputs/index/postings.dat`.
- Saves build metrics to `outputs/benchmarks/build_metrics.json`.

---

### 2. Validate Exact Retrieval (Boolean Tests)
Validate all 25 exact queries (Boolean, Phrase, Proximity, Wildcard) against the expected sets in `boolean_tests.tsv`:
```powershell
# First build an exact index without stemming to match the brute-force baseline:
python scripts/build_index.py --corpus corpus --output outputs/index_exact --no-stemming --no-stopwords

# Run the validation test suite:
python scripts/run_boolean_tests.py --index outputs/index_exact --tests boolean_tests.tsv --output outputs/evaluation/boolean_results.csv
```
*Expected Output*:
- **25/25 queries match exactly** (Precision = 1.0, Recall = 1.0, F1 = 1.0).
- Detailed CSV saved to `outputs/evaluation/boolean_results.csv`.

---

### 3. Validate Spelling Correction
Validate the 60 misspelling test cases from `misspellings.tsv`:
```powershell
python scripts/run_misspelling_tests.py --index outputs/index --tests misspellings.tsv --output outputs/evaluation/misspellings_results.csv
```
*Expected Output*:
- Evaluates Levenshtein distance $\le 2$ against the vocabulary.
- Tie-breaks multiple candidates using corpus term frequency.
- Output CSV saved to `outputs/evaluation/misspellings_results.csv`.

---

### 4. Run Index Construction Benchmarks
Evaluate 4 memory limits (10k, 25k, 50k, 100k), SPIMI vs BSBI, and parallel shard speedups (1, 2, 4 shards):
```powershell
python scripts/benchmark_indexing.py --corpus corpus --output outputs/benchmarks/indexing.csv
```
*Expected Output*:
- Table comparing:
  - SPIMI across 4 memory thresholds (recording blocks, bytes written, peak RSS memory, build time).
  - BSBI vs SPIMI.
  - 1-shard, 2-shard, and 4-shard parallel construction with speedup factor.
- Saved to `outputs/benchmarks/indexing.csv`.

---

### 5. Run Compression Experiments
Compare Uncompressed, Gap encoding, Variable-Byte, and Elias Gamma on the 100 most frequent terms and rarest 10,000 terms:
```powershell
python scripts/benchmark_compression.py --index outputs/index --output outputs/benchmarks/compression.csv
```
*Expected Output*:
- Reports bytes, compression ratio, and percentage space saved for each scheme.
- Saved to `outputs/benchmarks/compression.csv`.

---

### 6. Run Skip-Pointer Experiments
Evaluate comparison reductions on 8 term pairs across 3 skip spacings (4, 8, 16):
```powershell
python scripts/benchmark_skips.py --index outputs/index --output outputs/benchmarks/skips.csv
```
*Expected Output*:
- Reports comparisons without skips vs with skips, skips taken, and comparisons saved percentage.
- Saved to `outputs/benchmarks/skips.csv`.

---

### 7. Evaluate Training Queries & Baselines
Run tuning and baseline comparisons across the 40 training queries:
```powershell
python scripts/evaluate_train.py --index outputs/index --queries queries_train.tsv --qrels qrels.tsv --output outputs/evaluation/train_metrics.csv --pr-output outputs/evaluation/train_pr_curve.csv
```
*Expected Output*:
- Compares:
  1. `raw_tf_baseline`
  2. `tfidf_baseline`
  3. `full_system` (TF-IDF + Title/Body Zones + Length Normalization)
- Generates Precision@10, MAP, NDCG@10, and standard 11-point interpolated PR curve.
- Outputs saved to `outputs/evaluation/train_metrics.csv` and `outputs/evaluation/train_pr_curve.csv`.

---

### 8. Run Final Test Evaluation
Run the final evaluation once on the 40 test queries:
```powershell
python scripts/evaluate_test.py --index outputs/index --queries queries_test.tsv --qrels qrels.tsv --output outputs/evaluation/test_metrics.csv --pr-output outputs/evaluation/test_pr_curve.csv
```
*Expected Output*:
- Reports final Precision@10, MAP, NDCG@10, and 11-point PR curve on unseen test queries.
- Outputs saved to `outputs/evaluation/test_metrics.csv` and `outputs/evaluation/test_pr_curve.csv`.

---

### 9. Evaluate Pruning Strategies (Chapter 7)
Compare Exhaustive scoring vs Champion Lists ($r=50$) vs Impact Ordering:
```powershell
python scripts/evaluate_pruning.py --index outputs/index --queries queries_train.tsv --top-k 10 --output outputs/benchmarks/pruning.csv
```
*Expected Output*:
- Per-query and aggregate metrics:
  - Candidates scored (e.g. ~70-80% reduction for champion lists/impact ordering)
  - Query latency in milliseconds
  - Overlap@10 relative to exhaustive ranking
- Saved to `outputs/benchmarks/pruning.csv`.

---

### 10. Run Ablation Experiments & Statistical Significance
Disable individual pipeline components one-at-a-time to measure MAP drop and paired Student's t-test p-value:
```powershell
python scripts/run_ablations.py --index outputs/index --queries queries_train.tsv --qrels qrels.tsv --output outputs/evaluation/ablations.csv
```
*Expected Output*:
- Evaluates:
  - `full_system`
  - `no_stemming`
  - `no_stopwords`
  - `no_idf`
  - `no_length_norm`
  - `no_zone_weighting`
- For each, reports MAP, $\Delta\text{MAP}$, and two-tailed p-value with $p < 0.05$ significance flags.
- Saved to `outputs/evaluation/ablations.csv`.

---

### 11. Run Free-Text & Interactive Queries
Execute arbitrary queries via CLI:
```powershell
# Ranked retrieval with champion lists:
python scripts/run_query.py "federal reserve interest rates" --index outputs/index --type ranked --method champion_lists --top-k 10

# Exact boolean retrieval:
python scripts/run_query.py '"trade deficit" AND NOT japan' --index outputs/index_exact --type boolean
```

---

## Design Choices & Mathematical Formulations

### 1. Document Reading & Unified Positions
- **Title vs Body**: The first line of each Reuters text file is parsed as `title`, followed by a blank separator line and the `body`.
- **Continuous Positional Stream**: Positions $0, 1, 2, \dots$ increment continuously across the full document text. Each token is tagged with its zone (`title` or `body`). This ensures phrase queries like `"interest rates"` and proximity queries `economy /5 growth` resolve correctly according to original text offsets.

### 2. Compression Algorithms
- **Gap Encoding**: Postings $[d_1, d_2, d_3, \dots]$ are stored as positive differences: $[d_1, d_2 - d_1, d_3 - d_2, \dots]$.
- **Variable-Byte (VB)**: Numbers are split into 7-bit chunks. The highest bit (bit 7) is 0 for intermediate bytes and 1 for the terminal byte.
- **Elias Gamma**: An integer $n \ge 1$ is encoded with unary length followed by binary offset:
  $\text{unary}(1 + \lfloor\log_2 n\rfloor) \mathbin{\Vert} \text{low-order bits}$.
- **Dictionary Compression**: Terms are concatenated into a single UTF-8 string buffer, indexed by term pointer offsets.

### 3. Exact Retrieval & Permuterm Wildcards
- **AST Parser**: Precedence is strictly enforced: $\text{NOT} > \text{AND} > \text{OR}$. Parentheses override precedence.
- **Permuterm Index**: For each term $w$, we index all cyclic rotations of $w\$$.
  - $X* \to \$X*$
  - $*X \to X\$$
  - $X*Y \to Y\$X*$
  - Multi-wildcard queries ($pet*chem*$) anchor prefix and suffix in the permuterm index, then apply regex filtering over the candidate subset without scanning the document collection.

### 4. Ranking & Zone Weighting
- **Log Term Frequency**: $\text{TF}(t, d) = 1 + \log_{10}(\text{count}(t, d))$ if $\text{count} > 0$, else $0$.
- **Inverse Document Frequency**: $\text{IDF}(t) = \log_{10}(N / \text{DF}_t)$.
- **Zone Combination**: $\text{Weight}(t, d) = w_{\text{title}} \cdot \text{TF}_{\text{title}}(t, d) + w_{\text{body}} \cdot \text{TF}_{\text{body}}(t, d)$, with default $w_{\text{title}} = 0.3, w_{\text{body}} = 0.7$.
- **Length Normalization**: Document vectors are normalized by $\sqrt{\text{length}(d)}$ to prevent long documents from dominating results.

### 5. Candidate Pruning (Chapter 7)
- **Champion Lists**: Precomputes top-$r$ ($r=50$) documents for each term based on term frequency. At query time, only the union of champion lists is scored.
- **Impact Ordering**: Terms in the query are ordered by IDF descending (most discriminative terms first). Postings for each term are sorted by impact (TF descending), allowing early exit after evaluating top candidates.

### 6. Incremental Updates & Deletions
- **Incremental Additions**: New documents can be appended to the index on-the-fly via `IncrementalIndexer.add_document()` without requiring a complete index rebuild.
- **Tombstone Deletion**: `DeletionManager.delete_document()` flags documents as deleted in $O(1)$ time, dynamically filtering them during retrieval. `DeletionManager.vacuum()` reclaims disk space by compacting postings.

---

## Unit Testing Suite

The project includes 9 self-contained unit tests using synthetic data (no corpus required):

Run with Python's built-in `unittest` runner:
```powershell
python -m unittest discover -s tests
```
Or with pytest if installed:
```powershell
pytest tests/ -v
```
All 9 unit test suites verify:
1. `test_tokenizer.py`: Case folding, Porter stemming toggle, stopword filtering, continuous positions, and zone tagging.
2. `test_index.py`: Binary serialization/deserialization, skip-pointer intersection, incremental addition, and tombstone deletion.
3. `test_boolean.py`: AND, OR, NOT logic, nested parentheses, and operator precedence.
4. `test_phrase.py`: Positional consecutive matching for quoted phrase queries.
5. `test_proximity.py`: Bidirectional proximity window enforcement ($1 \le |p_1 - p_2| \le N$).
6. `test_wildcard.py`: Prefix, suffix, and infix permuterm vocabulary expansions.
7. `test_spelling.py`: Levenshtein edit distance and frequency-based candidate tie-breaking.
8. `test_ranking.py`: Vector cosine scoring, title/body zone weighting, P@10, AP, and NDCG@10.
9. `test_compression.py`: Correctness and reversibility of Gap, Variable-Byte, Elias Gamma, and Dictionary codecs.
