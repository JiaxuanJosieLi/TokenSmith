"""
Adaptive vs Baseline Retrieval Evaluation
==========================================

Compares retrieval quality (Recall@k, MRR) between:
  - Baseline system: fixed hybrid retrieval (FAISS 0.5 / BM25 0.5, no HyDE)
  - Adaptive system: query-aware strategy selection via RetrievalStrategySelector

Run with:
    pytest tests/test_adaptive_evaluation.py -s -v

Or to run only the retrieval evaluation (no LLM generation needed):
    pytest tests/test_adaptive_evaluation.py::test_retrieval_quality -s -v
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAGConfig
from src.evaluation.evaluator import (
    RetrievalEvaluator,
    aggregate_results,
    print_comparison_table,
)
from src.retriever import BM25Retriever, FAISSRetriever, load_artifacts
from src.ranking.ranker import EnsembleRanker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS_FILE = Path(__file__).parent / "benchmarks.yaml"
RESULTS_DIR = Path(__file__).parent / "results"
K = 10  # Recall@K cutoff — must match config top_k = 10

# Baseline uses a fixed hybrid strategy, no adaptive selector, no HyDE
BASELINE_WEIGHTS = {"faiss": 0.5, "bm25": 0.5}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eval_cfg(pytestconfig) -> RAGConfig:
    """Load RAGConfig from config/config.yaml."""
    config_path = Path(pytestconfig.getoption("--config", default="config/config.yaml"))
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    cfg = RAGConfig.from_yaml(config_path)
    return cfg


@pytest.fixture(scope="module")
def shared_artifacts(eval_cfg: RAGConfig):
    """
    Load FAISS + BM25 artifacts once and share across both evaluators.
    Building retrievers is expensive; we reuse them for both baseline and adaptive.
    """
    artifacts_dir = eval_cfg.get_artifacts_directory()
    index_prefix = "textbook_index"

    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
    )
    retrievers = [
        FAISSRetriever(faiss_index, eval_cfg.embed_model),
        BM25Retriever(bm25_index),
    ]
    return {
        "chunks": chunks,
        "sources": sources,
        "meta": metadata,
        "retrievers": retrievers,
    }


@pytest.fixture(scope="module")
def all_benchmarks() -> List[dict]:
    """Load all benchmark entries that have ideal_retrieved_chunks."""
    with open(BENCHMARKS_FILE) as f:
        data = yaml.safe_load(f)
    return [b for b in data["benchmarks"] if b.get("ideal_retrieved_chunks")]


# ---------------------------------------------------------------------------
# Main evaluation test
# ---------------------------------------------------------------------------

def test_retrieval_quality(shared_artifacts, eval_cfg, all_benchmarks):
    """
    Compare Recall@k and MRR between baseline and adaptive retrieval.

    Baseline : fixed HYBRID weights (faiss=0.5, bm25=0.5), no HyDE.
    Adaptive : RetrievalStrategySelector picks strategy per query.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Baseline evaluator -------------------------------------------
    baseline_cfg = deepcopy(eval_cfg)
    baseline_cfg.ranker_weights = BASELINE_WEIGHTS
    baseline_cfg.use_hyde = False

    baseline_evaluator = RetrievalEvaluator(
        artifacts=shared_artifacts,
        cfg=baseline_cfg,
        k=K,
        use_adaptive=False,
    )

    # --- Adaptive evaluator -------------------------------------------
    adaptive_evaluator = RetrievalEvaluator(
        artifacts=shared_artifacts,
        cfg=eval_cfg,
        k=K,
        use_adaptive=True,
    )

    # --- Run evaluation -----------------------------------------------
    print(f"\n[Evaluation] Running {len(all_benchmarks)} benchmark queries (k={K})...")

    baseline_results = baseline_evaluator.evaluate_all(all_benchmarks)
    adaptive_results = adaptive_evaluator.evaluate_all(all_benchmarks)

    # --- Print comparison table ---------------------------------------
    print_comparison_table(baseline_results, adaptive_results, k=K)

    # --- Aggregate metrics --------------------------------------------
    base_agg = aggregate_results(baseline_results)
    adap_agg = aggregate_results(adaptive_results)

    print(f"Baseline  — Mean Recall@{K}: {base_agg['recall_at_k']:.3f}  |  Mean MRR: {base_agg['mrr']:.3f}")
    print(f"Adaptive  — Mean Recall@{K}: {adap_agg['recall_at_k']:.3f}  |  Mean MRR: {adap_agg['mrr']:.3f}")
    delta_r = adap_agg["recall_at_k"] - base_agg["recall_at_k"]
    delta_m = adap_agg["mrr"] - base_agg["mrr"]
    print(f"Delta     — ΔRecall@{K}: {delta_r:+.3f}  |  ΔMRR: {delta_m:+.3f}\n")

    # --- Save results to JSON -----------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "k": K,
        "baseline_weights": BASELINE_WEIGHTS,
        "aggregate": {
            "baseline": base_agg,
            "adaptive": adap_agg,
            "delta": {"recall_at_k": delta_r, "mrr": delta_m},
        },
        "per_query": [
            {
                "benchmark_id": b.benchmark_id,
                "question": b.question,
                "baseline": {
                    "recall_at_k": b.recall_at_k,
                    "mrr": b.mrr,
                },
                "adaptive": {
                    "recall_at_k": a.recall_at_k,
                    "mrr": a.mrr,
                    "strategy": a.strategy,
                    "strategy_reason": a.strategy_reason,
                },
            }
            for b, a in zip(baseline_results, adaptive_results)
        ],
    }
    results_file = RESULTS_DIR / "adaptive_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Evaluation] Results saved to {results_file}")

    # --- Strategy distribution ----------------------------------------
    strategy_counts: Dict[str, int] = {}
    for r in adaptive_results:
        s = r.strategy or "unknown"
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    print("[Evaluation] Strategy distribution:")
    for strat, count in sorted(strategy_counts.items()):
        print(f"  {strat:<20} : {count} queries")

    # --- Soft assertion: adaptive should not be significantly worse ----
    # We allow a tolerance of 0.05 — adaptive should be roughly on par or better
    TOLERANCE = 0.05
    assert adap_agg["recall_at_k"] >= base_agg["recall_at_k"] - TOLERANCE, (
        f"Adaptive Recall@{K} ({adap_agg['recall_at_k']:.3f}) is more than "
        f"{TOLERANCE} below baseline ({base_agg['recall_at_k']:.3f})"
    )
    assert adap_agg["mrr"] >= base_agg["mrr"] - TOLERANCE, (
        f"Adaptive MRR ({adap_agg['mrr']:.3f}) is more than "
        f"{TOLERANCE} below baseline ({base_agg['mrr']:.3f})"
    )


# ---------------------------------------------------------------------------
# Per-strategy ablation test
# ---------------------------------------------------------------------------

def test_strategy_ablation(shared_artifacts, eval_cfg, all_benchmarks):
    """
    Ablation: force each strategy individually and compare Recall@k / MRR.
    This shows which strategy works best on average across all query types.
    """
    from src.retrieval.retrieval_strategy_selector import RetrievalStrategy

    RESULTS_DIR.mkdir(exist_ok=True)

    strategy_presets = {
        "bm25_only":  {"ranker_weights": {"faiss": 0.2, "bm25": 0.8}, "use_hyde": False},
        "dense_only": {"ranker_weights": {"faiss": 0.8, "bm25": 0.2}, "use_hyde": False},
        "hybrid":     {"ranker_weights": {"faiss": 0.5, "bm25": 0.5}, "use_hyde": False},
    }

    print(f"\n[Ablation] Forcing each strategy over {len(all_benchmarks)} queries (k={K})...\n")

    ablation_results = {}
    for strategy_name, preset in strategy_presets.items():
        forced_cfg = deepcopy(eval_cfg)
        forced_cfg.ranker_weights = preset["ranker_weights"]
        forced_cfg.use_hyde = preset["use_hyde"]

        evaluator = RetrievalEvaluator(
            artifacts=shared_artifacts,
            cfg=forced_cfg,
            k=K,
            use_adaptive=False,
        )
        results = evaluator.evaluate_all(all_benchmarks)
        agg = aggregate_results(results)
        ablation_results[strategy_name] = agg
        print(f"  {strategy_name:<20}  Recall@{K}: {agg['recall_at_k']:.3f}  MRR: {agg['mrr']:.3f}")

    # Also run adaptive for comparison
    adaptive_evaluator = RetrievalEvaluator(
        artifacts=shared_artifacts,
        cfg=eval_cfg,
        k=K,
        use_adaptive=True,
    )
    adaptive_results = adaptive_evaluator.evaluate_all(all_benchmarks)
    adap_agg = aggregate_results(adaptive_results)
    ablation_results["adaptive"] = adap_agg
    print(f"  {'adaptive (selector)':<20}  Recall@{K}: {adap_agg['recall_at_k']:.3f}  MRR: {adap_agg['mrr']:.3f}")

    # Save ablation results
    ablation_file = RESULTS_DIR / "ablation_results.json"
    with open(ablation_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "k": K,
            "results": ablation_results,
        }, f, indent=2)
    print(f"\n[Ablation] Results saved to {ablation_file}")
