"""
Retrieval evaluation metrics for TokenSmith.

Provides Recall@k and MRR computation, plus a RetrievalEvaluator that
runs a single query through a retrieval configuration and returns scores.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import RAGConfig
from src.ranking.ranker import EnsembleRanker
from src.retriever import filter_retrieved_chunks
from src.query_enhancement import generate_hypothetical_document


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """
    Fraction of relevant chunks that appear in the top-k retrieved results.

    recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k_set & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """
    Reciprocal rank of the first relevant chunk in the retrieved list.

    MRR = 1 / rank_of_first_hit  (0.0 if no hit found)
    """
    relevant_set = set(relevant_ids)
    for rank, idx in enumerate(retrieved_ids, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Per-query result container
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    benchmark_id: str
    question: str
    retrieved_ids: List[int]
    relevant_ids: List[int]
    recall_at_k: float
    mrr: float
    strategy: Optional[str] = None   # only set for adaptive system
    strategy_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """
    Runs retrieval for a list of benchmark queries and computes Recall@k / MRR.

    Parameters
    ----------
    artifacts : dict
        Must contain ``chunks``, ``retrievers``.
    cfg : RAGConfig
        Base configuration (top_k, num_candidates, ensemble settings).
    k : int
        Cut-off for Recall@k (defaults to cfg.top_k).
    use_adaptive : bool
        If True, wrap retrieval with RetrievalStrategySelector.
    """

    def __init__(
        self,
        artifacts: dict,
        cfg: RAGConfig,
        k: Optional[int] = None,
        use_adaptive: bool = False,
    ) -> None:
        self._artifacts = artifacts
        self._cfg = cfg
        self._k = k if k is not None else cfg.top_k
        self._use_adaptive = use_adaptive

        self._selector = None
        if use_adaptive:
            from src.retrieval.retrieval_strategy_selector import RetrievalStrategySelector
            self._selector = RetrievalStrategySelector(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_query(self, benchmark_id: str, question: str, relevant_ids: List[int]) -> QueryResult:
        """Retrieve chunks for *question* and compute Recall@k / MRR."""
        retrieved_ids, strategy, reason = self._retrieve(question)
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, self._k)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        return QueryResult(
            benchmark_id=benchmark_id,
            question=question,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            recall_at_k=r_at_k,
            mrr=mrr,
            strategy=strategy,
            strategy_reason=reason,
        )

    def evaluate_all(self, benchmarks: List[dict]) -> List[QueryResult]:
        """Run :meth:`evaluate_query` for every benchmark entry."""
        results = []
        for b in benchmarks:
            if not b.get("ideal_retrieved_chunks"):
                continue
            result = self.evaluate_query(
                benchmark_id=b["id"],
                question=b["question"],
                relevant_ids=b["ideal_retrieved_chunks"],
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, question: str):
        """
        Run retrieval and return (ordered_ids, strategy_name, reason).

        With adaptive=True, the strategy selector picks weights per query.
        With adaptive=False, the base config weights are used directly.
        """
        chunks = self._artifacts["chunks"]
        retrievers = self._artifacts["retrievers"]

        # Determine effective config + ranker
        effective_cfg = self._cfg
        strategy_name: Optional[str] = None
        reason: Optional[str] = None

        if self._selector is not None:
            decision = self._selector.select(question)
            effective_cfg = decision.modified_cfg
            strategy_name = decision.strategy.value
            reason = decision.reason

        ranker = EnsembleRanker(
            ensemble_method=effective_cfg.ensemble_method,
            weights=effective_cfg.ranker_weights,
            rrf_k=int(effective_cfg.rrf_k),
        )

        # Optionally apply HyDE
        retrieval_query = question
        if effective_cfg.use_hyde:
            try:
                retrieval_query = generate_hypothetical_document(
                    question,
                    effective_cfg.gen_model,
                    max_tokens=effective_cfg.hyde_max_tokens,
                )
            except Exception:
                retrieval_query = question  # fall back gracefully

        pool_n = max(effective_cfg.num_candidates, self._k + 10)
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)

        ordered_ids, _ = ranker.rank(raw_scores=raw_scores)
        return ordered_ids, strategy_name, reason


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def aggregate_results(results: List[QueryResult]) -> Dict[str, float]:
    """Return mean Recall@k and mean MRR across all query results."""
    if not results:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    return {
        "recall_at_k": sum(r.recall_at_k for r in results) / len(results),
        "mrr": sum(r.mrr for r in results) / len(results),
    }


def print_comparison_table(
    baseline_results: List[QueryResult],
    adaptive_results: List[QueryResult],
    k: int,
) -> None:
    """Print a side-by-side Recall@k / MRR comparison table."""
    base_map = {r.benchmark_id: r for r in baseline_results}
    adap_map = {r.benchmark_id: r for r in adaptive_results}
    all_ids = sorted(set(base_map) | set(adap_map))

    col_w = 24
    header = (
        f"{'Benchmark':<{col_w}} "
        f"{'Base R@'+str(k):>10} {'Adap R@'+str(k):>10} {'Δ R@'+str(k):>8}  "
        f"{'Base MRR':>10} {'Adap MRR':>10} {'Δ MRR':>8}  "
        f"{'Strategy':<18}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for bid in all_ids:
        b = base_map.get(bid)
        a = adap_map.get(bid)
        if b is None or a is None:
            continue
        delta_r = a.recall_at_k - b.recall_at_k
        delta_m = a.mrr - b.mrr
        sign_r = "+" if delta_r >= 0 else ""
        sign_m = "+" if delta_m >= 0 else ""
        strat = a.strategy or "—"
        print(
            f"{bid:<{col_w}} "
            f"{b.recall_at_k:>10.3f} {a.recall_at_k:>10.3f} {sign_r}{delta_r:>7.3f}  "
            f"{b.mrr:>10.3f} {a.mrr:>10.3f} {sign_m}{delta_m:>7.3f}  "
            f"{strat:<18}"
        )

    print("-" * len(header))
    base_agg = aggregate_results(baseline_results)
    adap_agg = aggregate_results(adaptive_results)
    dr = adap_agg["recall_at_k"] - base_agg["recall_at_k"]
    dm = adap_agg["mrr"] - base_agg["mrr"]
    sr = "+" if dr >= 0 else ""
    sm = "+" if dm >= 0 else ""
    print(
        f"{'MEAN':<{col_w}} "
        f"{base_agg['recall_at_k']:>10.3f} {adap_agg['recall_at_k']:>10.3f} {sr}{dr:>7.3f}  "
        f"{base_agg['mrr']:>10.3f} {adap_agg['mrr']:>10.3f} {sm}{dm:>7.3f}"
    )
    print("=" * len(header) + "\n")
