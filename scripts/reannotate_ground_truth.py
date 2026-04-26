"""
Re-annotate benchmarks.yaml with ground truth chunk IDs from the current index.

For each benchmark, scores all chunks against the expected_answer using BM25,
then shows the top candidates for manual confirmation before updating the file.

Usage:
    conda activate tokensmith
    python scripts/reannotate_ground_truth.py

Optional flags:
    --top_k 5          Number of ground truth chunks to keep per benchmark (default: 5)
    --auto             Skip confirmation prompts and write automatically
    --dry_run          Print proposed changes without writing
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCHMARKS_FILE = ROOT / "tests" / "benchmarks.yaml"
ARTIFACTS_DIR   = ROOT / "index" / "sections"
INDEX_PREFIX    = "textbook_index"
CHUNKS_FILE     = ARTIFACTS_DIR / f"{INDEX_PREFIX}_chunks.pkl"


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lower-case, split on whitespace and punctuation."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25(chunks: list[str]) -> BM25Okapi:
    tokenized = [tokenize(c) for c in chunks]
    return BM25Okapi(tokenized)


def top_k_chunks(bm25: BM25Okapi, query: str, k: int) -> list[tuple[int, float]]:
    """Return (chunk_id, score) for top-k chunks matching *query*."""
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in ranked[:k] if score > 0]


def preview(chunk_id: int, chunks: list[str], width: int = 120) -> str:
    text = chunks[chunk_id].replace("\n", " ").strip()
    return text[:width] + ("…" if len(text) > width else "")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",   type=int, default=5)
    parser.add_argument("--auto",    action="store_true",
                        help="Write without asking for confirmation")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print proposed IDs but do not write")
    args = parser.parse_args()

    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE} …")
    with open(CHUNKS_FILE, "rb") as f:
        chunks: list[str] = pickle.load(f)
    print(f"  {len(chunks):,} chunks loaded.\n")

    # Build BM25 over all chunks
    print("Building BM25 index …")
    bm25 = build_bm25(chunks)
    print("  Done.\n")

    # Load benchmarks
    with open(BENCHMARKS_FILE) as f:
        data = yaml.safe_load(f)
    benchmarks: list[dict] = data["benchmarks"]

    updated = []
    for bm in benchmarks:
        bid      = bm["id"]
        question = bm["question"]
        expected = bm["expected_answer"]
        keywords = " ".join(bm.get("keywords", []))

        # Search using expected_answer + keywords for broader coverage
        search_query = expected + " " + keywords
        hits = top_k_chunks(bm25, search_query, k=args.top_k * 3)

        if not hits:
            print(f"[{bid}] ⚠️  No BM25 hits — keeping old IDs.\n")
            updated.append(bm)
            continue

        # Pick top-k
        proposed_ids = [idx for idx, _ in hits[:args.top_k]]

        print(f"{'─'*70}")
        print(f"[{bid}]")
        print(f"  Q : {question}")
        print(f"  A : {expected[:100]}…")
        print(f"  Proposed ideal_retrieved_chunks: {proposed_ids}")
        print(f"  Old                            : {bm.get('ideal_retrieved_chunks', [])}")
        print()
        for rank, (idx, score) in enumerate(hits[:args.top_k], 1):
            print(f"  [{rank}] chunk {idx:>5}  score={score:.2f}  │  {preview(idx, chunks)}")
        print()

        if args.dry_run:
            updated.append(bm)
            continue

        if args.auto:
            accept = True
        else:
            ans = input("  Accept proposed IDs? [Y/n/custom]: ").strip().lower()
            if ans in ("", "y", "yes"):
                accept = True
            elif ans in ("n", "no"):
                accept = False
                print("  Keeping old IDs.\n")
            else:
                # User typed custom IDs like "12 34 56"
                try:
                    proposed_ids = [int(x) for x in ans.split()]
                    accept = True
                    print(f"  Using custom IDs: {proposed_ids}")
                except ValueError:
                    accept = False
                    print("  Could not parse — keeping old IDs.\n")

        if accept:
            bm = dict(bm)
            bm["ideal_retrieved_chunks"] = proposed_ids
        updated.append(bm)

    if not args.dry_run:
        data["benchmarks"] = updated
        with open(BENCHMARKS_FILE, "w") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        print(f"\n✅  benchmarks.yaml updated ({len(updated)} entries).")
    else:
        print("\n[dry-run] No file written.")


if __name__ == "__main__":
    main()
