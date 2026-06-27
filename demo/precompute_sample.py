#!/usr/bin/env python3
"""
precompute.py  —  Redrob AI Candidate Ranker  (v6)
Builds all artifacts rank.py depends on. Run ONCE before rank.py.
Usage:
    python precompute.py --candidates candidates.jsonl
"""
import argparse, json, pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
ARTIFACTS  = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)

# ── JD Text ───────────────────────────────────────────────────────────────────
# v6: Expanded with BM25-migration signals, evaluation emphasis, temporal language
JD_EXPANDED = """
Senior AI Engineer building ranking retrieval embedding systems.
Production deployment semantic search vector databases hybrid retrieval.
Dense retrieval ANN search FAISS Milvus Pinecone Weaviate Qdrant Elasticsearch.
Sentence-transformers SBERT text embeddings dense vectors semantic similarity.
NLP information retrieval NDCG MRR MAP A/B testing offline benchmarks.
Python production code software engineering quality.
LLM integration RAG retrieval-augmented generation fine-tuning LoRA QLoRA PEFT.
Recommendation systems search ranking candidate matching scale real users.
Product company experience. Not pure research. Not pure services consulting.
Scrappy engineering mindset ships working systems fast deployed production.
Migrated upgraded legacy BM25 keyword search to hybrid semantic retrieval embeddings.
Improved recruiter engagement metrics click-through conversion ranking quality.
Offline benchmark online A/B experiment evaluation NDCG MRR recruiter feedback loop.
Pre-LLM retrieval ranking information retrieval experience before 2022.
Embedding drift index refresh retrieval quality regression monitoring.
Not framework tutorial demo LangChain only — systems thinker production engineer.
"""

# v6: Second anchor now includes offline/online benchmark and BM25/hybrid vocab
JD_SKILL_ANCHORS = [
    "production embeddings retrieval semantic search vector database",
    "NDCG MRR MAP ranking evaluation offline benchmark online A/B experiment recruiter feedback",
    "Python production code software engineering quality",
    "LLM fine-tuning LoRA QLoRA PEFT language model",
    "NLP natural language processing information retrieval dense retrieval BM25 hybrid search",
    "recommendation systems candidate matching personalization ranking search improvement",
    "distributed systems inference optimization scalability latency",
]

# ── Keyword Lists ─────────────────────────────────────────────────────────────
# v6: PROD_T1 includes evaluation vocabulary (JD: "evaluate rigorously")
PROD_T1 = [
    # Core production evidence
    "production", "deployed", "real users", "at scale", "serving",
    "latency", "a/b test", "throughput", "millions of", "live system",
    "queries per second", "qps", "p99", "p95", "sla", "uptime",
    "index refresh", "embedding drift", "retrieval quality regression",
    "model monitoring", "inference pipeline", "feature store",
    # v6: evaluation vocabulary
    "offline benchmark", "online experiment", "ndcg", "mrr",
    "precision at k", "recall at k", "click-through", "recruiter feedback",
    "a/b experiment", "evaluation framework", "ranking quality",
]

# v6: PROD_T2 includes BM25-to-hybrid migration signals (JD 90-day mandate)
PROD_T2 = [
    # Core ownership signals
    "shipped", "launched", "built end-to-end", "owned", "led design",
    "search", "ranking", "retrieval", "recommendation", "matching",
    "pipeline", "infrastructure", "architecture",
    # v6: BM25-migration signals
    "bm25", "tf-idf", "keyword search", "migrated", "migrated from",
    "replaced", "improved precision", "improved recall", "reduced latency",
    "recruiter engagement", "click-through rate", "conversion rate",
    "hybrid retrieval", "hybrid search", "re-ranking", "reranking",
]

# v6: RESEARCH_NEGS includes framework-enthusiast signals (JD: "LangChain tutorials")
RESEARCH_NEGS = [
    # Pure research anti-patterns
    "arxiv", "research lab", "academic", "thesis",
    "proof of concept", "prototype only", "exploration paper",
    # v6: framework-enthusiast anti-patterns — mild penalty, not a gate
    "langchain tutorial", "how i built", "getting started with",
    "demo project", "side project only",
]

PROFICIENCY_WEIGHT = {
    "beginner": 0.50, "intermediate": 0.75, "advanced": 1.00, "expert": 1.10,
}


# ── v6: Temporal Signal Helper ─────────────────────────────────────────────────
# JD: "people who understood retrieval and ranking before it became fashionable"
def _has_pre_llm_ai_experience(candidate: dict) -> bool:
    """Returns True if candidate has AI/ML/search role starting before 2022."""
    ai_keywords = {
        "machine learning", "ml", "nlp", "search", "ranking", "retrieval",
        "recommendation", "embedding", "ai engineer", "data scientist",
    }
    for role in candidate.get("career_history", []):
        start_year_str = str(role.get("start_date", ""))[:4]
        try:
            start_year = int(start_year_str)
        except ValueError:
            continue
        combined = role.get("description", "").lower() + " " + role.get("title", "").lower()
        if start_year < 2022 and any(kw in combined for kw in ai_keywords):
            return True
    return False


def compute_production_raw(candidate: dict) -> float:
    """Score based on production evidence keywords + v6 temporal penalty."""
    texts = [candidate["profile"]["summary"]]
    texts += [r["description"] for r in candidate["career_history"]]
    full  = " ".join(texts).lower()

    t1  = sum(1 for kw in PROD_T1 if kw in full)
    t2  = sum(1 for kw in PROD_T2 if kw in full)
    neg = sum(1 for kw in RESEARCH_NEGS if kw in full)

    t1s  = min(t1 / 5.0, 1.0)
    t2s  = min(t2 / 8.0, 0.60)
    base = 0.65 * t1s + 0.35 * t2s
    res  = min(neg * 0.05, 0.20)

    gh   = candidate["redrob_signals"]["github_activity_score"]
    gh_c = 0.10 if gh >= 60 else 0.05 if gh >= 30 else 0.0

    # v6: temporal penalty — post-2022-only ML experience is mild negative
    has_ai_skills = any(
        kw in full for kw in ["embedding", "retrieval", "ranking", "nlp", "machine learning"]
    )
    temporal_penalty = 0.0
    if has_ai_skills and not _has_pre_llm_ai_experience(candidate):
        temporal_penalty = 0.05  # mild — not a gate

    return max(0.0, min(base - res + gh_c - temporal_penalty, 1.0))


def compute_skill_raw(candidate: dict, skill_embed_map: dict,
                      jd_skill_embs: np.ndarray) -> float:
    """Raw (un-normalized) skill score."""
    total = 0.0
    sig   = candidate["redrob_signals"]
    for skill in candidate["skills"]:
        emb = skill_embed_map.get(skill["name"])
        if emb is None:
            continue
        max_sim = float(np.max(np.dot(jd_skill_embs, emb)))
        if max_sim < 0.35:
            continue
        prof     = PROFICIENCY_WEIGHT.get(skill["proficiency"], 0.75)
        duration = min(skill.get("duration_months", 0) / 36.0, 1.0)
        endorses = min(1 + skill["endorsements"] / 40.0, 1.25)
        raw_a    = sig["skill_assessment_scores"].get(skill["name"], None)
        assess   = 1.0 if raw_a is None else 1.0 + (raw_a - 50) / 150
        total   += max_sim * prof * (0.65 + 0.35 * duration) * endorses * assess
    return total  # raw — NOT divided by denominator yet


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings and normalization constants.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found.")
        return

    print("=" * 60)
    print("Redrob AI Candidate Ranker — precompute.py (v6)")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading sentence transformer: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # ── Encode JD ───────────────────────────────────────────────────────────
    print("[2/5] Encoding JD text and skill anchors ...")
    jd_emb        = model.encode(JD_EXPANDED,       normalize_embeddings=True)
    jd_skill_embs = model.encode(JD_SKILL_ANCHORS,  normalize_embeddings=True)
    np.save(ARTIFACTS / "jd_emb.npy",        jd_emb)
    np.save(ARTIFACTS / "jd_skill_embs.npy", jd_skill_embs)
    print(f"   JD embedding shape:          {jd_emb.shape}")
    print(f"   Skill anchors shape:         {jd_skill_embs.shape}")

    # ── Load candidates ─────────────────────────────────────────────────────
    print("[3/5] Loading candidates ...")
    candidates = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"   Loaded {len(candidates):,} candidates")

    # ── Encode candidate text ────────────────────────────────────────────────
    print("[4/5] Encoding candidate text (headline+summary and career) ...")
    print("   Truncating to 500/1000 chars (MiniLM max is 256 tokens) ...")
    summaries, careers, cids = [], [], []
    for c in candidates:
        p = c["profile"]
        # Truncate: MiniLM has 256-token limit — long strings waste tokenizer time
        summary_text = (p["headline"] + " " + p["summary"])[:500]
        career_text  = " ".join(r["description"] for r in c["career_history"])[:1000]
        summaries.append(summary_text)
        careers.append(career_text)
        cids.append(c["candidate_id"])

    summary_embs = model.encode(
        summaries, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    )
    career_embs = model.encode(
        careers, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    )

    np.save(ARTIFACTS / "summary_embs.npy", summary_embs)
    np.save(ARTIFACTS / "career_embs.npy",  career_embs)
    np.save(ARTIFACTS / "cid_order.npy",    np.array(cids, dtype=object))
    print(f"   summary_embs shape: {summary_embs.shape}")
    print(f"   career_embs shape:  {career_embs.shape}")

    # ── Encode unique skills ─────────────────────────────────────────────────
    print("   Encoding unique skill names ...")
    unique_skills = list({s["name"] for c in candidates for s in c["skills"]})
    skill_embs    = model.encode(
        unique_skills, batch_size=256, normalize_embeddings=True, show_progress_bar=False
    )
    skill_embed_map = dict(zip(unique_skills, skill_embs))
    with open(ARTIFACTS / "skill_embed_map.pkl", "wb") as f:
        pickle.dump(skill_embed_map, f)
    print(f"   Encoded {len(unique_skills):,} unique skills")

    # ── Measure normalization constants ───────────────────────────────────────
    print("\n[5/5] Measuring score distributions for normalization constants ...")

    # 1. Semantic score range
    print("   Computing semantic scores ...")
    raw_semantics = [
        0.35 * float(np.dot(jd_emb, s)) + 0.65 * float(np.dot(jd_emb, c))
        for s, c in zip(summary_embs, career_embs)
    ]
    sem_p1  = float(np.percentile(raw_semantics, 1))
    sem_p99 = float(np.percentile(raw_semantics, 99))
    print(f"   Semantic raw range : [{min(raw_semantics):.4f}, {max(raw_semantics):.4f}]")
    print(f"   Semantic P1 / P99  : [{sem_p1:.4f}, {sem_p99:.4f}]")
    spread = sem_p99 - sem_p1
    verdict = "COMPRESSED — rescaling is critical" if spread < 0.30 else "OK"
    print(f"   Effective spread   : {spread:.4f}  ({verdict})")

    # 2. Skill total distribution
    print("   Computing skill totals (may take ~60 sec) ...")
    raw_skill_totals = []
    for c in tqdm(candidates, desc="   Skill totals", ncols=70):
        raw_skill_totals.append(compute_skill_raw(c, skill_embed_map, jd_skill_embs))
    skill_p90 = float(np.percentile(raw_skill_totals, 90))
    skill_p95 = float(np.percentile(raw_skill_totals, 95))
    skill_p99 = float(np.percentile(raw_skill_totals, 99))
    print(f"   Skill  P50={np.percentile(raw_skill_totals,50):.3f}  "
          f"P90={skill_p90:.3f}  P95={skill_p95:.3f}  P99={skill_p99:.3f}")
    skill_denom = skill_p95  # P95: preserves headroom at top of pool
    print(f"   Using skill denominator (P95): {skill_denom:.4f}")

    # 3. Production score distribution
    print("   Computing production scores ...")
    raw_production = [
        compute_production_raw(c)
        for c in tqdm(candidates, desc="   Production  ", ncols=70)
    ]
    prod_p50 = float(np.percentile(raw_production, 50))
    prod_p90 = float(np.percentile(raw_production, 90))
    prod_p99 = float(np.percentile(raw_production, 99))
    print(f"   Production P50={prod_p50:.3f}  P90={prod_p90:.3f}  P99={prod_p99:.3f}")

    if prod_p99 < 0.50:
        print(f"   WARNING: Production P99={prod_p99:.3f} — score is compressed.")
        print(f"            PROD_RESCALE=True will activate in rank.py")
        prod_rescale = True
    else:
        prod_rescale = False
        print(f"   Production range OK — no rescaling needed.")

    # ── Save normalization constants ─────────────────────────────────────────
    constants = {
        "semantic_p1":        sem_p1,
        "semantic_p99":       sem_p99,
        "skill_denom":        skill_denom,
        "production_p99":     prod_p99,
        "production_rescale": prod_rescale,
        "reference_date":     "2026-06-17",
        "model_name":         MODEL_NAME,
        "n_candidates":       len(candidates),
        "n_unique_skills":    len(unique_skills),
    }
    with open(ARTIFACTS / "normalization_constants.json", "w") as f:
        json.dump(constants, f, indent=2)

    print("\n" + "=" * 60)
    print("PRECOMPUTE COMPLETE")
    print("=" * 60)
    print(f"\nNormalization constants saved:")
    for k, v in constants.items():
        print(f"   {k}: {v}")
    print(f"\nArtifacts written to: {ARTIFACTS.resolve()}/")
    print("  jd_emb.npy, jd_skill_embs.npy")
    print("  summary_embs.npy, career_embs.npy, cid_order.npy")
    print("  skill_embed_map.pkl, normalization_constants.json")
    print("\nNext: python rank.py --candidates candidates.jsonl --out submission.csv")

    # ── Quick sanity check on semantic spread ────────────────────────────────
    if spread < 0.20:
        print(f"\n⚠️  ALERT: Semantic spread={spread:.4f} is very narrow.")
        print("    P1/P99 rescaling is essential. rank.py handles this automatically.")
    elif spread < 0.30:
        print(f"\n⚠️  NOTE: Semantic spread={spread:.4f} is somewhat compressed.")
        print("    P1/P99 rescaling is active and important.")
    else:
        print(f"\n✅  Semantic spread={spread:.4f} — good separation expected.")


if __name__ == "__main__":
    main()
