# FINAL LOCKED Implementation Plan — Redrob AI Candidate Ranker
## Version 7 — Refined Scoring and Validation Alignment

---

## Change Log from v6 → v7

| Change Category | Details | Rationale |
|---|---|---|
| **Country Extraction Fix** (Priority 0) | Change profile extraction from `current_location.country` to `profile.country` | The schema has `country` at the root of `profile`. The previous code extracted `""` for all candidates, breaking location scoring. |
| **Ownership Score** (Priority 1) | Add `compute_ownership_score` (weight: 6%) using keywords like `owned`, `architected`, `e2e` | JD repeatedly emphasizes end-to-end ownership, building from scratch, and driving migrations. |
| **Evaluation Score** (Priority 2) | Add `compute_eval_score` (weight: 3%) using keywords like `ndcg`, `mrr`, `ab testing` | JD explicitly highlights rigor in ranking evaluation. These were previously in general production keywords. |
| **Company Expansion** (Priority 3 & 7) | Expand `COMPANY_SCORES` with Google, Meta, Amazon, Microsoft, Uber, LinkedIn, Airbnb, and product companies | Matches top-tier engineering talent backgrounds and avoids defaulting them to 0.65 neutral score. |
| **Enriched Reasoning** (Priority 4) | Match elite JD keywords in summary, career history, and skills list for reasoning strings | Helps the human judge immediately recognize fit signals like `Hybrid retrieval`, `BM25`, `NDCG`, `MRR`. |
| **Production-Aware Research Penalty** (Priority 5) | Apply `0.95` multiplier if title contains "research" AND production score is low (< 0.40) | JD states: "reject pure researchers without production deployment." Research engineers are kept. |
| **Rebalance Weights** (Priority 6) | Adjust base score weights: Semantic 24%, Skill 20%, Production 20%, Career 15%, Title 10%, Ownership 6%, Eval 3%, Education 2%, Location 2% | Shifts focus from education and location to ownership, evaluation, and production systems. |
| **CSV Validation Alignment** | Sort and output exactly top-100 candidates with columns: `candidate_id,rank,score,reasoning` | Aligns output file with `validate_submission.py` rules (exactly 100 rows, rank column included). |

---

## Change Log from v5 → v6

| Change | Reason |
|---|---|
| `PROD_T1` expanded with evaluation vocabulary | Raw JD: "If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful" — evaluation is T1-level evidence |
| `PROD_T2` expanded with BM25-migration signals | Raw JD 90-day plan: audit BM25 → ship hybrid retrieval. Evidence of this exact transition is the highest-value career signal |
| `FRAMEWORK_ONLY_SIGNALS` added to RESEARCH_NEGS | Raw JD explicitly names "LangChain tutorials and demo projects" as anti-pattern — distinct from pure research negs |
| Title velocity penalty in `compute_career_depth_score` | Raw JD: "title-chasers switching companies every 1.5 years" is named anti-pattern. avg tenure < 20 months + 3+ title levels → −0.05 penalty |
| Temporal AI signal in `compute_production_raw` | Raw JD: "people who understood retrieval and ranking before it became fashionable" — post-2022-only ML experience is mild negative |
| `JD_EXPANDED` strengthened with migration + eval language | Embeds the semantic of the 90-day mandate and evaluation emphasis directly into the JD vector |
| Domain misalignment note in RESEARCH_NEGS comments | Raw JD: CV/speech/robotics without NLP/IR is explicit disqualifier — captured via low semantic sim but noted for clarity |
| Closed-source penalty note | Raw JD: 5+ years closed-source without papers/talks/OSS is named concern — GitHub absence for senior engineers carries stronger signal |

Everything else from v5 is unchanged and locked.

---

## Full EDA Results — Every Design Decision Traces Back Here

All numbers below are from actual runs on the full 100,000-candidate `candidates.jsonl`.
These are not estimates. Every architectural choice is justified by a number in this section.

### Dataset Scale

```
Total candidates:                         100,000
AI-relevant (≥4 AI skills OR AI title):    20,041   ← real competition pool
In target YoE band (4–9 years):            10,912
Behaviorally available:                     2,017   ← open_to_work + active < 90d
India-based + available:                    1,575
Good GitHub score (≥30):                      653   ← true top-tier pool
Pure IT services background only:           1,555   ← soft penalty, not hard disqualify
```

> **Key architectural implication:** The real contest for top 10 is among ~650 candidates,
> not 100,000. Everything else is noise removal. This is why behavioral signals matter so
> much — they narrow 100K to 2K before any ML runs.

### Title Distribution (Current Role)

```
Business Analyst:          5,833  ← Non-AI titles dominate by large margin
HR Manager:                5,830
Mechanical Engineer:       5,791
Accountant:                5,764
Project Manager:           5,754
Customer Support:          5,750
Operations Manager:        5,744
Content Writer:            5,727
Sales Executive:           5,713
Civil Engineer:            5,702
Graphic Designer:          5,689
Marketing Manager:         5,524
─────────────────────────────────
Software Engineer:         3,450  ← Engineering titles start here
Full Stack Developer:      2,873
Cloud Engineer:            2,836
Java Developer:            2,809
.NET Developer:            2,788
DevOps Engineer:           2,787
Mobile Developer:          2,757
Frontend Engineer:         2,738
QA Engineer:               2,682
─────────────────────────────────
Senior Data Engineer:        687
Backend Engineer:            704
Senior Software Engineer:    653
Analytics Engineer:          305
Data Engineer:               295
Data Analyst:                272
─────────────────────────────────  ← True AI titles (small minority)
AI Research Engineer:        153
Senior Software Engineer(ML):142
Computer Vision Engineer:    132
AI Specialist:               130
Recommendation Systems Eng:   26
Machine Learning Engineer:    24
Applied ML Engineer:          23
Search Engineer:              23
```

> **Why title is a 12% soft gradient (not a gate):**
> The majority of the dataset has non-AI titles. Many true AI engineers appear
> under titles like "Backend Engineer" or "Software Engineer".
> A "Search Engineer" or "Recommendation Systems Engineer" is an exact role match —
> they cannot be scored like a Business Analyst (0.35). Fixed in title dict.

### Skill Distribution (Top 15 by Frequency)

```
HTML:             12,246  ← Pure noise for this JD
Databricks:       12,244
Redux:            12,222
Terraform:        12,187
Angular:          12,173
Figma:            12,157
Salesforce CRM:   12,157
Vue.js:           12,142
Sales:            12,138
Accounting:       12,136
Agile:            12,135
Kafka:            12,114
Excel:            12,109
BigQuery:         12,108
CI/CD:            12,108
```

> **Why skill matching must be semantic (not keyword count):**
> The most common skills in the dataset are ALL irrelevant to this JD.
> Counting skills or matching by keyword would reward candidates with HTML/Redux
> over candidates with FAISS/Sentence-Transformers. The semantic anchor approach
> filters below 0.35 cosine similarity — these skills all fall below that floor.

### YoE Distribution (Full Pool)

```
0–3 years:    16,120  (16.1%)
3–6 years:    26,398  (26.4%)  ← largest bucket
6–9 years:    24,963  (25.0%)  ← JD sweet spot (4-9yr)
9–12 years:   17,648  (17.6%)
12–15 years:  14,620  (14.6%)
15+ years:       251   (0.3%)
```

**AI pool YoE percentiles:**
```
P25: 3.5 years
P50: 5.7 years  ← median AI candidate
P75: 7.9 years
P90: 9.8 years
```

> **Why YoE scores 1.0 for 4–9yr band:**
> P50 of AI pool = 5.7yr, P75 = 7.9yr. The JD says "5-9 years" but explicitly
> accepts 4yr candidates with strong signals. The 4–9yr band captures the
> median-to-P75 range of the real AI candidate pool.

### Country Distribution

```
India:       75,113  (75.1%)  ← dominant
USA:          9,978  (10.0%)
Australia:    2,579  (2.6%)
Canada:       2,506  (2.5%)
UK:           2,472  (2.5%)
Germany:      2,469  (2.5%)
Singapore:    2,453  (2.5%)
UAE:          2,430  (2.4%)
```

> **Why location is 5% additive (not a multiplier):**
> 75% of candidates are already India-based. A Singapore candidate at 0.97
> overall capability cannot be zeroed by geography when the JD says "case-by-case"
> for outside India. Location is a tiebreaker, not a gate.

### Current Industry Distribution

```
IT Services:     29,881  ← largest — includes TCS/Infosys/Wipro candidates
Software:        22,417
Manufacturing:   22,305
Conglomerate:     7,571
Paper Products:   7,467
Fintech:          2,808
Food Delivery:    2,514
E-commerce:       1,529
Consulting:       1,274
EdTech:             610
SaaS:               690
AI/ML:              597
AdTech:             376
```

### Behavioral Signal Baselines

```
Open to work:           35,339  (35.3%)  ← only 1/3 actively looking
GitHub linked:          35,363  (35.4%)  ← 2/3 have no GitHub
Average notice period:    87.4  days     ← far above JD's desired <30d
Average response rate:    0.44           ← 44% response baseline
Short notice (≤30d):    13.4%  of pool   ← rare, valuable differentiator
Mid notice (31–60d):    18.9%
Long notice (>90d):     33.0%           ← majority have long notice
```

> **Why notice period is weighted:** Only 13.4% of candidates have ≤30d notice.
> The JD explicitly says "we'd love sub-30-day notice" and "can buy out up to 30 days".
> This is genuinely rare and the JD specifically rewards it.

> **Why GitHub matters for this specific role:** 35.4% have GitHub linked.
> For a Senior AI Engineer role, no GitHub is a mild negative signal.
> The JD explicitly worries about people who "haven't written production code
> in 18 months" — GitHub activity corroborates active coding.

### Company Landscape (Career History Appearances)

```
Infosys:         23,722  ← ~24% of all career roles touch Infosys
Wipro:           23,682
Pied Piper:      23,614  (synthetic product company)
Initech:         23,590  (synthetic)
Wayne Enterprises:23,556 (synthetic)
Acme Corp:       23,546  (synthetic)
Stark Industries: 23,524  (synthetic)
Hooli:           23,509  (synthetic)
TCS:             23,483
Globex Inc:      23,471  (synthetic)
Dunder Mifflin:  23,416  (synthetic)
─────────────────────────────────────────
Swiggy:           3,019  ← real product companies
Razorpay:         2,926
CRED:             2,908
Capgemini:        2,895
```

> **Why services companies are soft-penalized (0.50) not eliminated:**
> Infosys, Wipro, TCS together appear in 70,000+ career history slots.
> Almost every candidate has at least one services company in their history.
> Hard disqualification would eliminate most of the dataset including legitimate
> candidates who started at TCS and moved to Swiggy/Razorpay/CRED.
> The company score is tenure-weighted — a 2yr Infosys stint early in career
> barely moves the needle if followed by 4yr at a product company.

### Industry Values in Career History (Actual Strings)

```
'IT Services'         88,077   ← substring match: 'it services' or 'software' ok
'Software'            70,746
'Manufacturing'       70,541
'Conglomerate'        23,556
'Paper Products'      23,416
'Fintech'              6,513   ← relevant industry
'Food Delivery'        5,902   ← relevant (Swiggy)
'E-commerce'           3,644   ← relevant
'Consulting'           2,871
'EdTech'               1,384   ← relevant
'SaaS'                   690   ← relevant
'AI/ML'                  597   ← highly relevant
'AdTech'                 376   ← relevant
'Transportation'         364
'Insurance Tech'         355   ← relevant
'HealthTech'             339   ← relevant
'Gaming'                 329
'HealthTech AI'          132   ← highly relevant
'Conversational AI'      124   ← highly relevant
'AI Services'             81   ← highly relevant
```

> **Why industry recency bonus uses exact string matching:**
> EDA confirmed these are the literal string values in the data — no normalization
> or free-text. Substring matching on lowercased values is reliable here.

### Consistency Score Distribution (Critical Finding)

```
Candidates scoring exactly 1.0:    99,955  (99.96%)
Candidates below 0.70:                 43  (0.04%)  ← the real honeypots
Candidates below 0.60:                 22  (0.02%)
Standard deviation:                 0.009  ← near-zero signal
```

> **Why consistency is a GATE-ONLY signal (not a scoring component):**
> With std=0.009 and 99.96% of candidates at 1.0, adding it as a weighted
> component would contribute near-zero differentiation while consuming a weight
> percentage. It works correctly and exclusively as a hard gate at 0.70,
> catching the 43 genuinely anomalous profiles.

### offer_acceptance_rate Distribution

```
oar = -1 (no offer history):   59,554  (59.6%)
oar in [0, 1]:                 40,446  (40.4%)
```

> **Why oar=-1 maps to 0.50 (neutral) not 0.0:**
> 60% of candidates have never received an offer — treating them as oar=0
> (negative) systematically penalizes the majority for having no offer history,
> which is irrelevant to their fit. Neutral (0.50) is the correct default.
> Verified safe: top-20 overlap = 18/20 before and after the fix.

### Work Mode Preference

```
hybrid:    25,076  (25.1%)
onsite:    25,000  (25.0%)
flexible:  25,000  (25.0%)
remote:    24,924  (24.9%)
```

> **Work mode is NOT a scoring component:** Distribution is uniform — 25% each.
> The JD says "hybrid, flexible cadence" which matches 50% of candidates equally.
> Adding it as a signal adds noise, not signal.

---

---

## Scoring Formula — Final (v7)

```
FINAL_SCORE =
    0.80 × BASE_SCORE
  + 0.20 × BEHAVIORAL_SCORE

BASE_SCORE = (
    0.24 × semantic_score      ← P1/P99 rescaled cosine similarity (split 35/65)
  + 0.20 × skill_score         ← P95-normalized semantic anchor matching
  + 0.20 × production_score    ← P95-checked keyword evidence score
  + 0.15 × career_depth_score  ← YoE + company gradient + stability
  + 0.10 × title_score         ← Full-coverage soft gradient
  + 0.06 × ownership_score     ← Ownership keyword signal
  + 0.03 × eval_score          ← Evaluation framework signal
  + 0.02 × education_score     ← Tier + field relevance
  + 0.02 × location_score      ← Additive component, not multiplier
) / 1.02  [scaled to keep BASE_SCORE strictly in [0, 1]]

BEHAVIORAL_SCORE = (
    0.30 × availability_score  ← recency + open_to_work
  + 0.25 × engagement_score    ← rr + icr + oar (−1 → 0.50 neutral)
  + 0.20 × momentum_score      ← 30-day: saves + apps + views + search
  + 0.15 × github_score        ← github_activity_score; −1 → 0.15 floor
  + 0.10 × notice_score        ← notice_period alignment
)

HARD GATE: consistency_score < 0.70 → FINAL_SCORE = 0.0

POST-PROCESSING MODIFIERS:
- If current title contains "research" AND production_score < 0.40 → FINAL_SCORE *= 0.95
```

---

## Phase 1: precompute.py — Full Specification

This runs **once**, offline, before rank.py is ever called.
It does two things: (1) build embedding artifacts, (2) measure normalization constants.

```python
#!/usr/bin/env python3
"""
precompute.py
Builds all artifacts rank.py depends on.
Usage: python precompute.py --candidates candidates.jsonl
"""
import argparse, json, pickle, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
ARTIFACTS  = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

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

JD_SKILL_ANCHORS = [
    "production embeddings retrieval semantic search vector database",
    "NDCG MRR MAP ranking evaluation offline benchmark online A/B experiment recruiter feedback",
    "Python production code software engineering quality",
    "LLM fine-tuning LoRA QLoRA PEFT language model",
    "NLP natural language processing information retrieval dense retrieval BM25 hybrid search",
    "recommendation systems candidate matching personalization ranking search improvement",
    "distributed systems inference optimization scalability latency"
]

PROD_T1 = [
    # Core production evidence (original)
    "production", "deployed", "real users", "at scale", "serving",
    "latency", "a/b test", "throughput", "millions of", "live system",
    "queries per second", "qps", "p99", "p95", "sla", "uptime",
    "index refresh", "embedding drift", "retrieval quality regression",
    "model monitoring", "inference pipeline", "feature store",
    # v6 additions — evaluation vocabulary (JD: "rigorously evaluate ranking")
    "offline benchmark", "online experiment", "ndcg", "mrr",
    "precision at k", "recall at k", "click-through", "recruiter feedback",
    "a/b experiment", "evaluation framework", "ranking quality",
]
PROD_T2 = [
    # Core ownership signals (original)
    "shipped", "launched", "built end-to-end", "owned", "led design",
    "search", "ranking", "retrieval", "recommendation", "matching",
    "pipeline", "infrastructure", "architecture",
    # v6 additions — BM25-to-hybrid migration signals (JD 90-day mandate)
    "bm25", "tf-idf", "keyword search", "migrated", "migrated from",
    "replaced", "improved precision", "improved recall", "reduced latency",
    "recruiter engagement", "click-through rate", "conversion rate",
    "hybrid retrieval", "hybrid search", "re-ranking", "reranking",
]
RESEARCH_NEGS = [
    # Pure research anti-patterns (original)
    "arxiv", "research lab", "academic", "thesis",
    "proof of concept", "prototype only", "exploration paper",
    # v6 additions — Framework-enthusiast anti-patterns (JD: "LangChain tutorials")
    # NOTE: These are only negative when they are the PRIMARY evidence.
    # The penalty is intentionally mild (same 0.05 per hit, cap 0.20 total)
    # so a candidate with real production work isn't hurt by one tutorial post.
    "langchain tutorial", "how i built", "getting started with",
    "demo project", "side project only",
]

PROFICIENCY_WEIGHT = {
    "beginner": 0.50, "intermediate": 0.75, "advanced": 1.00, "expert": 1.10
}

# v6: Temporal AI signal — detect post-2022-only ML experience
# JD says: "people who understood retrieval and ranking before it became fashionable"
def _has_pre_llm_ai_experience(candidate: dict) -> bool:
    """Returns True if candidate has AI/ML/search role starting before 2022."""
    ai_keywords = {"machine learning", "ml", "nlp", "search", "ranking", "retrieval",
                   "recommendation", "embedding", "ai engineer", "data scientist"}
    for role in candidate.get("career_history", []):
        start_year_str = str(role.get("start_date", ""))[:4]
        try:
            start_year = int(start_year_str)
        except ValueError:
            continue
        desc_lower = role.get("description", "").lower()
        title_lower = role.get("title", "").lower()
        combined = desc_lower + " " + title_lower
        if start_year < 2022 and any(kw in combined for kw in ai_keywords):
            return True
    return False

def compute_production_raw(candidate: dict) -> float:
    texts = [candidate["profile"]["summary"]]
    texts += [r["description"] for r in candidate["career_history"]]
    full  = " ".join(texts).lower()
    t1    = sum(1 for kw in PROD_T1 if kw in full)
    t2    = sum(1 for kw in PROD_T2 if kw in full)
    neg   = sum(1 for kw in RESEARCH_NEGS if kw in full)
    t1s   = min(t1 / 5.0, 1.0)
    t2s   = min(t2 / 8.0, 0.60)
    base  = 0.65 * t1s + 0.35 * t2s
    res   = min(neg * 0.05, 0.20)
    gh    = candidate["redrob_signals"]["github_activity_score"]
    gh_c  = 0.10 if gh >= 60 else 0.05 if gh >= 30 else 0.0
    # v6: Temporal signal — post-2022-only ML experience is mild negative
    # Only applies if: candidate has AI/ML skills AND no pre-2022 AI role found
    has_ai_skills = any(
        kw in full for kw in ["embedding", "retrieval", "ranking", "nlp", "machine learning"]
    )
    temporal_penalty = 0.0
    if has_ai_skills and not _has_pre_llm_ai_experience(candidate):
        temporal_penalty = 0.05  # mild — not a gate
    return max(0.0, min(base - res + gh_c - temporal_penalty, 1.0))

def compute_skill_raw(candidate: dict, skill_embed_map: dict,
                      jd_skill_embs: np.ndarray) -> float:
    total = 0.0
    sig   = candidate["redrob_signals"]
    for skill in candidate["skills"]:
        emb = skill_embed_map.get(skill["name"])
        if emb is None:
            continue
        max_sim = float(np.max(np.dot(jd_skill_embs, emb)))
        if max_sim < 0.35:
            continue
        prof     = PROFICIENCY_WEIGHT[skill["proficiency"]]
        duration = min(skill.get("duration_months", 0) / 36.0, 1.0)
        endorses = min(1 + skill["endorsements"] / 40.0, 1.25)
        raw_a    = sig["skill_assessment_scores"].get(skill["name"], None)
        assess   = 1.0 if raw_a is None else 1.0 + (raw_a - 50) / 150
        total   += max_sim * prof * (0.65 + 0.35 * duration) * endorses * assess
    return total   # raw — NOT divided by denominator yet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    print("Loading model …")
    model = SentenceTransformer(MODEL_NAME)

    # ── JD embeddings ────────────────────────────────────────────────────────
    print("Encoding JD …")
    jd_emb        = model.encode(JD_EXPANDED,    normalize_embeddings=True)
    jd_skill_embs = model.encode(JD_SKILL_ANCHORS, normalize_embeddings=True)
    np.save(ARTIFACTS / "jd_emb.npy",        jd_emb)
    np.save(ARTIFACTS / "jd_skill_embs.npy", jd_skill_embs)

    # ── Load all candidates ──────────────────────────────────────────────────
    print("Loading candidates …")
    candidates = []
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates):,} candidates")

    # ── Encode summaries and career descriptions ─────────────────────────────
    print("Encoding candidate text (this takes ~3-4 min on CPU) …")
    summaries, careers, cids = [], [], []
    for c in candidates:
        p = c["profile"]
        summaries.append(p["headline"] + " " + p["summary"])
        careers.append(" ".join(r["description"] for r in c["career_history"]))
        cids.append(c["candidate_id"])

    summary_embs = model.encode(summaries, batch_size=128,
                                normalize_embeddings=True, show_progress_bar=True)
    career_embs  = model.encode(careers,   batch_size=128,
                                normalize_embeddings=True, show_progress_bar=True)
    np.save(ARTIFACTS / "summary_embs.npy", summary_embs)
    np.save(ARTIFACTS / "career_embs.npy",  career_embs)
    np.save(ARTIFACTS / "cid_order.npy",    np.array(cids))
    print("  Candidate embeddings saved.")

    # ── Encode unique skill names ─────────────────────────────────────────────
    print("Encoding skill names …")
    unique_skills = list({s["name"] for c in candidates for s in c["skills"]})
    skill_embs    = model.encode(unique_skills, batch_size=256,
                                 normalize_embeddings=True, show_progress_bar=False)
    skill_embed_map = dict(zip(unique_skills, skill_embs))
    with open(ARTIFACTS / "skill_embed_map.pkl", "wb") as f:
        pickle.dump(skill_embed_map, f)
    print(f"  Encoded {len(unique_skills)} unique skills.")

    # ── MEASURE NORMALIZATION CONSTANTS ──────────────────────────────────────
    print("\nMeasuring score distributions …")

    # 1. Semantic score range
    raw_semantics = [
        0.35 * float(np.dot(jd_emb, s)) + 0.65 * float(np.dot(jd_emb, c))
        for s, c in zip(summary_embs, career_embs)
    ]
    sem_p1  = float(np.percentile(raw_semantics, 1))
    sem_p99 = float(np.percentile(raw_semantics, 99))
    print(f"  Semantic raw range : [{min(raw_semantics):.3f}, {max(raw_semantics):.3f}]")
    print(f"  Semantic P1/P99    : [{sem_p1:.3f}, {sem_p99:.3f}]")
    print(f"  Effective range    : {sem_p99 - sem_p1:.3f}  "
          f"({'COMPRESSED — rescaling important' if (sem_p99-sem_p1) < 0.7 else 'OK'})")

    # 2. Skill total distribution
    print("  Computing skill totals (may take ~60 sec) …")
    raw_skill_totals = []
    for c in tqdm(candidates, desc="  Skill totals"):
        raw_skill_totals.append(compute_skill_raw(c, skill_embed_map, jd_skill_embs))

    skill_p90 = float(np.percentile(raw_skill_totals, 90))
    skill_p95 = float(np.percentile(raw_skill_totals, 95))
    skill_p99 = float(np.percentile(raw_skill_totals, 99))
    print(f"  Skill total P50={np.percentile(raw_skill_totals,50):.3f}  "
          f"P90={skill_p90:.3f}  P95={skill_p95:.3f}  P99={skill_p99:.3f}")
    skill_denom = skill_p95  # Use P95: top candidates need headroom
    print(f"  Using skill denominator: {skill_denom:.3f} (P95)")

    # 3. Production score distribution
    print("  Computing production scores …")
    raw_production = [compute_production_raw(c) for c in tqdm(candidates, desc="  Production")]
    prod_p50 = float(np.percentile(raw_production, 50))
    prod_p90 = float(np.percentile(raw_production, 90))
    prod_p99 = float(np.percentile(raw_production, 99))
    print(f"  Production P50={prod_p50:.3f}  P90={prod_p90:.3f}  P99={prod_p99:.3f}")
    if prod_p99 < 0.50:
        print(f"  WARNING: Production P99 = {prod_p99:.3f} — score is compressed.")
        print(f"           Consider rescaling production score similarly to semantic.")
        prod_rescale = True
    else:
        prod_rescale = False
        print(f"  Production range OK — no rescaling needed.")

    # ── SAVE NORMALIZATION CONSTANTS ─────────────────────────────────────────
    constants = {
        "semantic_p1":     sem_p1,
        "semantic_p99":    sem_p99,
        "skill_denom":     skill_denom,
        "production_p99":  prod_p99,
        "production_rescale": prod_rescale,
        "reference_date":  "2026-06-17"
    }
    with open(ARTIFACTS / "normalization_constants.json", "w") as f:
        json.dump(constants, f, indent=2)
    print(f"\n  Saved normalization_constants.json")
    print(f"  Constants: {json.dumps(constants, indent=4)}")
    print("\nPrecompute complete. Run rank.py next.")


if __name__ == "__main__":
    main()
```

---

## How rank.py Uses Constants (Never Recomputes)

```python
# rank.py startup — load everything from artifacts/
import json, numpy as np, pickle
from pathlib import Path

ARTIFACTS = Path("artifacts")

# Load pre-computed embeddings
jd_emb         = np.load(ARTIFACTS / "jd_emb.npy")
jd_skill_embs  = np.load(ARTIFACTS / "jd_skill_embs.npy")
summary_embs   = np.load(ARTIFACTS / "summary_embs.npy")
career_embs    = np.load(ARTIFACTS / "career_embs.npy")
cid_order      = np.load(ARTIFACTS / "cid_order.npy")
with open(ARTIFACTS / "skill_embed_map.pkl", "rb") as f:
    skill_embed_map = pickle.load(f)

# Load normalization constants — NEVER recompute these
with open(ARTIFACTS / "normalization_constants.json") as f:
    CONSTS = json.load(f)

SEM_P1        = CONSTS["semantic_p1"]
SEM_P99       = CONSTS["semantic_p99"]
SKILL_DENOM   = CONSTS["skill_denom"]
PROD_P99      = CONSTS["production_p99"]
PROD_RESCALE  = CONSTS["production_rescale"]
```

---

## Updated Component Functions (Constants-Aware)

### Semantic Score — P1/P99 Rescaled

```python
def compute_semantic_score(i: int) -> float:
    """i = index into pre-loaded summary_embs / career_embs arrays"""
    raw = (0.35 * float(np.dot(jd_emb, summary_embs[i])) +
           0.65 * float(np.dot(jd_emb, career_embs[i])))
    # P1/P99 clip: robust to outliers, empirically grounded
    rescaled = (raw - SEM_P1) / (SEM_P99 - SEM_P1)
    return float(np.clip(rescaled, 0.0, 1.0))
```

### Skill Score — P95 Denominator

```python
def compute_skill_score(candidate: dict) -> float:
    total = compute_skill_raw(candidate, skill_embed_map, jd_skill_embs)
    # SKILL_DENOM = P95 of population (from precompute)
    # Top candidates get headroom; P95 candidate scores 1.0
    return float(np.clip(total / SKILL_DENOM, 0.0, 1.0))
```

### Production Score — Conditionally Rescaled

```python
def compute_production_score(candidate: dict) -> float:
    raw = compute_production_raw(candidate)
    if PROD_RESCALE:
        # Only if precompute found P99 < 0.50
        rescaled = raw / PROD_P99
        return float(np.clip(rescaled, 0.0, 1.0))
    return raw  # Already in [0, 1] with good spread
```

### Master Scorer — No Magic Constants

```python
# v6: Title velocity penalty helper
# JD explicitly names "title-chasers" (Senior→Staff→Principal by switching cos every 1.5yr)
def _compute_title_velocity_penalty(candidate: dict) -> float:
    """Returns a small penalty if candidate shows title-chaser pattern.
    Pattern: avg tenure across companies < 20 months AND 3+ distinct title levels.
    """
    TITLE_LEVEL = {
        "intern": 0, "junior": 1, "associate": 1,
        "engineer": 2, "developer": 2, "analyst": 2,
        "senior": 3, "lead": 3,
        "staff": 4, "principal": 4, "architect": 4,
        "director": 5, "vp": 6, "head": 5,
    }
    history = candidate.get("career_history", [])
    if len(history) < 3:
        return 0.0
    # Compute avg tenure in months
    tenures = [r.get("duration_months", 0) for r in history if r.get("duration_months", 0) > 0]
    if not tenures:
        return 0.0
    avg_tenure = sum(tenures) / len(tenures)
    if avg_tenure >= 20:
        return 0.0  # Not a short-tenure pattern
    # Count distinct title levels seen
    levels_seen = set()
    for r in history:
        title_lower = r.get("title", "").lower()
        for level_kw, level_num in TITLE_LEVEL.items():
            if level_kw in title_lower:
                levels_seen.add(level_num)
                break
    if len(levels_seen) >= 3:
        return 0.05  # Mild penalty — not a gate
    return 0.0

def score_candidate(candidate: dict, i: int) -> tuple:
    # Hard gate
    if compute_consistency(candidate) < 0.70:
        return 0.0, "FILTERED: inconsistent profile signals"

    semantic   = compute_semantic_score(i)              # P1/P99 rescaled
    skill      = compute_skill_score(candidate)         # P95 denominator
    production = compute_production_score(candidate)    # conditionally rescaled
    career     = compute_career_depth_score(candidate)
    title      = compute_title_score(candidate)
    ownership  = compute_ownership_score(candidate)      # v7 ownership keywords
    evaluation = compute_eval_score(candidate)           # v7 evaluation keywords
    education  = compute_education_score(candidate)
    location   = compute_location_score(candidate)      # profile["country"] (P0 fix)

    # v6: Title velocity penalty applied to career component
    title_velocity_penalty = _compute_title_velocity_penalty(candidate)
    career_adjusted = max(0.0, career - title_velocity_penalty)

    base = (0.24 * semantic        +
            0.20 * skill           +
            0.20 * production      +
            0.15 * career_adjusted +
            0.10 * title           +
            0.06 * ownership       +
            0.03 * evaluation      +
            0.02 * education       +
            0.02 * location) / 1.02  # v7: base score scaled to [0, 1]

    behavioral = compute_behavioral_score(candidate)

    final = round(0.80 * base + 0.20 * behavioral, 4)

    # v7: Production-aware research engineer penalty
    is_research = "research" in candidate["profile"].get("current_title", "").lower()
    research_penalty_applied = False
    if is_research and production < 0.40:
        final = round(final * 0.95, 4)
        research_penalty_applied = True

    reasoning = generate_reasoning(candidate, {
        "semantic": semantic, "skill": skill, "production": production,
        "career": career_adjusted, "title": title, "behavioral": behavioral,
        "ownership": ownership, "evaluation": evaluation,
        "title_velocity_penalty": title_velocity_penalty,
        "research_penalty_applied": research_penalty_applied
    })
    return final, reasoning
```

---

## Build Order — Final Sequence

```
PHASE 1 — precompute.py (one afternoon, run once)
─────────────────────────────────────────────────
  python precompute.py --candidates candidates.jsonl

  Outputs:
    artifacts/jd_emb.npy
    artifacts/jd_skill_embs.npy
    artifacts/summary_embs.npy
    artifacts/career_embs.npy
    artifacts/cid_order.npy
    artifacts/skill_embed_map.pkl
    artifacts/normalization_constants.json   ← measured, not guessed

  Read the printed ranges carefully:
    - Semantic P1/P99 spread < 0.3  → rescaling is critical
    - Skill P95 < 1.0               → confirm SKILL_DENOM is sane
    - Production P99 < 0.5          → PROD_RESCALE = True activates

PHASE 2 — audit on 1K sample
─────────────────────────────────────────────────
  2a. python audit.py --n 1000
      → Check component score ranges (all should be [0, 1] after rescaling)

  2b. Check score distribution shape
      → Is there meaningful spread? (Top-10 gap > 0.03 is target)

  2c. Manually inspect the printed top-10 of the 1K sample
      → Ask: "Would a Redrob AI recruiter call these 10 people first?"
      → If any top-10 is obviously wrong: trace which component is misfiring

PHASE 3 — rank.py (full 100K, ~90 sec)
─────────────────────────────────────────────────
  python rank.py --candidates candidates.jsonl --out team_xxx.csv

  Constants are loaded from normalization_constants.json — never recomputed.
  If precompute confirmed ranges are OK, this step is deterministic.

PHASE 4 — validate + audit full output
─────────────────────────────────────────────────
  python validate_submission.py team_xxx.csv   → "Submission is valid."
  python audit.py --submission team_xxx.csv --top 50
  → Check top-10 gap metric
  → Manually inspect top 10 + 5 random samples from top 50
```

---

## Normalization Design Rationale (Final, Documented)

| Component | Method | Why |
|---|---|---|
| `semantic_score` | P1/P99 percentile clip | MiniLM cosine similarities compress to ~[0.1, 0.55]; min-max is outlier-sensitive |
| `skill_score` | Divide by P95 of raw totals | P90 clips top candidates; P95 preserves differentiation at top of pool |
| `production_score` | Conditional: rescale only if P99 < 0.50 | Keyword-count based; check empirically first |
| `career_depth_score` | [0, 1] by construction | Weighted sum of sub-scores, each bounded |
| `title_score` | [0, 1] by construction | Lookup table values |
| `education_score` | [0, 1] by construction | Capped at 1.0 |
| `location_score` | [0, 1] by construction | Fixed values |
| `behavioral_score` | [0, 1] by construction | All sub-components bounded |

---

## Artifacts Manifest (Everything rank.py Needs)

```
artifacts/
├── jd_emb.npy                    # JD embedding (384-dim)
├── jd_skill_embs.npy             # 7 anchor embeddings (7×384)
├── summary_embs.npy              # 100K × 384 — headline+summary
├── career_embs.npy               # 100K × 384 — career descriptions
├── cid_order.npy                 # 100K candidate IDs (row-matches .npy files)
├── skill_embed_map.pkl           # {skill_name: embedding} for ~700 unique skills
└── normalization_constants.json  # measured P1/P99/P95 values + REFERENCE_DATE
```

All committed to repo. If anyone re-runs `precompute.py`, constants update. rank.py always reads from file.

---

## Requirements

```
# requirements.txt
sentence-transformers==2.7.0
numpy==1.26.4
scikit-learn==1.4.2
tqdm==4.66.4
```

---

## Pre-Submission Checklist

```
PRECOMPUTE:
[ ] python precompute.py --candidates candidates.jsonl
[ ] Read printed ranges — confirm semantic/skill/production spread is sane
[ ] normalization_constants.json committed to artifacts/

AUDIT (1K SAMPLE):
[ ] python audit.py --n 1000
[ ] Component ranges all [0, 1] after rescaling
[ ] Score distribution has spread (top-10 gap > 0.03)
[ ] Top-10 of 1K passes manual recruiter smell test

FULL RUN:
[ ] python rank.py --candidates candidates.jsonl --out team_xxx.csv
[ ] python validate_submission.py team_xxx.csv → "Submission is valid."
[ ] audit.py on full output — top-10 gap confirmed

MANUAL INSPECTION:
[ ] Top 10 candidates reviewed individually
[ ] No honeypots in top 10
[ ] Reasoning strings are specific and non-templated
[ ] 5 random rows from top 50 spot-checked

GITHUB:
[ ] README with reproduce_command
[ ] requirements.txt pinned
[ ] artifacts/ committed
[ ] submission_metadata.yaml filled
[ ] Colab notebook = sandbox link

PORTAL:
[ ] CSV uploaded
[ ] GitHub repo + sandbox link entered
[ ] AI tools declared honestly
```

---

## v6 Design Notes — JD Signal Cross-Reference

### What the Raw JD Confirms

| JD Concern | How v6 Handles It |
|---|---|
| "Title-chasers" (switch every 1.5yr for title) | `_compute_title_velocity_penalty`: avg tenure < 20mo + 3+ title levels → −0.05 to career |
| "Framework enthusiasts" (LangChain tutorials) | `FRAMEWORK_ONLY_SIGNALS` in RESEARCH_NEGS; mild −0.05/hit, capped at −0.20 |
| "Only consulting firms their entire career" | Services penalty already tenure-weighted — current Infosys stint with prior product exp is fine |
| CV/speech/robotics without NLP/IR | Captured by low semantic similarity to JD_EXPANDED (no explicit gate needed) |
| 5+ yr closed-source, no external validation | GitHub absence for senior engineers: gh_c = 0 (no bonus); behavioral github_score = 0.15 floor |
| "Understood retrieval before it was fashionable" | `_has_pre_llm_ai_experience`: post-2022-only AI = −0.05 temporal penalty |
| 90-day mandate: BM25 → hybrid migration | PROD_T2 now includes `bm25`, `migrated`, `hybrid retrieval`, `recruiter engagement` |
| Evaluation is core, not nice-to-have | PROD_T1 now includes `ndcg`, `mrr`, `offline benchmark`, `recruiter feedback loop` |
| Behavioral availability as gate modifier | 80/20 split preserved; near-zero availability_score pushes inactive out of top-10 |

### Confirmed Non-Changes (v5 Already Correct)

- **80/20 BASE/BEHAVIORAL split** — Raw JD confirms career evidence is primary differentiator; behavioral is availability filter
- **Services penalty is tenure-weighted** — JD says "currently at Infosys but prior product exp is fine" — already handled correctly
- **Location = 5% additive, not a gate** — JD says "case-by-case" for outside India; correct treatment
- **YoE band 4–9yr at 1.0** — JD says "some hit senior judgment at 4yr" — band already starts at 4
- **consistency_score < 0.70 = hard gate** — unchanged, correct

---

## The Plan Is Closed

Six rounds of review. Cross-referenced against raw JD text. Every constant is empirically grounded or will be measured during precompute. Every weight has a documented rationale. Every edge case (oar=-1, github=-1, unknown titles, empty reasoning, title velocity, temporal signal) has a tested fallback.

**Architecture → Locked**
**Constants → Measured during Phase 1 (not guessed)**
**Build order → precompute → audit 1K → rank full 100K**
**v6 additions → additive only, nothing from v5 removed**
