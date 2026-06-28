# Redrob AI — Intelligent Candidate Ranking System

> **India Runs Data & AI Challenge — Submission**  
> A two-stage, multi-signal candidate ranker for 100K profiles against a Senior AI Engineer JD.  
> Scores are deterministic, fully reproducible, and grounded in population-level EDA.

---

## Table of Contents

1. [Problem Statement & Trap Avoidance](#1-problem-statement--trap-avoidance)
2. [System Architecture — Two-Stage Pipeline](#2-system-architecture--two-stage-pipeline)
3. [Repository Structure](#3-repository-structure)
4. [Quick Start — Reproduce the Submission](#4-quick-start--reproduce-the-submission)
5. [Complete Scoring Formula](#5-complete-scoring-formula)
6. [Component Deep-Dive](#6-component-deep-dive)
   - [6.1 Semantic Score — 20%](#61-semantic-score--20-of-base)
   - [6.2 Production Score — 25%](#62-production-score--25-of-base)
   - [6.3 Ownership Score — 12%](#63-ownership-score--12-of-base)
   - [6.4 Evaluation Score — 12%](#64-evaluation-score--12-of-base)
   - [6.5 Retrieval Score — 8%](#65-retrieval-score--8-of-base)
   - [6.6 Skill Score — 8%](#66-skill-score--8-of-base)
   - [6.7 Career Depth — 7%](#67-career-depth-score--7-of-base)
   - [6.8 Title Score — 5%](#68-title-score--5-of-base)
   - [6.9 Location Score — 3%](#69-location-score--3-of-base)
   - [6.10 Behavioral Score — 25% of final](#610-behavioral-score--25-of-final)
7. [Bonuses Applied After Base](#7-post-base-bonuses)
8. [Penalties & Hard Gates](#8-penalties--hard-gates)
9. [Integrity & Fraud Detection](#9-integrity--fraud-detection)
10. [Evidence-Based Reasoning Strings](#10-evidence-based-reasoning-strings)
11. [Design Decisions — Every Weight Has a Number](#11-design-decisions--every-weight-has-a-number)
12. [Anti-Pattern Handling (JD-Explicit)](#12-anti-pattern-handling-jd-explicit)
13. [Score Distribution & Validation](#13-score-distribution--validation)
14. [Requirements & Environment](#14-requirements--environment)
15. [AI Tools Declaration](#15-ai-tools-declaration)

---

## 1. Problem Statement & Trap Avoidance

Given **100,000 candidate profiles** and a Senior AI Engineer JD (Redrob AI, Series A, Pune/Noida), rank all candidates from most to least suitable and output `candidate_id`, `rank`, `score [0–1]`, and `reasoning`.

The JD contains an explicit warning:

> *"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

**How this system avoids the trap:**

| Trap | How Countered |
|---|---|
| Skills-section keyword stuffing | Semantic score reads career text (65% weight), not skills lists |
| "Marketing Manager" with perfect skill list | Title score is a hard downweight (0.10); semantic sim to JD will be near-zero |
| All-AI-keywords, zero production work | `production_score` requires PROD_T1 evidence of deployed, real-user systems |
| Post-2022-only LLM framework users | Temporal penalty: −0.05 if AI skills present but all roles started ≥ 2022 |
| Behaviorally unavailable perfect profiles | `behavioral_score` at 25% of final; inactive >180d candidates score ≤ 0.10 on availability |

---

## 2. System Architecture — Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — precompute.py  (run once, ~3 hours on CPU)           │
│                                                                  │
│  1. Load all-MiniLM-L6-v2 (22M params, 384-dim output)          │
│  2. Encode expanded JD text → jd_emb.npy  (1 × 384)            │
│  3. Encode 7 skill anchors  → jd_skill_embs.npy  (7 × 384)     │
│  4. Encode 100K summaries   → summary_embs.npy  (100K × 384)   │
│  5. Encode 100K careers     → career_embs.npy   (100K × 384)   │
│  6. Build skill embed map   → skill_embed_map.pkl (~700 skills) │
│  7. Measure population statistics → normalization_constants.json│
│     · semantic_p1 / semantic_p99  (for P1/P99 rescaling)        │
│     · skill_denom  (P95 of population skill totals)             │
│     · production_p99  (if < 0.50, PROD_RESCALE=True activates)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ artifacts/ (committed to repo)
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — rank.py  (run once, ~90 sec on CPU)                  │
│                                                                  │
│  · Loads all artifacts at startup — zero model calls at runtime │
│  · Scores all 100K candidates using 9 base components +         │
│    4 bonuses + 4 penalty types + 1 power transform               │
│  · Integrity gate: CRITICAL or 2×MAJOR → score = 0.0            │
│  · Outputs sorted CSV (top 100) + JSON + MD for inspection       │
└─────────────────────────────────────────────────────────────────┘
```

**Why pre-compute embeddings? (One-time Precomputation vs. Ultra-fast Ranking)**  
* **One-time Precomputation (`precompute.py`)**: Encoding the 100K profiles is a heavy computational lift. On typical consumer CPUs (e.g. Ryzen 5 7520U), summary and career encoding take about 1.5 hours each (totaling ~3 hours). However, this only needs to be run **once** to generate the static files in the `artifacts/` folder.
* **Ultra-fast Candidate Ranking (`rank.py`)**: Once the precomputation is done, the actual ranking script `rank.py` runs **extremely quickly (under 90 seconds)**. It loads the pre-compiled numpy arrays and executes pure vectorized matrix dot-products and keyword searches, bypassing all heavy neural model inference at runtime. This allows quick iterations, tuning of scoring weights, or auditing without waiting.

---

## 3. Repository Structure

```
candidate_ranking_system/
│
├── precompute.py              ← Stage 1: build all artifacts (run once)
├── rank.py                    ← Stage 2: score 100K candidates → CSV
├── audit.py                   ← Validate scores, inspect top-N candidates
│
├── requirements.txt           ← Pinned dependencies
├── README.md                  ← This file
│
├── artifacts/                 ← Generated by precompute.py
│   ├── jd_emb.npy             ← JD embedding (1 × 384)
│   ├── jd_skill_embs.npy      ← 7 skill anchor embeddings (7 × 384)
│   ├── summary_embs.npy       ← 100K candidate headline+summary embeddings
│   ├── career_embs.npy        ← 100K candidate career description embeddings
│   ├── cid_order.npy          ← Candidate ID order (row-index → candidate_id)
│   ├── skill_embed_map.pkl    ← {skill_name → 384-dim embedding} for ~700 skills
│   └── normalization_constants.json  ← Measured P1/P99/P95 constants
│
└── [PUB] India_runs_data_and_ai_challenge/
    └── India_runs_data_and_ai_challenge/
        ├── candidates.jsonl           ← 100K candidates (~487 MB)
        ├── job_description.docx       ← Source JD
        ├── candidate_schema.json      ← Field definitions
        ├── sample_submission.csv      ← Expected output format
        └── validate_submission.py     ← Official validator
```

---

## 4. Quick Start — Reproduce the Submission

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Build artifacts (run once, ~3 hours on CPU)
python precompute.py --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"

# 2. Validate output from precompute
#    Check printed ranges:
#    · semantic P1/P99 spread should be ~0.18–0.46 (MiniLM compresses cosine scores)
#    · production_p99 < 0.50 → PROD_RESCALE activates automatically

# 3. Score all 100K candidates (~90 sec)
python rank.py 
  --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" 
  --out submission.csv

# 4. Validate
python "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py" submission.csv

# 5. Inspect top-50
python audit.py 
  --submission submission.csv 
  --candidates "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" 
  --top 50

# 6. Run interactive Streamlit Sandbox locally
streamlit run app.py
```

**Total CPU runtime (from scratch)**: ~3 hours.  
* **Precomputation stage** (`precompute.py`): ~3 hours (run once to generate arrays).  
* **Ranking stage** (`rank.py` or hosted Streamlit app): **~90 seconds**. The Streamlit app does not run the 3-hour precomputation step on every file upload—it loads the generated static files and executes instantly.

---

## 5. Complete Scoring Formula

This is the exact formula as implemented in [`rank.py`](rank.py), lines 1122–1219.

### Base Score (before behavioral)

```
BASE_SCORE = (
    0.20 × semantic_score      ← P1/P99-rescaled cosine similarity (career-weighted)
  + 0.25 × production_score    ← T1 + T2 keyword evidence, conditionally rescaled
  + 0.12 × ownership_score     ← Search-ownership signal hits / 5.0
  + 0.12 × evaluation_score    ← Eval signal hits / 6.0  (NDCG, MRR, A/B, etc.)
  + 0.08 × retrieval_score     ← Retrieval signal hits / 6.0  (BM25, dense, LTR, etc.)
  + 0.08 × skill_score         ← P95-normalized semantic anchor matching
  + 0.07 × career_depth_score  ← YoE band + product_ratio + stability + industry
  + 0.05 × title_score         ← Soft gradient lookup (full coverage, no hard zeroes)
  + 0.03 × location_score      ← Country-based additive signal
)
```

> **Note:** education_score is removed from the base formula in the current system — it was present in earlier plan versions but was dropped in favour of `product_ratio` and `retrieval_score`.

### Final Score

```
PRE_FINAL = 0.75 × BASE_SCORE + 0.25 × BEHAVIORAL_SCORE

FINAL = power_transform(PRE_FINAL, exponent=0.85)
      + domain_bonus    (up to +0.06)
      + hybrid_bonus    (up to +0.04)
      + ops_bonus       (up to +0.03)
      + boring_bonus    (up to +0.04)
      + marketplace_bonus (+0.03 if applicable)
      − cv_speech_penalty   (×0.95 multiplier if CV/Speech, no IR evidence)
      − services_penalty    (−0.05 if current employer is a pure services firm)
      − research_penalty    (×0.92 if title contains "research" AND production < 0.40)
      − availability_penalty (−0.05 per flag for inactive+closed, poor response rate)
      − overlap_penalty     (−0.02 per overlapping same-level degree)

HARD GATE: Any 1 CRITICAL or 2 MAJOR integrity flags → FINAL = 0.0
```

The **power transform** `x^0.85` is applied to sharpen score separation at the top:
- 1.00 → 1.000  ·  0.95 → 0.957  ·  0.90 → 0.913  ·  0.80 → 0.822

---

## 6. Component Deep-Dive

### 6.1 Semantic Score — 20% of base

**Function:** `compute_semantic_score(candidate_idx)`

```python
raw = 0.35 × cosine(jd_emb, summary_embs[idx])
    + 0.65 × cosine(jd_emb, career_embs[idx])

rescaled = (raw − SEM_P1) / (SEM_P99 − SEM_P1)
score = clip(rescaled, 0.0, 1.0)
```

| Parameter | Value (measured) | Rationale |
|---|---|---|
| Summary weight | 0.35 | Summaries are aspirational; career descriptions contain actual work |
| Career weight | 0.65 | Career text proves what was built, not what is claimed |
| Normalization | P1/P99 percentile clip | MiniLM compresses cosine into ~[0.18, 0.46]; min/max is outlier-sensitive |
| SEM_P1 | 0.1827 (measured) | 1st percentile of population cosine scores |
| SEM_P99 | 0.4629 (measured) | 99th percentile — rescaling treats this as score = 1.0 |

**The JD text is expanded** before encoding to embed the semantic of the 90-day mandate:  
`"Migrated upgraded legacy BM25 keyword search to hybrid semantic retrieval embeddings. Improved recruiter engagement metrics..."`  
This means candidates whose careers actually describe this work will score higher on semantic similarity than candidates who merely list "BM25" as a skill.

---

### 6.2 Production Score — 25% of base

**Function:** `compute_production_score(candidate)` → wraps `compute_production_raw(candidate)`

This is the **highest-weighted single component** because the JD is unambiguous:  
> *"If you've spent your career in pure research environments without any production deployment — we will not move forward."*

**Two-tier keyword system on full profile text (summary + all career descriptions):**

**PROD_T1** — Direct production evidence (29 signals, score = min(hits/5, 1.0)):
```
"production", "deployed", "real users", "at scale", "serving", "latency",
"a/b test", "throughput", "live system", "queries per second", "qps",
"p99", "p95", "sla", "uptime", "index refresh", "embedding drift",
"retrieval quality regression", "model monitoring", "inference pipeline",
"feature store", "offline benchmark", "online experiment", "ndcg", "mrr",
"precision at k", "recall at k", "click-through", "recruiter feedback",
"a/b experiment", "evaluation framework", "ranking quality"
```

**PROD_T2** — Supporting ownership context (18 signals, score = min(hits/8, 0.60)):
```
"shipped", "launched", "built end-to-end", "owned", "led design",
"search", "ranking", "retrieval", "recommendation", "matching",
"pipeline", "infrastructure", "architecture", "bm25", "tf-idf",
"keyword search", "migrated", "hybrid retrieval", "hybrid search",
"re-ranking", "reranking", "recruiter engagement", "click-through rate"
```

**RESEARCH_NEGS** — Anti-patterns (penalty = min(hits × 0.05, 0.20)):
```
"arxiv", "research lab", "academic", "thesis", "proof of concept",
"prototype only", "langchain tutorial", "how i built",
"getting started with", "demo project", "side project only"
```

**Score formula:**
```
base   = 0.65 × T1_score + 0.35 × T2_score
penalty = min(neg_hits × 0.05, 0.20)
gh_bonus = 0.10 if github≥60 else 0.05 if github≥30 else 0.0
temporal_penalty = −0.05 if (has_AI_skills AND all_roles_started ≥ 2022)

raw = clip(base − penalty + gh_bonus − temporal_penalty, 0, 1)
```

If `PROD_P99 < 0.50` (measured at 0.3912): score is rescaled by dividing by P99,  
ensuring the top production candidates reach 1.0 rather than clustering at 0.39.

---

### 6.3 Ownership Score — 12% of base

**Function:** `compute_ownership_score(candidate)`

Measures evidence that the candidate **owned** the system end-to-end, not just contributed to it.

**Signals** (hits / 5.0, clipped to 1.0):
```
"ranking layer", "designed ranker", "owned ranking",
"migration from keyword", "hybrid retrieval architecture", "feedback loop"
```

12% weight reflects the JD's emphasis on IC ownership:  
> *"Own the intelligence layer of Redrob's product."*

---

### 6.4 Evaluation Score — 12% of base

**Function:** `compute_eval_score(candidate)`

The JD states explicitly: *"If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful."*

**Signals** (hits / 6.0, clipped to 1.0):
```
"ndcg", "mrr", "map", "offline benchmark", "online experiment",
"ab test", "a/b test", "evaluation framework", "ranking quality",
"precision at k", "recall at k", "recruiter feedback"
```

At 12% this is equal to ownership — both reflect the JD's repeated emphasis  
on *evaluation rigor* and *end-to-end ownership* as first-class requirements.

---

### 6.5 Retrieval Score — 8% of base

**Function:** `compute_retrieval_score(candidate)`

**Signals** (hits / 6.0, clipped to 1.0):
```
"bm25", "dense retrieval", "hybrid retrieval", "learning-to-rank",
"learning to rank", "reranking", "re-ranking", "faiss", "hnsw",
"ann search", "embedding model", "retrieval quality"
```

Separate from production score because retrieval vocabulary is a strong JD signal  
(Weeks 4–8 mandate: "embeddings, hybrid retrieval, LLM-based re-ranking").

---

### 6.6 Skill Score — 8% of base

**Function:** `compute_skill_score(candidate)`

**Semantic anchor matching** — not keyword counting. Each candidate skill is embedded  
(using the pre-built `skill_embed_map.pkl`) and compared against 7 JD-aligned anchors:

| Anchor | Domain |
|---|---|
| "production embeddings retrieval semantic search vector database" | Core IR/retrieval |
| "NDCG MRR MAP ranking evaluation offline benchmark online A/B experiment" | Evaluation rigor |
| "Python production code software engineering quality" | Code quality |
| "LLM fine-tuning LoRA QLoRA PEFT language model" | LLM stack |
| "NLP information retrieval dense retrieval BM25 hybrid search" | Search/NLP |
| "recommendation systems candidate matching personalization ranking" | Rec-sys domain |
| "distributed systems inference optimization scalability latency" | Scale/Ops |

**Per-skill weighting:**
```python
score_contribution = max_sim × proficiency × (0.65 + 0.35×duration_factor) × endorsement_factor × assessment_factor
```

| Factor | Detail |
|---|---|
| `max_sim` | Best cosine across 7 anchors; skills below 0.35 threshold discarded (catches HTML, Redux, Excel) |
| `proficiency` | beginner=0.50, intermediate=0.75, advanced=1.00, expert=1.10 |
| `duration_factor` | min(months / 36, 1.0) — recency and depth of practice |
| `endorsement_factor` | min(1 + endorsements/40, 1.25) — capped to avoid endorsement-farming |
| `assessment_factor` | 1.0 + (score − 50)/150 if assessed; else 1.0 neutral |

**Denominator = population P95** (measured: 1.1768) — preserves differentiation among the top candidates without clipping their scores.

---

### 6.7 Career Depth Score — 7% of base

**Function:** `compute_career_depth_score(candidate)` + `_compute_title_velocity_penalty()`

**Sub-components (sum to career_depth_score):**

```
career = 0.45 × yoe_score
       + 0.15 × product_ratio_score
       + 0.05 × company_score          (tenure-weighted)
       + 0.25 × stability_score
       + 0.10 × industry_bonus
```

**YoE band mapping:**

| Years | Score | Rationale |
|---|---|---|
| 4–9 yr | 1.00 | JD target band |
| 3 yr | 0.80 | Slightly junior |
| 10 yr | 0.90 | Slightly senior — still excellent |
| 11 yr | 0.80 | Growing seniority mismatch |
| 12+ yr | 0.40 − decreasing | Increasingly over-senior for IC role |
| < 3 yr | 0.40 + (yoe × 0.13) | Too junior |

**Product Ratio** (`compute_product_ratio_score`):  
Fraction of career months at product companies vs services. The threshold uses  
`COMPANY_SCORE ≥ 0.75` for unknown companies (not 0.65 default) — preventing  
unrecognised-but-legitimate companies from earning undeserved product credit.

**Company scores (tenure-weighted across full career):**

| Category | Score range | Examples |
|---|---|---|
| Tier-1 global tech | 0.95–0.98 | Google, Meta, Netflix, Amazon, Uber, LinkedIn |
| Top India product | 0.88–0.95 | Swiggy, Razorpay, CRED, Flipkart, PhonePe, Zepto |
| Mid India product | 0.78–0.88 | Paytm, Zomato, Meesho, Nykaa, Byju's |
| Services/consulting | 0.50–0.55 | TCS, Infosys, Wipro, Accenture, Genpact |
| Unknown companies | 0.65 (neutral) | Neither penalised nor rewarded |

**Stability score:**
- ≥ 36 months avg tenure → 1.0  
- ≥ 24 months → 0.80–1.0 (linear)  
- ≥ 18 months → 0.65–0.80 (linear)  
- < 18 months → max(0.30, avg/18 × 0.65)

**Title velocity penalty** (`_compute_title_velocity_penalty`):  
−0.05 if avg tenure < 20 months AND 3+ distinct title levels observed.  
*Catches the JD's named anti-pattern: "Senior→Staff→Principal by switching companies every 1.5 years."*

**Industry recency bonus** (up to +0.15):  
+0.08 per recent role in a high-relevance industry (AI/ML, Fintech, E-commerce, SaaS, etc.), capped at 0.15.

---

### 6.8 Title Score — 5% of base

**Function:** `compute_title_score(candidate)`  
Full soft gradient — **no hard zeroes**, full dataset coverage.

| Title | Score | Title | Score |
|---|---|---|---|
| Search Engineer | 1.00 | Data Scientist | 0.70 |
| Recommendation Systems Eng | 1.00 | Junior ML Engineer | 0.75 |
| Applied ML Engineer | 0.98 | Full Stack Developer | 0.50 |
| Machine Learning Engineer | 0.97 | Cloud Engineer | 0.48 |
| AI Research Engineer | 0.95 | Data Analyst | 0.38 |
| Senior ML Engineer | 0.95 | Business Analyst | 0.22 |
| NLP Engineer | 0.95 | HR Manager | 0.10 |
| Computer Vision Engineer | 0.80 | Marketing Manager | 0.10 |
| Senior Software Engineer | 0.78 | Content Writer / Sales | 0.08 |
| Unknown title | 0.35 (default) | | |

Low weight (5%) is intentional: the JD explicitly states *"Some people hit senior engineer judgment at 4 years; some never hit it after 15."* Title alone is a weak signal.

---

### 6.9 Location Score — 3% of base

**Function:** `compute_location_score(candidate)` — reads `profile.country` directly.

| Country | Score | Notes |
|---|---|---|
| India | 1.00 | JD primary location |
| UAE | 0.65 | Close timezone, no visa |
| Singapore | 0.60 | Frequent India↔SG movement |
| USA / United States | 0.60 | Case-by-case per JD |
| Australia / Germany | 0.55 | Possible, no visa |
| UK | 0.40 | Possible, no visa sponsorship |
| Canada | 0.35 | More timezone friction |
| Unknown | 0.40 | Neutral default |

Location is additive at 3% (not a gate): a Singapore-based candidate at 0.98 overall  
cannot be zeroed out by geography when the JD explicitly says "case-by-case."

---

### 6.10 Behavioral Score — 25% of final

**Function:** `compute_behavioral_score(candidate)`

```
BEHAVIORAL = 0.30 × availability
           + 0.25 × engagement
           + 0.20 × momentum
           + 0.15 × github
           + 0.10 × notice
```

Acts as an **availability-weighted gate modifier** — a perfect base-scorer who logged in 9 months ago and never responds to recruiters will be materially outranked by a slightly weaker profile that is actively engaged.

**Availability** (30% of behavioral):
```
≤ 14 days inactive → 1.00      (recency weight 60%)
≤ 30 days         → 0.90
≤ 60 days         → 0.75
≤ 90 days         → 0.55
≤ 180 days        → 0.30
> 180 days        → 0.10

open_to_work = True  → 1.0  (weight 40%)
open_to_work = False → 0.50
```

**Engagement** (25% of behavioral):
```
0.50 × recruiter_response_rate
+ 0.30 × interview_completion_rate
+ 0.20 × offer_acceptance_rate   (oar = −1 → 0.50 neutral, not 0.0)
```
> `oar = −1` maps to 0.50 (neutral) because 60% of candidates have never received an offer.  
> Treating them as 0.0 systematically penalises the majority for missing data, not poor behavior.

**Momentum** (20% of behavioral):  
30-day platform activity: saves (÷8), apps submitted (÷5), profile views (÷20), search appearances (÷100).

**GitHub** (15% of behavioral):  
`github_activity_score / 80`; if score = −1 → floor at 0.15.  
Senior engineers with no external validation get a mild negative (per JD: *"5+ years closed-source without papers/talks/OSS"*).

**Notice Period** (10% of behavioral):
```
≤ 30 days  → 1.00   (JD: "we'd love sub-30-day notice; can buy out up to 30 days")
≤ 60 days  → 0.65
≤ 90 days  → 0.35
≤ 120 days → 0.05
> 120 days → 0.02
```

---

## 7. Post-Base Bonuses

Applied after the 75/25 base/behavioral blend, before the power transform:

| Bonus | Max Value | Trigger signals |
|---|---|---|
| `domain_bonus` | +0.06 | Recruiter/candidate-search domain overlap: "candidate corpus", "recruiter engagement", "time-to-shortlist", "recruiter-facing" (hits/4 × 0.06) |
| `hybrid_bonus` | +0.04 | Hybrid search stack evidence: "hybrid retrieval", "bm25", "dense retrieval", "reranking", "llm-based re-ranker" (hits/4 × 0.04) |
| `boring_bonus` | +0.04 | Boring-but-critical infra signals: "rollback", "dashboard", "versioning", "latency budget", "monitoring", "index refresh", "embedding drift" (hits/4 × 0.04) |
| `ops_bonus` | +0.03 | Ops/reliability signals: "embedding drift", "index refresh", "rollback", "versioning", "retrieval quality regression" (hits/4 × 0.03) |
| `marketplace_bonus` | +0.03 | Any of: "recommendation system", "search ranking", "candidate matching", "retrieval system", "marketplace", "relevance ranking" |

Maximum total bonus: **+0.20** (rarely achieved; typical top-10 candidates accumulate 0.10–0.14).

---

## 8. Penalties & Hard Gates

Applied sequentially after bonuses, before the power transform:

| Penalty | Amount | Condition |
|---|---|---|
| CV/Speech domain mismatch | ×0.95 multiplier | Profile mentions CV/speech/robotics keywords AND retrieval evidence is weak (retrieval<0.20, eval<0.20, production<0.60) |
| Services employer penalty | −0.05 | Current employer (career_history[0]) is in SERVICE_COMPANIES |
| Research title penalty | ×0.92 multiplier | Current title contains "research" AND production_score < 0.40 |
| Inactive + closed penalty | −0.05 | Last active > 120 days AND open_to_work = False |
| Low response rate penalty | −0.05 | recruiter_response_rate < 0.15 |
| Overlapping degree penalty | −0.02/flag | Same-level degrees with ≥ 2yr date overlap (minor flag) |

**Hard Gate (score = 0.0):**  
Any **1 CRITICAL** or **≥ 2 MAJOR** integrity flags trigger a hard zero.  
Zero-score candidates are excluded from the top-100 output.

---

## 9. Integrity & Fraud Detection

**Function:** `evaluate_integrity_rules(candidate)` — 7 rule categories:

| Rule | Severity | What it catches |
|---|---|---|
| Experience mismatch | CRITICAL | Stated YoE vs documented career history differs by > 3 years (either direction) |
| Technology anachronism | CRITICAL | RAG/LLM/vector-DB terminology in a role that ended before 2021 |
| Company founding date | CRITICAL | Start date predates company founding (e.g. "worked at Sarvam AI starting 2020" — founded 2023) |
| Education-to-career gap | CRITICAL (8+yr) / MAJOR (5–7yr) | Graduated then no documented job for 5+ years |
| Impossible chronology | CRITICAL | Masters degree completed before next undergraduate degree started |
| Duplicated narratives | MAJOR | Two role descriptions share >95% trigram similarity |
| Multiple employment overlaps | MAJOR | 2+ simultaneous full-time roles with overlapping dates |
| Overlapping same-level degrees | MINOR | Two Bachelor's or Master's degrees with overlapping years |
| Skill duration inflation | MINOR / MAJOR | Total skill-months exceed (career span + 48-month college buffer) × 15 |

**Measured outcome on 100K candidates:**
- Hard-filtered (score = 0.0): ~3–4% of pool  
- Minor flags only (soft penalty applied): ~8–10% of pool  
- Clean profiles: ~87–89%

---

## 10. Evidence-Based Reasoning Strings

**Function:** `generate_reasoning(candidate, scores)`

The reasoning column is **not a keyword list** — it is a synthesised, evidence-based narrative  
following the template:

```
Role (Xyr) | Core production claim | Retrieval ownership | Evaluation ownership |
Deployment shipping evidence | Search relevance specifics |
Semantic match note | Availability | Notice | OTW
```

**Example (Rank #1, CAND_0086022):**
```
Senior Applied Scientist (5.3yr) | who built production recruiter-facing search and
ranking systems | Led hybrid retrieval combining sparse (BM25) and dense embeddings |
owned learning-to-rank | and offline→online evaluation (NDCG/MRR/A/B testing) |
Shipped production relevance improvements | with recruiter-search relevance ownership |
Strong semantic match to Redrob's intelligence-layer ownership | Notice: 0d ✓ | OTW: open
```

**Key design choices in the reasoning generator:**

| Old pattern (removed) | New pattern |
|---|---|
| `Key skills: Recommendation Systems, pgvector, LangChain` | *(removed entirely)* |
| `JD keywords: Hybrid retrieval, BM25, Dense retrieval` | *(removed entirely)* |
| `Production recommendation systems` | `who built production recruiter-facing search and ranking systems` |
| `Designed evaluation framework` | `and offline→online evaluation (NDCG/MRR/A/B testing)` |

Every claim in the reasoning is grounded in a detected signal from the candidate's text —  
not extracted from the skills list, not a template variable, not a keyword dump.

---

## 11. Design Decisions — Every Weight Has a Number

All weights derived from EDA on the full 100K dataset:

```
Dataset size:                        100,000  candidates
AI-relevant (≥4 AI skills OR AI title):  20,041  ← real competition pool
In JD target YoE band (4–9 years):       10,912
Behaviorally available:                   2,017  ← open_to_work + active < 90d
India-based + available:                  1,575
Good GitHub score (≥30):                    653  ← true top-tier pool

Title distribution (non-AI dominate):
  Business Analyst:       5,833       HR Manager:    5,830
  Mechanical Engineer:    5,791       Accountant:    5,764
  Software Engineer:      3,450       ML Engineer:   ~800
  Search/Rec-sys Eng:      ~400
```

**Why production_score is 25% (highest single weight):**  
The JD has an explicit disqualifier for pure research. This signal is also the one most  
gamed by keyword stuffing in a naive system — so it deserves a strong weight *with*  
the T1/T2/negative architecture that distinguishes real deployment from tutorial blogs.

**Why behavioral is 25% of final (not base):**  
The real contest for top-10 is among ~650 candidates (good GitHub, available, India).  
Behavioral signals narrow 100K → 2K before base scoring differentiates. At 25%,  
a candidate inactive >180d (availability ≈ 0.10) is materially outranked even if their  
technical profile would otherwise put them top-5.

**Why semantic is 20% of base:**  
It is the only signal that catches a strong candidate who doesn't use the "right keywords"  
but whose career text is substantively about retrieval/ranking. The JD's Tier-5 example  
(recommendation system at product company, no buzzwords) is precisely captured here.

**Why oar=−1 maps to 0.50 neutral:**  
60% of 100K candidates have never received an offer (oar=−1). Mapping to 0.0 would  
systematically penalise the majority for missing data rather than poor behaviour.

**Why the P1/P99 rescaling is essential:**  
Measured population statistics:
```
semantic P1  = 0.1827    (worst 1% cosine similarity to JD)
semantic P99 = 0.4629    (best 1% cosine similarity to JD)
raw spread   = 0.2802    (without rescaling, all scores cluster in this 0.28-wide band)
```
Without rescaling, semantic similarity differences of 0.01 raw cosine would represent  
3.6% of total range — meaningless differentiation. After rescaling, the full [0,1] range  
is used, and semantic becomes a genuine ranking signal.

**Why production P99 rescaling activates (PROD_RESCALE=True):**  
```
production P99 = 0.3912    (best 1% of raw production scores)
```
Without rescaling, even the strongest production candidates max out at 0.39.  
Dividing by P99 stretches the top candidates to 1.0 while preserving relative ordering.

---

## 12. Anti-Pattern Handling (JD-Explicit)

| JD Anti-Pattern | Detection Method | Scoring Effect |
|---|---|---|
| **Keyword stuffers** — all AI skills, wrong background | Semantic score reads career text (not skills list); title_score applies soft downweight | Low semantic + low production = naturally ranked low |
| **Title-chasers** — Senior→Staff→Principal in 1.5yr cycles | `_compute_title_velocity_penalty()`: avg tenure < 20mo AND 3+ distinct title levels → −0.05 | Applied to career_depth_score |
| **Framework enthusiasts** — LangChain tutorials, demo projects | RESEARCH_NEGS: "langchain tutorial", "demo project", "how i built" → −0.05 per hit, capped −0.20 | Subtracted from production_raw |
| **Pure services career** — TCS/Infosys entire career | `product_ratio_score` (15% of career_depth) + services penalty: −0.05 if current employer is services firm | Dual penalty: lower career + lower final |
| **Post-2022-only ML** — GPT-era framework users | `_has_pre_llm_ai_experience()`: if AI skills present but all roles started ≥ 2022 → temporal_penalty = −0.05 | Subtracted from production_raw |
| **Behaviorally unavailable** — Logged in 8 months ago | availability_score down to 0.10; additional −0.10 penalty if >120d inactive + not OTW | 25% behavioral weight = material ranking impact |
| **Pure researchers** — No production deployment | RESEARCH_NEGS + research_title_penalty (×0.92 if title="research" AND production<0.40) | Double penalty path |
| **CV/Speech engineers** — No NLP/IR exposure | ×0.95 multiplier if CV keywords present AND retrieval<0.20, eval<0.20, production<0.60 | Mild penalty preserves valid cross-domain candidates |

---

## 13. Score Distribution & Validation

**Observed score distribution (current submission):**

| Position | Score |
|---|---|
| #1 | 1.0000 |
| #10 | 0.8300 |
| #50 | 0.7739 |
| #100 | 0.7094 |
| Top-10 gap (#1 − #10) | 0.1700 ✅ (target: > 0.03) |

**Validation checklist:**

```
PRECOMPUTE:
[x] precompute.py ran without errors
[x] Semantic P1/P99 printed and in expected range (0.18, 0.46)
[x] PROD_RESCALE=True activated (P99 = 0.3912 < 0.50)
[x] All 7 artifacts present in artifacts/

SCORING:
[x] All component scores verified in [0, 1] via audit.py
[x] Score distribution has meaningful spread (top-10 gap = 0.17)
[x] ~3–4% hard-filtered (only genuine integrity violations)

SUBMISSION:
[x] validate_submission.py → "Submission is valid."
[x] Top-10 manually reviewed — all are AI/ML engineers, India-based or willing to relocate
[x] Top-10 are all open-to-work or recently active
[x] Zero honeypot candidates in top-50 (audit.py inspection)
[x] Reasoning strings are evidence-based, not keyword lists
```

---

## 14. Requirements & Environment

```
sentence-transformers==2.7.0   # SBERT models + cosine similarity utilities
numpy==1.26.4                  # PINNED — numpy 2.x breaks sklearn/scipy (ABI conflict)
scikit-learn==1.4.2            # Available for audit scripts
tqdm==4.66.4                   # Progress bars during precompute
```

**Python:** 3.10+

**Hardware:** CPU is sufficient.

| Step | Time (Ryzen 5 7520U class CPU) |
|---|---|
| Model download (first run) | ~2 min |
| Encoding 100K summaries (batch_size=256) | ~1.5 hr |
| Encoding 100K career texts (batch_size=256) | ~1.5 hr |
| Skill totals computation | ~1 min |
| `rank.py` full run (100K candidates) | ~90 sec |
| **Total (first run)** | **≈ 3 hr** |

**Memory peak:** ~3–4 GB  
- 100K × 384 × 2 embedding matrices ≈ 300 MB each  
- skill_embed_map.pkl ≈ 50 MB

> **Why truncate to 1000 chars for career text?**  
> `all-MiniLM-L6-v2` has a hard 256-token limit (~200 words). Career descriptions  
> concatenated can exceed 5000 chars, causing the tokenizer to silently drop text.  
> Capping at 1000 chars (~200 tokens) gives the model everything it can actually use.

---

