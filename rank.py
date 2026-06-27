#!/usr/bin/env python3
"""
newrank.py  —  Redrob AI Candidate Ranker  (v7 Optimized)
Scores all 100K candidates using pre-computed artifacts with custom recruiters-guided signals.
Usage:
    python newrank.py --candidates candidates.jsonl --out submission.csv
"""
import argparse, csv, json, pickle, sys
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

# ── Artifact Paths ─────────────────────────────────────────────────────────────
ARTIFACTS = Path("artifacts")

# ── Load pre-computed artifacts (at module load, not per-candidate) ────────────
print("Loading pre-computed artifacts ...")
jd_emb        = np.load(ARTIFACTS / "jd_emb.npy")
jd_skill_embs = np.load(ARTIFACTS / "jd_skill_embs.npy")
summary_embs  = np.load(ARTIFACTS / "summary_embs.npy")
career_embs   = np.load(ARTIFACTS / "career_embs.npy")
cid_order     = np.load(ARTIFACTS / "cid_order.npy", allow_pickle=True).tolist()

with open(ARTIFACTS / "skill_embed_map.pkl", "rb") as f:
    skill_embed_map = pickle.load(f)

with open(ARTIFACTS / "normalization_constants.json") as f:
    CONSTS = json.load(f)

# Constants: NEVER recomputed — always read from file (measured by precompute.py)
SEM_P1       = CONSTS["semantic_p1"]
SEM_P99      = CONSTS["semantic_p99"]
SKILL_DENOM  = CONSTS["skill_denom"]
PROD_P99     = CONSTS["production_p99"]
PROD_RESCALE = CONSTS["production_rescale"]
print(f"  semantic P1={SEM_P1:.4f}  P99={SEM_P99:.4f}")
print(f"  skill_denom={SKILL_DENOM:.4f}")
print(f"  production_p99={PROD_P99:.4f}  rescale={PROD_RESCALE}")
print(f"  Artifacts loaded. cid_order has {len(cid_order):,} entries.")

# Build O(1) lookup: candidate_id → index in embedding arrays
CID_TO_IDX = {cid: i for i, cid in enumerate(cid_order)}

# ── Reference date for behavioral signals ──────────────────────────────────────
REFERENCE_DATE = date(2026, 6, 17)

# ── Keyword Lists ──────────────────────────────────────────────────────────────
PROD_T1 = [
    "production", "deployed", "real users", "at scale", "serving",
    "latency", "a/b test", "throughput", "millions of", "live system",
    "queries per second", "qps", "p99", "p95", "sla", "uptime",
    "index refresh", "embedding drift", "retrieval quality regression",
    "model monitoring", "inference pipeline", "feature store",
    "offline benchmark", "online experiment", "ndcg", "mrr",
    "precision at k", "recall at k", "click-through", "recruiter feedback",
    "a/b experiment", "evaluation framework", "ranking quality",
]
PROD_T2 = [
    "shipped", "launched", "built end-to-end", "owned", "led design",
    "search", "ranking", "retrieval", "recommendation", "matching",
    "pipeline", "infrastructure", "architecture",
    "bm25", "tf-idf", "keyword search", "migrated", "migrated from",
    "replaced", "improved precision", "improved recall", "reduced latency",
    "recruiter engagement", "click-through rate", "conversion rate",
    "hybrid retrieval", "hybrid search", "re-ranking", "reranking",
]
RESEARCH_NEGS = [
    "arxiv", "research lab", "academic", "thesis",
    "proof of concept", "prototype only", "exploration paper",
    "langchain tutorial", "how i built", "getting started with",
    "demo project", "side project only",
]

PROFICIENCY_WEIGHT = {
    "beginner": 0.50, "intermediate": 0.75, "advanced": 1.00, "expert": 1.10,
}

# ── Evaluation and Retrieval Signals ───────────────────────────────────────────
EVAL_SIGNALS = [
    "ndcg", "mrr", "map", "offline benchmark", "online experiment",
    "ab test", "a/b test", "evaluation framework", "ranking quality",
    "precision at k", "recall at k", "recruiter feedback"
]

RETRIEVAL_SIGNALS = [
    "bm25",
    "dense retrieval",
    "hybrid retrieval",
    "learning-to-rank",
    "learning to rank",
    "reranking",
    "re-ranking",
    "faiss",
    "hnsw",
    "ann search",
    "embedding model",
    "retrieval quality"
]

# ── Title score dictionary — full coverage ─────────────────────────────────────
TITLE_SCORES = {
    "Search Engineer":                1.00,
    "Recommendation Systems Engineer": 1.00,
    "Applied ML Engineer":             0.98,
    "Machine Learning Engineer":       0.97,
    "AI Research Engineer":            0.95,
    "Senior Machine Learning Engineer": 0.95,
    "Senior Software Engineer(ML)":    0.95,
    "NLP Engineer":                    0.95,
    "AI Specialist":                   0.90,
    "Computer Vision Engineer":        0.80,
    "Senior Software Engineer":        0.78,
    "Analytics Engineer":              0.75,
    "Backend Engineer":                0.72,
    "Software Engineer":               0.68,
    "Data Scientist":                  0.70,
    "Data Engineer":                   0.60,
    "Senior Data Engineer":            0.62,
    "Full Stack Developer":            0.50,
    "Cloud Engineer":                  0.48,
    "DevOps Engineer":                 0.45,
    "Java Developer":                  0.40,
    ".NET Developer":                  0.38,
    "Mobile Developer":                0.38,
    "Frontend Engineer":               0.35,
    "QA Engineer":                     0.32,
    "Data Analyst":                    0.38,
    "Project Manager":                 0.25,
    "Business Analyst":                0.22,
    "Operations Manager":              0.20,
    "HR Manager":                      0.10,
    "Marketing Manager":               0.10,
    "Content Writer":                  0.08,
    "Sales Executive":                 0.08,
    "Customer Support":                0.08,
    "Accountant":                      0.08,
    "Graphic Designer":                0.08,
    "Civil Engineer":                  0.10,
    "Mechanical Engineer":             0.10,
    "Junior ML Engineer":              0.75,
}
TITLE_DEFAULT = 0.35

# ── Education lookup ───────────────────────────────────────────────────────────
EDU_TIER = {
    "IIT": 1.0, "IIM": 0.95, "BITS": 0.90, "NIT": 0.85,
    "IISc": 1.0, "IIIT": 0.80,
}
EDU_FIELD_BOOST = {
    "computer science": 0.20, "information technology": 0.15,
    "machine learning": 0.25, "artificial intelligence": 0.25,
    "data science": 0.20, "statistics": 0.15, "mathematics": 0.10,
    "electrical engineering": 0.10, "electronics": 0.08,
}

# ── Company score dictionary ───────────────────────────────────────────────────
COMPANY_SCORES = {
    # Synthetic product companies (high quality)
    "Pied Piper": 1.0, "Hooli": 1.0,
    "Wayne Enterprises": 0.85, "Stark Industries": 0.85,
    "Acme Corp": 0.75, "Globex Inc": 0.75,
    "Initech": 0.70, "Dunder Mifflin": 0.65,
    # Real product companies (India)
    "Swiggy": 0.95, "Razorpay": 0.95, "CRED": 0.95,
    "Zomato": 0.90, "Paytm": 0.88, "PhonePe": 0.90,
    "Flipkart": 0.90, "Meesho": 0.88, "Ola": 0.85,
    "Byju's": 0.80, "Unacademy": 0.78, "Nykaa": 0.80,
    "Dunzo": 0.78, "Slice": 0.80, "Zepto": 0.82,
    # Premium Global / Product tech companies
    "Google": 0.98, "Meta": 0.98,
    "Amazon": 0.95, "Microsoft": 0.95, "Uber": 0.95, "LinkedIn": 0.95, "Airbnb": 0.95,
    "NVIDIA": 0.95, "Databricks": 0.95, "Snowflake": 0.95, "Stripe": 0.95,
    "Atlassian": 0.95, "Salesforce": 0.95, "Adobe": 0.95, "Netflix": 0.95,
    # Consulting / IT services
    "TCS": 0.50, "Infosys": 0.50, "Wipro": 0.50,
    "Accenture": 0.50, "Cognizant": 0.50, "Capgemini": 0.50,
    "HCL": 0.50, "Tech Mahindra": 0.52, "Mphasis": 0.52,
    # Fix 3: Genpact was in SERVICE_COMPANIES but missing here → inconsistent treatment
    "Genpact": 0.55, "Genpact AI": 0.55,
}
COMPANY_DEFAULT = 0.65

# ── Product vs Service company classification ──────────────────────────────────
SERVICE_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mphasis", "Genpact", "Genpact AI",
    "IBM", "Hexaware", "Mindtree", "L&T Infotech", "Persistent Systems",
}

# Fix 4: Lowered threshold 0.75 → 0.70 so unknown-but-legitimate companies
# (which fall through to COMPANY_DEFAULT=0.65) aren't silently excluded from
# both numerator and denominator, giving them an undeserved 0.5 product_ratio.
PRODUCT_COMPANIES = {
    c for c in COMPANY_SCORES if c not in SERVICE_COMPANIES and COMPANY_SCORES[c] >= 0.70
}

# ── Domain overlap signals (recruiter-facing search/ranking) ───────────────────
DOMAIN_SIGNALS = [
    "candidate corpus",
    "candidate search",
    "candidate-jd matching",
    "recruiter engagement",
    "time-to-shortlist",
    "recruiter-facing"
]

# ── Hybrid search signals ──────────────────────────────────────────────────────
HYBRID_SIGNALS = [
    "hybrid retrieval",
    "bm25",
    "dense retrieval",
    "reranking",
    "llm-based re-ranker"
]

# ── Ops/Infrastructure signals ─────────────────────────────────────────────────
OPS_SIGNALS = [
    "embedding drift",
    "index refresh",
    "rollback",
    "versioning",
    "retrieval quality regression"
]

# ── Search/Ranking Ownership signals ───────────────────────────────────────────
OWNERSHIP_SEARCH = [
    "ranking layer",
    "designed ranker",
    "owned ranking",
    "migration from keyword",
    "hybrid retrieval architecture",
    "feedback loop"
]

# ── Boring Infrastructure signals ──────────────────────────────────────────────
BORING_SIGNALS = [
    "rollback",
    "dashboard",
    "versioning",
    "latency budget",
    "monitoring",
    "index refresh",
    "embedding drift"
]

# ── Industry recency bonus ─────────────────────────────────────────────────────
HIGH_RELEVANCE_INDUSTRIES = {
    "AI/ML", "Conversational AI", "HealthTech AI", "AI Services",
    "AdTech", "Fintech", "Food Delivery", "E-commerce",
    "EdTech", "SaaS", "Insurance Tech", "HealthTech",
}

# ── Location scores ────────────────────────────────────────────────────────────
LOCATION_SCORES = {
    "india": 1.0,
    "usa": 0.60, "united states": 0.60,
    "canada": 0.35,  # was 0.55
    "uk": 0.40,      # was 0.55
    "australia": 0.55,
    "germany": 0.55,
    "singapore": 0.60,
    "uae": 0.65,
}

# ── Title velocity penalty map ─────────────────────────────────────────────────
TITLE_LEVEL_MAP = {
    "intern": 0, "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3, "lead": 3,
    "staff": 4, "principal": 4, "architect": 4,
    "director": 5, "vp": 6, "head": 5,
}


def _compute_title_velocity_penalty(candidate: dict) -> float:
    """Returns −0.05 if candidate shows title-chaser pattern, else 0.0.
    Pattern: avg tenure < 20 months AND 3+ distinct title levels seen.
    """
    history = candidate.get("career_history", [])
    if len(history) < 3:
        return 0.0

    tenures = [r.get("duration_months", 0) for r in history if r.get("duration_months", 0) > 0]
    if not tenures:
        return 0.0

    avg_tenure = sum(tenures) / len(tenures)
    if avg_tenure >= 20:
        return 0.0

    levels_seen = set()
    for r in history:
        title_lower = r.get("title", "").lower()
        for level_kw, level_num in TITLE_LEVEL_MAP.items():
            if level_kw in title_lower:
                levels_seen.add(level_num)
                break

    return 0.05 if len(levels_seen) >= 3 else 0.0


def _has_pre_llm_ai_experience(candidate: dict) -> bool:
    """True if candidate has AI/ML/search role that started before 2022."""
    ai_keywords = {
        "machine learning", "ml", "nlp", "search", "ranking", "retrieval",
        "recommendation", "embedding", "ai engineer", "data scientist",
    }
    for role in candidate.get("career_history", []):
        start_str = str(role.get("start_date", ""))[:4]
        try:
            start_year = int(start_str)
        except ValueError:
            continue
        combined = role.get("description", "").lower() + " " + role.get("title", "").lower()
        if start_year < 2022 and any(kw in combined for kw in ai_keywords):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_semantic_score(candidate_idx: int) -> float:
    """P1/P99 rescaled cosine similarity — 35% summary, 65% career."""
    raw = (0.35 * float(np.dot(jd_emb, summary_embs[candidate_idx])) +
           0.65 * float(np.dot(jd_emb, career_embs[candidate_idx])))
    rescaled = (raw - SEM_P1) / max(SEM_P99 - SEM_P1, 1e-6)
    return float(np.clip(rescaled, 0.0, 1.0))


def compute_skill_score(candidate: dict) -> float:
    """Semantic anchor skill matching, P95-normalized."""
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
    return float(np.clip(total / max(SKILL_DENOM, 1e-6), 0.0, 1.0))


def compute_production_raw(candidate: dict) -> float:
    """Keyword-based production evidence score with v6 temporal penalty."""
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

    # v6: temporal penalty
    has_ai_skills = any(
        kw in full for kw in ["embedding", "retrieval", "ranking", "nlp", "machine learning"]
    )
    temporal_penalty = 0.0
    if has_ai_skills and not _has_pre_llm_ai_experience(candidate):
        temporal_penalty = 0.05

    return max(0.0, min(base - res + gh_c - temporal_penalty, 1.0))


def compute_production_score(candidate: dict) -> float:
    """Conditionally rescaled production score."""
    raw = compute_production_raw(candidate)
    if PROD_RESCALE:
        return float(np.clip(raw / max(PROD_P99, 1e-6), 0.0, 1.0))
    return raw


def compute_product_ratio_score(candidate: dict) -> float:
    """Fraction of career months spent at product companies vs service companies."""
    history = candidate.get("career_history", [])
    product_months = 0
    service_months = 0
    for role in history:
        months = role.get("duration_months", 0)
        company = role.get("company", "")
        if company in SERVICE_COMPANIES:
            service_months += months
        elif company in PRODUCT_COMPANIES or COMPANY_SCORES.get(company, 0) >= 0.75:
            # Two-tier intentional:
            #   PRODUCT_COMPANIES uses threshold >= 0.70 (named, curated companies)
            #   Inline fallback uses >= 0.75 (unknown companies need higher confidence
            #   to earn product credit; COMPANY_DEFAULT=0.65 deliberately excluded)
            product_months += months
    total = product_months + service_months
    if total == 0:
        return 0.5
    return product_months / total


def compute_career_depth_score(candidate: dict) -> float:
    """YoE band + tenure-weighted company score + stability + industry recency bonus."""
    profile  = candidate["profile"]
    history  = candidate["career_history"]
    yoe      = profile["years_of_experience"]

    if 4 <= yoe <= 9:
        yoe_score = 1.0
    elif yoe == 3:
        yoe_score = 0.80
    elif yoe == 10:
        yoe_score = 0.90
    elif yoe == 11:
        yoe_score = 0.80
    elif yoe < 3:
        yoe_score = max(0.0, 0.40 + yoe * 0.13)
    else:
        yoe_score = max(0.40, 0.80 - (yoe - 11) * 0.04)

    total_months = sum(r["duration_months"] for r in history if r["duration_months"] > 0)
    if total_months == 0:
        company_score = COMPANY_DEFAULT
    else:
        weighted_sum = sum(
            COMPANY_SCORES.get(r["company"], COMPANY_DEFAULT) * r["duration_months"]
            for r in history if r["duration_months"] > 0
        )
        company_score = weighted_sum / total_months

    valid_tenures = [r["duration_months"] for r in history if r["duration_months"] > 0]
    if valid_tenures:
        avg_tenure = sum(valid_tenures) / len(valid_tenures)
        if avg_tenure >= 36:
            stability = 1.0
        elif avg_tenure >= 24:
            stability = 0.80 + (avg_tenure - 24) / 12 * 0.20
        elif avg_tenure >= 18:
            stability = 0.65 + (avg_tenure - 18) / 6 * 0.15
        else:
            stability = max(0.30, avg_tenure / 18 * 0.65)
    else:
        stability = 0.50

    recent_roles = sorted(history, key=lambda r: r.get("start_date", ""), reverse=True)[:2]
    industry_bonus = 0.0
    for r in recent_roles:
        if r.get("industry", "") in HIGH_RELEVANCE_INDUSTRIES:
            industry_bonus = min(industry_bonus + 0.08, 0.15)

    product_ratio = compute_product_ratio_score(candidate)

    # 1. Reduce Company Influence & Replace with Product Ratio
    # yoe: 45%, product_ratio: 15%, company: 5%, stability: 25%, industry: 10%
    raw = (0.45 * yoe_score + 0.15 * product_ratio + 0.05 * company_score + 0.25 * stability + 0.10 * industry_bonus)
    return float(np.clip(raw, 0.0, 1.0))


def compute_title_score(candidate: dict) -> float:
    """Soft gradient title score — full dataset coverage, no hard zeroes."""
    title = candidate["profile"]["current_title"]
    return TITLE_SCORES.get(title, TITLE_DEFAULT)


def compute_education_score(candidate: dict) -> float:
    """Tier + field relevance, capped at 1.0."""
    edu = candidate.get("education", [])
    if not edu:
        return 0.30

    best = 0.0
    for entry in edu:
        institution = entry.get("institution", "")
        field       = entry.get("field_of_study", "").lower()
        degree_type = entry.get("degree", "").lower()

        tier = 0.50
        for kw, val in EDU_TIER.items():
            if kw.lower() in institution.lower():
                tier = val
                break

        field_boost = 0.0
        for kw, boost in EDU_FIELD_BOOST.items():
            if kw in field:
                field_boost = max(field_boost, boost)

        degree_bonus = 0.05 if "phd" in degree_type or "doctorate" in degree_type else 0.0

        score = min(tier + field_boost + degree_bonus, 1.0)
        best  = max(best, score)

    return float(np.clip(best, 0.0, 1.0))


def compute_location_score(candidate: dict) -> float:
    """Additive location component using profile.country directly."""
    country = candidate["profile"].get("country", "").lower()
    return LOCATION_SCORES.get(country, 0.40)


def compute_ownership_score(candidate: dict) -> float:
    """Computes ownership score scaled by hits / 5.0 using search-focused ownership signals."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    
    hits = sum(1 for kw in OWNERSHIP_SEARCH if kw in full)
    return min(hits / 5.0, 1.0)


def compute_eval_score(candidate: dict) -> float:
    """Computes evaluation score scaled by hits / 6.0."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    
    hits = sum(1 for kw in EVAL_SIGNALS if kw in full)
    return min(hits / 6.0, 1.0)


def compute_retrieval_score(candidate: dict) -> float:
    """Computes retrieval score scaled by hits / 6.0."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    
    hits = sum(1 for kw in RETRIEVAL_SIGNALS if kw in full)
    return min(hits / 6.0, 1.0)


# ── Behavioral Score ───────────────────────────────────────────────────────────

def compute_availability_score(sig: dict) -> float:
    """Recency of activity + open-to-work flag."""
    last_active = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_ago    = (REFERENCE_DATE - last_active).days

    if days_ago <= 14:
        recency = 1.00
    elif days_ago <= 30:
        recency = 0.90
    elif days_ago <= 60:
        recency = 0.75
    elif days_ago <= 90:
        recency = 0.55
    elif days_ago <= 180:
        recency = 0.30
    else:
        recency = 0.10

    otw = 1.0 if sig["open_to_work_flag"] else 0.50
    return float(0.60 * recency + 0.40 * otw)


def compute_engagement_score(sig: dict) -> float:
    """Recruiter response rate + interview completion + offer acceptance."""
    rr  = sig["recruiter_response_rate"]
    icr = sig["interview_completion_rate"]
    raw_oar = sig["offer_acceptance_rate"]
    oar = 0.50 if raw_oar < 0 else raw_oar

    return float(np.clip(0.50 * rr + 0.30 * icr + 0.20 * oar, 0.0, 1.0))


def compute_momentum_score(sig: dict) -> float:
    """30-day platform activity."""
    saves  = min(sig["saved_by_recruiters_30d"] / 8.0, 1.0)
    apps   = min(sig["applications_submitted_30d"] / 5.0, 1.0)
    views  = min(sig["profile_views_received_30d"] / 20.0, 1.0)
    search = min(sig["search_appearance_30d"] / 100.0, 1.0)
    return float(0.30 * saves + 0.25 * apps + 0.25 * views + 0.20 * search)


def compute_github_score(sig: dict) -> float:
    """GitHub activity — -1 maps to 0.15 floor."""
    gh = sig["github_activity_score"]
    if gh < 0:
        return 0.15
    return float(np.clip(gh / 80.0, 0.0, 1.0))


def compute_notice_score(sig: dict) -> float:
    """Notice period alignment."""
    days = sig["notice_period_days"]
    if days <= 30:
        return 1.00
    elif days <= 60:
        return 0.65
    elif days <= 90:
        return 0.35
    elif days <= 120:
        return 0.05
    else:
        return 0.02


def compute_behavioral_score(candidate: dict) -> float:
    """Weighted behavioral score."""
    sig         = candidate["redrob_signals"]
    availability = compute_availability_score(sig)
    engagement   = compute_engagement_score(sig)
    momentum     = compute_momentum_score(sig)
    github       = compute_github_score(sig)
    notice       = compute_notice_score(sig)

    return float(np.clip(
        0.30 * availability +
        0.25 * engagement   +
        0.20 * momentum     +
        0.15 * github       +
        0.10 * notice,
        0.0, 1.0
    ))


# ── Extra Bonuses ──────────────────────────────────────────────────────────────

def compute_domain_bonus(candidate: dict) -> float:
    """Bonus for recruiter/candidate-search domain overlap."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in DOMAIN_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.06


def compute_hybrid_bonus(candidate: dict) -> float:
    """Bonus for hybrid search experience."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in HYBRID_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.04


def compute_ops_bonus(candidate: dict) -> float:
    """Bonus for ops/infrastructure experience."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in OPS_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.03


def compute_boring_bonus(candidate: dict) -> float:
    """Bonus for boring infrastructure experience."""
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in BORING_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.04


# ── Consistency gate ───────────────────────────────────────────────────────────

def evaluate_integrity_rules(candidate: dict) -> tuple[bool, list[str], list[str]]:
    """Evaluates profile integrity using rule-based criteria instead of scores.
    Categorizes signals into CRITICAL, MAJOR, and MINOR.
    Returns: (should_reject, reject_reasons_list, minor_flags_list)
    
    Rejects candidate if:
      - Any 1 CRITICAL rule triggers
      - Any 2 MAJOR rules trigger
    """
    profile = candidate["profile"]
    history = candidate["career_history"]
    skills  = candidate["skills"]
    edu     = candidate.get("education", [])
    sig     = candidate["redrob_signals"]

    critical_triggers = []
    major_triggers = []
    minor_triggers = []

    # ── 1. Experience mismatch (Stated vs Documented) ──
    total_months  = sum(r["duration_months"] for r in history if r["duration_months"] > 0)
    stated_months = profile["years_of_experience"] * 12
    
    # Critical: Stated YoE vs Documented mismatch by >3 years
    if stated_months > total_months + 36:
        critical_triggers.append(
            f"Experience mismatch: claimed {profile['years_of_experience']}yr experience but only "
            f"{total_months//12:.0f}yr documented in career history"
        )
    elif total_months > stated_months + 36:
        critical_triggers.append(
            f"Experience mismatch: documented history ({total_months//12:.0f}yr) "
            f"greatly exceeds claimed experience ({profile['years_of_experience']}yr)"
        )

    # ── 2. Impossible education chronology ──
    if edu:
        parsed_edu = []
        for e in edu:
            try:
                sy = int(e.get("start_year", 0) or 0)
                ey = int(e.get("end_year", 0) or 0)
                parsed_edu.append((sy, ey, e.get("degree", "").lower()))
            except (ValueError, TypeError):
                continue

        parsed_edu.sort(key=lambda x: x[0])
        for i in range(len(parsed_edu) - 1):
            sy0, ey0, deg0 = parsed_edu[i]
            sy1, ey1, deg1 = parsed_edu[i + 1]
            # Normalize degree string (remove periods) so "m.s." -> "ms", "ph.d" -> "phd"
            d0_norm = deg0.replace(".", "").strip()
            d1_norm = deg1.replace(".", "").strip()
            higher = {"msc", "mtech", "ms", "mba", "phd", "doctorate", "me", "ma", "master"}
            if any(h in d0_norm for h in higher) and ey0 > 0 and sy1 > 0 and sy1 > ey0 + 2:
                if not any(h in d1_norm for h in higher):
                    # Masters degree finished before next undergrad started (impossible chronology)
                    critical_triggers.append(
                        f"Impossible education chronology: {deg0.upper()} ({sy0}–{ey0}) "
                        f"precedes {deg1.upper()} ({sy1}–{ey1})"
                    )

        # Check for overlapping degrees of the same level (e.g. Ira Dalal)
        degrees_by_level = {"bachelor": [], "master": [], "doctor": []}
        for e in edu:
            try:
                sy = int(e.get("start_year", 0) or 0)
                ey = int(e.get("end_year", 0) or 0)
                if sy <= 0 or ey <= 0:
                    continue
                deg = e.get("degree", "").lower().replace(".", "").strip()
                level = None
                if any(h in deg for h in ["phd", "doctor"]):
                    level = "doctor"
                elif any(h in deg for h in ["msc", "mtech", "ms", "mba", "me", "ma", "mca", "master", "pg"]):
                    level = "master"
                elif any(h in deg for h in ["btech", "be", "bsc", "ba", "bca", "bba", "bachelor", "ug"]):
                    level = "bachelor"
                if level:
                    degrees_by_level[level].append((sy, ey, e.get("degree", "")))
            except (ValueError, TypeError):
                continue

        for lvl, degs in degrees_by_level.items():
            if len(degs) >= 2:
                for i in range(len(degs)):
                    for j in range(i + 1, len(degs)):
                        sy0, ey0, d0 = degs[i]
                        sy1, ey1, d1 = degs[j]
                        overlap = min(ey0, ey1) - max(sy0, sy1) + 1
                        if overlap >= 2:
                            minor_triggers.append(
                                f"Overlapping {lvl} degrees: {d0} ({sy0}–{ey0}) & {d1} ({sy1}–{ey1})"
                            )

    # ── 3. Technology anachronism ──
    anachronistic_techs = [
        "rag", "llm", "bge embeddings", "bge", "llama", "chatgpt", "gpt-4",
        "vector database", "llm-based re-ranker", "pinecone", "weaviate",
        "qdrant", "milvus", "faiss hnsw",
    ]
    for role in history:
        end_date = role.get("end_date") or role.get("start_date")
        if not end_date:
            continue
        try:
            end_year = int(str(end_date)[:4])
        except ValueError:
            continue
        if end_year < 2021:
            desc_lower = role.get("description", "").lower()
            found = [t for t in anachronistic_techs if t in desc_lower]
            if len(found) >= 1:
                critical_triggers.append(
                    f"Technology anachronism: {', '.join(found[:3])} mentioned in "
                    f"role ending {end_year} at {role.get('company','?')}"
                )

    # ── 4. Duplicated career narratives (trigram threshold raised to 95%) ──
    descs = [r.get("description", "").strip().lower() for r in history if r.get("description")]
    if len(descs) >= 2:
        def _trigram_sim(a: str, b: str) -> float:
            def trigrams(s):
                words = s.split()
                return set(" ".join(words[i:i+3]) for i in range(len(words)-2))
            ta, tb = trigrams(a), trigrams(b)
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / max(len(ta), len(tb))

        high_sim_pairs = []
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                sim = _trigram_sim(descs[i], descs[j])
                if sim >= 0.95:
                    high_sim_pairs.append((i, j, sim))

        if high_sim_pairs:
            pair_desc = "; ".join(
                f"{history[i].get('company','?')} ↔ {history[j].get('company','?')}"
                for i, j, sim in high_sim_pairs
            )
            major_triggers.append(f"Duplicated role descriptions: {pair_desc}")

    # ── 5. Multiple employment overlaps ──
    definite_roles = [
        r for r in history
        if r.get("start_date") and r.get("end_date") and not r.get("is_current")
    ]
    if len(definite_roles) >= 2:
        try:
            sorted_roles = sorted(definite_roles, key=lambda r: str(r["start_date"]))
            overlapping = []
            for i in range(len(sorted_roles) - 1):
                end_i   = str(sorted_roles[i]["end_date"])[:7]
                start_j = str(sorted_roles[i + 1]["start_date"])[:7]
                if start_j < end_i:
                    overlapping.append(
                        f"{sorted_roles[i].get('company','?')} & {sorted_roles[i+1].get('company','?')}"
                    )
            if len(overlapping) >= 2:
                major_triggers.append(
                    "Multiple employment overlaps: " + ", ".join(overlapping[:2])
                )
            elif len(overlapping) == 1:
                minor_triggers.append(f"Employment date overlap: {overlapping[0]}")
        except Exception:
            pass

    # ── 5a. Company founding date validation (Honeypot detection) ──
    company_founding_years = {
        "krutrim": 2023,
        "sarvam ai": 2023,
        "sarvam": 2023,
        "xai": 2023,
        "mistral": 2023,
        "mistral ai": 2023,
        "anthropic": 2021,
        "cohere": 2019,
        "openai": 2015,
    }
    for role in history:
        comp_name = str(role.get("company", "")).strip().lower()
        clean_comp = comp_name.replace("inc.", "").replace("corp.", "").replace("ltd.", "").replace("software", "").strip()
        matched_year = None
        for k, yr in company_founding_years.items():
            if k in clean_comp:
                matched_year = yr
                break
        
        if matched_year:
            start_date = role.get("start_date")
            if start_date:
                try:
                    start_year = int(str(start_date)[:4])
                    if start_year < matched_year:
                        critical_triggers.append(
                            f"Experience mismatch: worked at {role.get('company')} starting in {start_year}, but company was founded in {matched_year}."
                        )
                except ValueError:
                    pass

    # ── 5b. Unexplained education-to-career gap ──
    if edu and history:
        try:
            grad_years = [int(e.get("end_year", 0) or 0) for e in edu if e.get("end_year")]
            job_starts = [
                datetime.strptime(r["start_date"], "%Y-%m-%d").year
                for r in history if r.get("start_date")
            ]
            if grad_years and job_starts:
                latest_grad = max(grad_years)
                earliest_job = min(job_starts)
                gap = earliest_job - latest_grad
                if gap >= 8:
                    # 8+ year gap with no career history in between = classic honeypot
                    critical_triggers.append(
                        f"Impossible timeline: graduated {latest_grad} but first job not until "
                        f"{earliest_job} ({gap}-year gap with no documented history)"
                    )
                elif gap >= 5:
                    major_triggers.append(
                        f"Large unexplained gap: {gap} years between graduation ({latest_grad}) "
                        f"and first job ({earliest_job})"
                    )
        except (ValueError, TypeError):
            pass

    # ── 6. Skill duration inflation ──
    if total_months > 0:
        # Allow a learning buffer: skills can predate first job by up to 4 years (48 months)
        # to account for B.Tech/M.Tech students learning during college.
        LEARNING_BUFFER_MONTHS = 48
        effective_span = total_months + LEARNING_BUFFER_MONTHS

        total_skill_months = sum(s.get("duration_months", 0) for s in skills)
        if total_skill_months > effective_span * 15:
            minor_triggers.append("Skill duration inflation")

        impossible_skills = [
            s["name"] for s in skills
            if s.get("duration_months", 0) > effective_span + 12
        ]
        expert_impossible = [
            s["name"] for s in skills
            if s.get("duration_months", 0) > effective_span + 12
            and s.get("proficiency") == "expert"
        ]

        # MAJOR: 3+ skills with impossible durations (strong honeypot signal)
        if len(impossible_skills) >= 3:
            major_triggers.append(
                f"Impossible skill durations ({len(impossible_skills)} skills exceed career + college span): "
                + ", ".join(impossible_skills[:4])
            )
        # MAJOR: even 1 expert skill claimed for longer than entire career + college span
        elif len(expert_impossible) >= 1:
            major_triggers.append(
                f"Expert skill duration exceeds career + college span: "
                + ", ".join(expert_impossible[:3])
            )
        elif len(impossible_skills) >= 1:
            minor_triggers.append(f"Skill duration inflation: {', '.join(impossible_skills[:3])}")

    # ── 7. Sparse profile vs seniority ──
    completeness = sig["profile_completeness_score"]
    if completeness < 20 and profile["years_of_experience"] > 10:
        minor_triggers.append("Sparse profile")

    # Decide Reject
    should_reject = len(critical_triggers) >= 1 or len(major_triggers) >= 2
    reject_reasons = critical_triggers + major_triggers
    
    return should_reject, reject_reasons, minor_triggers


def compute_consistency(candidate: dict) -> float:
    """Delegates to rules. Returns 0.0 if rejected, 1.0 otherwise."""
    rejected, _, _ = evaluate_integrity_rules(candidate)
    return 0.0 if rejected else 1.0


# ── Reasoning generator ────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, scores: dict) -> str:
    """
    Generates evidence-based narrative reasoning following the template:
      Role + Exp | Core production work | Retrieval/Ranking ownership |
      Evaluation ownership | Shipping evidence | Availability/notice
    No keyword lists — every claim is grounded in profile evidence.
    """
    profile  = candidate["profile"]
    history  = candidate["career_history"]
    skills   = candidate["skills"]
    sig       = candidate["redrob_signals"]

    title = profile.get("current_title", "Unknown")
    yoe   = profile["years_of_experience"]

    # ── Build full-text corpus for signal detection ──
    search_text = (
        profile.get("summary", "") + " "
        + " ".join(r.get("description", "") for r in history)
    ).lower()
    skill_names_lower = " ".join(s["name"].lower() for s in skills)
    full_text = search_text + " " + skill_names_lower

    # ── Score sub-dimensions ──
    eval_score       = scores.get("evaluation", 0.0)
    ownership_score  = scores.get("ownership", 0.0)
    production_score = scores.get("production", 0.0)
    sem              = scores.get("semantic", 0.0)
    minor_flags      = scores.get("minor_flags", [])

    # ── Signal detection helpers ──
    has_bm25          = any(p in full_text for p in ["bm25", "tf-idf", "sparse retrieval", "keyword search"])
    has_dense         = any(p in full_text for p in ["dense retrieval", "dense vector", "vector search",
                                                      "faiss", "milvus", "qdrant", "pinecone", "weaviate",
                                                      "ann search", "hnsw", "embeddings", "sentence-transformers"])
    has_hybrid        = any(p in full_text for p in ["hybrid retrieval", "hybrid search", "bm25 + dense",
                                                      "sparse and dense", "sparse + dense"])
    has_ltr           = any(p in full_text for p in ["learning-to-rank", "learning to rank", "ltr",
                                                      "xgboost rank", "lambdamart", "ranknet"])
    has_ndcg          = "ndcg" in full_text
    has_mrr           = "mrr" in full_text
    has_ab            = any(p in full_text for p in ["a/b test", "ab test", "a/b testing", "online experiment",
                                                      "online a/b"])
    has_recruiter     = any(p in full_text for p in ["recruiter", "recruiter-facing", "recruiter search",
                                                      "recruiter feedback", "candidate search", "candidate matching",
                                                      "candidate-jd", "jd matching"])
    has_prod_deploy   = any(p in full_text for p in ["shipped", "production", "prod", "live", "deployed",
                                                      "serving", "queries per month", "qps", "p95", "p99"])
    has_search_rel    = any(p in full_text for p in ["search relevance", "relevance improvement",
                                                      "relevance judgment", "relevance label",
                                                      "time-to-shortlist", "engagement metric"])

    # ── Part 1: Role + Exp (always present) ──
    parts = [f"{title} ({yoe}yr)"]

    # ── Part 2: Core production work ──
    if production_score >= 0.70:
        if has_recruiter:
            parts.append("who built production recruiter-facing search and ranking systems")
        elif "recommend" in full_text:
            parts.append("who built and shipped production recommendation and ranking systems")
        else:
            parts.append("who shipped production retrieval/ranking systems")

    # ── Part 3: Hybrid retrieval evidence ──
    if has_hybrid or (has_bm25 and has_dense):
        # Synthesise into one evidence statement — no list
        parts.append("Led hybrid retrieval combining sparse (BM25) and dense embeddings")
    elif has_bm25:
        parts.append("Owned BM25/sparse retrieval pipeline")
    elif has_dense:
        parts.append("Owned dense-vector retrieval pipeline")

    # ── Part 4: Learning-to-rank ──
    if has_ltr and ownership_score >= 0.60:
        parts.append("owned learning-to-rank")

    # ── Part 5: Evaluation evidence (NDCG/MRR + A/B) ──
    eval_parts = []
    if has_ndcg and has_mrr:
        eval_parts.append("NDCG/MRR")
    elif has_ndcg:
        eval_parts.append("NDCG")
    elif has_mrr:
        eval_parts.append("MRR")
    if has_ab:
        eval_parts.append("A/B testing")

    if eval_parts and eval_score >= 0.50:
        parts.append(f"and offline→online evaluation ({'/'.join(eval_parts)})")
    elif eval_score >= 0.50:
        parts.append("and designed offline/online evaluation framework")

    # ── Part 6: Shipping / deployment evidence ──
    if has_prod_deploy and production_score >= 0.70:
        parts.append("Shipped production relevance improvements")

    # ── Part 7: Search relevance / recruiter-feedback specifics ──
    if has_search_rel or has_recruiter:
        if has_recruiter and has_search_rel:
            parts.append("with recruiter-search relevance ownership")
        elif has_recruiter:
            parts.append("with recruiter-facing search experience")
        else:
            parts.append("with search relevance engineering evidence")

    # ── Part 8: Semantic alignment (brief, no list) ──
    if sem >= 0.75:
        parts.append("Strong semantic match to Redrob's intelligence-layer ownership")

    # ── Part 9: Availability ──
    avail = compute_availability_score(sig)
    if avail <= 0.30:
        parts.append("⚠️ LOW AVAILABILITY — inactive >90 days")
    elif avail >= 0.85:
        parts.append("Highly available (recently active)")

    notice = sig["notice_period_days"]
    if notice <= 30:
        parts.append(f"Notice: {notice}d ✓")
    elif notice > 90:
        parts.append(f"Notice: {notice}d (long)")

    otw = "open" if sig["open_to_work_flag"] else "not flagged open"
    parts.append(f"OTW: {otw}")

    # ── Integrity flags ──
    if minor_flags:
        for flag in minor_flags:
            parts.append(f"⚠️ {flag}")

    # Join with " | " for readability — first part has no leading pipe
    return parts[0] + " | " + " | ".join(parts[1:]) if len(parts) > 1 else parts[0]


# ── Master scorer ──────────────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> tuple[float, str]:
    """Returns (final_score, reasoning_string) for a single candidate."""
    rejected, reject_reasons, minor_flags = evaluate_integrity_rules(candidate)
    if rejected:
        reason_summary = " • ".join(reject_reasons)
        return 0.0, f"FILTERED: Reason: {reason_summary}"

    cid = candidate["candidate_id"]
    idx = CID_TO_IDX.get(cid)
    if idx is None:
        return 0.0, f"FILTERED: candidate_id {cid} not found in embedding artifacts"

    semantic   = compute_semantic_score(idx)
    skill      = compute_skill_score(candidate)
    production = compute_production_score(candidate)
    career     = compute_career_depth_score(candidate)
    title      = compute_title_score(candidate)
    ownership  = compute_ownership_score(candidate)
    evaluation = compute_eval_score(candidate)
    retrieval  = compute_retrieval_score(candidate)
    behavioral = compute_behavioral_score(candidate)

    domain_bonus = compute_domain_bonus(candidate)
    hybrid_bonus = compute_hybrid_bonus(candidate)
    ops_bonus    = compute_ops_bonus(candidate)
    boring_bonus = compute_boring_bonus(candidate)

    tvp             = _compute_title_velocity_penalty(candidate)
    career_adjusted = max(0.0, career - tvp)

    location = compute_location_score(candidate)

    # Base score weights sum to 1.00
    base = (
        0.20 * semantic +
        0.08 * skill +
        0.25 * production +
        0.07 * career_adjusted +
        0.05 * title +
        0.12 * ownership +
        0.12 * evaluation +
        0.08 * retrieval +
        0.03 * location
    )

    # Behavioral weight: 0.75 * base + 0.25 * behavioral
    final = round(0.75 * base + 0.25 * behavioral + domain_bonus + hybrid_bonus + ops_bonus + boring_bonus, 4)

    # Conditional CV/Speech Penalty
    full = (
        candidate["profile"]["summary"] + " " +
        " ".join(r["description"] for r in candidate["career_history"])
    ).lower()

    padded_full = f" {full} "
    is_cv = any(
        kw in full for kw in [
            "computer vision",
            "object detection",
            "segmentation",
            "opencv",
            "yolo",
            "speech recognition",
        ]
    ) or " asr " in padded_full

    has_ir = (
        retrieval > 0.20 or
        evaluation > 0.20 or
        production > 0.60
    )

    if is_cv and not has_ir:
        final = round(final * 0.95, 4)

    # Marketplace/Search Bonus
    marketplace_bonus = 0.0
    if any(
        kw in full for kw in [
            "recommendation system",
            "recommendation engine",
            "search ranking",
            "candidate matching",
            "retrieval system",
            "marketplace",
            "relevance ranking"
        ]
    ):
        marketplace_bonus = 0.03

    final = round(final + marketplace_bonus, 4)
    final = min(final, 1.0)

    # consulting-only current employer penalty
    current_employer_services_penalty = 0.0
    if candidate.get("career_history"):
        current_company = candidate["career_history"][0].get("company", "")
        if current_company in SERVICE_COMPANIES:
            current_employer_services_penalty = 0.05
            final = round(max(0.0, final - current_employer_services_penalty), 4)

    # Strict downweight for completely non-AI/non-Engineering titles (JD trap avoidance)
    current_title = candidate["profile"].get("current_title", "")
    non_tech_titles = {
        "Marketing Manager", "HR Manager", "Civil Engineer", "Mechanical Engineer",
        "Operations Manager", "Project Manager", "Sales Executive", "Graphic Designer",
        "Content Writer", "Customer Support", "Accountant", "Business Analyst"
    }
    if current_title in non_tech_titles:
        final = round(final * 0.60, 4)

    is_research = "research" in candidate["profile"].get("current_title", "").lower()
    research_penalty_applied = False
    if is_research and production < 0.40:
        final = round(final * 0.92, 4)
        research_penalty_applied = True

    # Downweight highly inactive/unavailable candidates (e.g. Candidate #10)
    availability_penalty = 0.0
    sig = candidate["redrob_signals"]
    last_active = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
    days_ago    = (REFERENCE_DATE - last_active).days
    if days_ago > 120 and not sig["open_to_work_flag"]:
        availability_penalty += 0.05
    if sig["recruiter_response_rate"] < 0.15:
        availability_penalty += 0.05

    if availability_penalty > 0.0:
        final = round(max(0.0, final - availability_penalty), 4)

    # ── Soft Integrity Penalty for Overlapping degrees of same level ──
    for flag in minor_flags:
        if "Overlapping" in flag:
            final = round(max(0.0, final - 0.02), 4)

    # ── Score spread sharpening ────────────────────────────────────────────────
    # Apply a mild power transform to pull apart bunched top scores.
    # f(x) = x^0.85 maps 1.0→1.0, 0.95→0.957, 0.90→0.913, 0.80→0.822
    # This gives a modest but meaningful push to the gaps at the top.
    final = round(float(np.power(final, 0.85)), 4)
    final = min(final, 1.0)

    reasoning = generate_reasoning(candidate, {
        "semantic": semantic, "skill": skill, "production": production,
        "career": career_adjusted, "title": title, "behavioral": behavioral,
        "ownership": ownership, "evaluation": evaluation, "retrieval": retrieval,
        "title_velocity_penalty": tvp,
        "research_penalty_applied": research_penalty_applied,
        "minor_flags": minor_flags,
    })

    return final, reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — score all candidates and write CSV
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Score all candidates and produce submission CSV.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found.")
        return

    print(f"\n{'='*60}")
    print("Redrob AI Candidate Ranker — newrank.py (v7 Optimized)")
    print(f"{'='*60}")
    print(f"Input:  {candidates_path}")
    print(f"Output: {args.out}\n")

    results = []
    filtered_count = 0
    n_processed = 0

    print("Scoring candidates ...")
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            n_processed += 1

            score, reasoning = score_candidate(candidate)
            if score == 0.0:
                filtered_count += 1

            results.append({
                "candidate_id": candidate["candidate_id"],
                "score":        score,
                "reasoning":    reasoning,
                "candidate":    candidate,
                "raw_line":     line,
            })

            if n_processed % 10000 == 0:
                print(f"  Processed {n_processed:,} candidates ...")

    print(f"\n  Total processed:  {n_processed:,}")
    print(f"  Filtered (score=0): {filtered_count:,} ({filtered_count/n_processed*100:.2f}%)")

    # Sort by score descending, break ties by candidate_id ascending
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Write top 10 raw candidate lines to to_check and topresume
    with open("to_check", "w", encoding="utf-8") as f_check:
        for r in results[:10]:
            f_check.write(r["raw_line"] + "\n")
    print("✅  Top 10 raw candidates written to: to_check")

    with open("topresume", "w", encoding="utf-8") as f_topresume:
        for r in results[:10]:
            f_topresume.write(r["raw_line"] + "\n")
    print("✅  Top 10 raw candidates written to: topresume")

    # Prepare top 100 rows with rank
    top_100_results = []
    top_resumes_json = []
    for rank, r in enumerate(results[:100], 1):
        top_100_results.append({
            "candidate_id": r["candidate_id"],
            "rank":         rank,
            "score":        r["score"],
            "reasoning":    r["reasoning"],
        })
        
        cand = r["candidate"]
        top_resumes_json.append({
            "rank": rank,
            "score": r["score"],
            "reasoning": r["reasoning"],
            "candidate_id": r["candidate_id"],
            "profile": cand.get("profile", {}),
            "career_history": cand.get("career_history", []),
            "skills": cand.get("skills", []),
            "redrob_signals": cand.get("redrob_signals", {}),
            "education": cand.get("education", []),
        })

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(top_100_results)

    # Write JSON with top resumes
    json_out_path = out_path.parent / "top_resumes.json"
    with open(json_out_path, "w", encoding="utf-8") as jsonfile:
        json.dump(top_resumes_json, jsonfile, indent=2, ensure_ascii=False)

    # Write beautiful human-readable MD file for top 20 candidates
    md_out_path = out_path.parent / "top_resumes.md"
    with open(md_out_path, "w", encoding="utf-8") as mdfile:
        mdfile.write("# Top Candidates Ranking Analysis\n\n")
        mdfile.write("This file contains the detailed profiles (resumes) of the top-ranked candidates to help you understand the ranking decision.\n\n")
        mdfile.write("| Rank | Candidate ID | Score | Title | Years of Exp | Country | Recent Company | Open to Work | Notice Period | \n")
        mdfile.write("|---|---|---|---|---|---|---|---|---|\n")
        
        for r in top_resumes_json[:20]:
            p = r["profile"]
            history = r["career_history"]
            recent_company = history[0]["company"] if history else "N/A"
            sig = r["redrob_signals"]
            otw = "Yes" if sig.get("open_to_work_flag") else "No"
            notice = f"{sig.get('notice_period_days', 999)}d"
            mdfile.write(f"| {r['rank']} | {r['candidate_id']} | {r['score']:.4f} | {p.get('current_title', 'N/A')} | {p.get('years_of_experience', 0)} | {p.get('country', 'N/A')} | {recent_company} | {otw} | {notice} |\n")
        
        mdfile.write("\n---\n\n## Detailed Resumes of Top-10 Candidates\n\n")
        for r in top_resumes_json[:10]:
            p = r["profile"]
            history = r["career_history"]
            skills = r["skills"]
            sig = r["redrob_signals"]
            edu = r["education"]
            
            mdfile.write(f"### Rank #{r['rank']}: Candidate {r['candidate_id']} (Score: {r['score']:.4f})\n\n")
            mdfile.write(f"**Current Title:** {p.get('current_title', 'N/A')}  \n")
            mdfile.write(f"**Experience:** {p.get('years_of_experience', 0)} years  \n")
            mdfile.write(f"**Country:** {p.get('country', 'N/A')}  \n")
            mdfile.write(f"**Anonymized Name:** {p.get('anonymized_name', 'N/A')}  \n\n")
            
            mdfile.write(f"#### 💡 System Reasoning\n> {r['reasoning']}\n\n")
            
            mdfile.write("#### 📝 Professional Summary\n")
            summary_text = p.get("summary", "No summary provided.").strip()
            mdfile.write(f"{summary_text}\n\n")
            
            mdfile.write("#### 💼 Career History\n")
            for job in history:
                mdfile.write(f"- **{job.get('title', 'N/A')}** at *{job.get('company', 'N/A')}* ({job.get('start_date', 'N/A')} - {job.get('end_date', 'Present')}, {job.get('duration_months', 0)} months)\n")
                mdfile.write(f"  - *Industry:* {job.get('industry', 'N/A')}\n")
                desc = job.get('description', '').replace('\n', '\n  ')
                mdfile.write(f"  - *Description:* {desc}\n\n")
                
            mdfile.write("#### 🛠️ Skills\n")
            skills_sorted = sorted(skills, key=lambda s: -s.get("endorsements", 0))
            skills_line = []
            for s in skills_sorted:
                skills_line.append(f"{s.get('name')} ({s.get('proficiency')}, {s.get('duration_months', 0)}m, {s.get('endorsements', 0)} endorsements)")
            mdfile.write(", ".join(skills_line) + "\n\n")
            
            mdfile.write("#### 🎓 Education\n")
            for school in edu:
                mdfile.write(f"- **{school.get('degree', 'N/A')} in {school.get('field_of_study', 'N/A')}** - {school.get('institution', 'N/A')} (Grade: {school.get('grade', 'N/A')})\n")
            mdfile.write("\n")
            
            mdfile.write("#### ⚡ Platform & Behavioral Signals\n")
            mdfile.write(f"- **Open to Work:** {'Yes' if sig.get('open_to_work_flag') else 'No'}\n")
            mdfile.write(f"- **Notice Period:** {sig.get('notice_period_days', 999)} days\n")
            mdfile.write(f"- **GitHub Activity Score:** {sig.get('github_activity_score', -1)}\n")
            mdfile.write(f"- **Recruiter Response Rate:** {sig.get('recruiter_response_rate', 0.0):.2%}\n")
            mdfile.write(f"- **Profile Completeness:** {sig.get('profile_completeness_score', 0)}%\n")
            mdfile.write("\n---\n\n")

    print(f"✅  Top resumes JSON written to: {json_out_path.resolve()}")
    print(f"✅  Top resumes MD written to: {md_out_path.resolve()}")

    print(f"\n✅  Submission written to: {out_path.resolve()}")
    print(f"\nTop-10 candidates:")
    print(f"{'Rank':<6} {'Candidate ID':<40} {'Score':<8} Reasoning snippet")
    print("-" * 100)
    for rank, r in enumerate(top_100_results[:10], 1):
        snippet = r["reasoning"][:70] + "..." if len(r["reasoning"]) > 70 else r["reasoning"]
        print(f"{rank:<6} {r['candidate_id']:<40} {r['score']:<8.4f} {snippet}")

    # Score distribution of top-100
    print(f"\nScore distribution (top 100):")
    print(f"  #1 score:   {top_100_results[0]['score']:.4f}")
    print(f"  #10 score:  {top_100_results[9]['score']:.4f}")
    print(f"  #50 score:  {top_100_results[49]['score']:.4f}")
    print(f"  #100 score: {top_100_results[99]['score']:.4f}")
    top10_gap = top_100_results[0]["score"] - top_100_results[9]["score"]
    print(f"  Top-10 gap: {top10_gap:.4f}  ({'✅ > 0.03' if top10_gap > 0.03 else '⚠️ < 0.03 — check spread'})")

    print(f"\nNext: python validate_submission.py {args.out}")
    print(f"      python audit.py --submission {args.out} --top 50")


if __name__ == "__main__":
    main()
