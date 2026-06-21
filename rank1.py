#!/usr/bin/env python3
"""
rank.py  —  Redrob AI Candidate Ranker  (v6)
Scores all 100K candidates using pre-computed artifacts.
Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
"""
import argparse, csv, json, pickle
from datetime import datetime, date
from pathlib import Path

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

# ── Keyword Lists (identical to precompute.py — must stay in sync) ─────────────
PROD_T1 = [
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
PROD_T2 = [
    "shipped", "launched", "built end-to-end", "owned", "led design",
    "search", "ranking", "retrieval", "recommendation", "matching",
    "pipeline", "infrastructure", "architecture",
    # v6: BM25-migration signals
    "bm25", "tf-idf", "keyword search", "migrated", "migrated from",
    "replaced", "improved precision", "improved recall", "reduced latency",
    "recruiter engagement", "click-through rate", "conversion rate",
    "hybrid retrieval", "hybrid search", "re-ranking", "reranking",
]
RESEARCH_NEGS = [
    "arxiv", "research lab", "academic", "thesis",
    "proof of concept", "prototype only", "exploration paper",
    # v6: framework-enthusiast signals
    "langchain tutorial", "how i built", "getting started with",
    "demo project", "side project only",
]

PROFICIENCY_WEIGHT = {
    "beginner": 0.50, "intermediate": 0.75, "advanced": 1.00, "expert": 1.10,
}

# ── Title score dictionary — full coverage ─────────────────────────────────────
# Sourced from EDA: every title in the 100K pool is covered, no gaps
TITLE_SCORES = {
    # Exact AI/search match — highest tier
    "Search Engineer":                1.00,
    "Recommendation Systems Engineer": 1.00,
    "Applied ML Engineer":             0.98,
    "Machine Learning Engineer":       0.97,
    "AI Research Engineer":            0.95,
    "Senior Machine Learning Engineer": 0.95,
    "Senior Software Engineer(ML)":    0.95,
    "NLP Engineer":                    0.95,
    "AI Specialist":                   0.90,
    "Computer Vision Engineer":        0.80,  # domain mismatch penalty
    # Strong adjacent — engineers who commonly do ML work
    "Senior Software Engineer":        0.78,
    "Analytics Engineer":              0.75,
    "Backend Engineer":                0.72,
    "Software Engineer":               0.68,
    "Data Scientist":                  0.70,
    "Data Engineer":                   0.60,
    "Senior Data Engineer":            0.62,
    # Neutral technical — non-ML but engineering
    "Full Stack Developer":            0.50,
    "Cloud Engineer":                  0.48,
    "DevOps Engineer":                 0.45,
    "Java Developer":                  0.40,
    ".NET Developer":                  0.38,
    "Mobile Developer":                0.38,
    "Frontend Engineer":               0.35,
    "QA Engineer":                     0.32,
    "Data Analyst":                    0.38,
    # Non-technical management
    "Project Manager":                 0.25,
    "Business Analyst":                0.22,
    "Operations Manager":              0.20,
    # Non-technical / irrelevant
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
TITLE_DEFAULT = 0.35  # generic engineering titles not in dict

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
# Tenure-weighted in compute_career_depth_score — so a current Infosys stint
# with prior product experience doesn't unfairly penalize the candidate (EDA confirmed)
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
    # Consulting / IT services (soft penalty per JD)
    "TCS": 0.50, "Infosys": 0.50, "Wipro": 0.50,
    "Accenture": 0.50, "Cognizant": 0.50, "Capgemini": 0.50,
    "HCL": 0.50, "Tech Mahindra": 0.52, "Mphasis": 0.52,
}
COMPANY_DEFAULT = 0.65  # unknown companies — neutral, not penalized

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
    "canada": 0.55,
    "uk": 0.55, "united kingdom": 0.55,
    "australia": 0.55,
    "germany": 0.55,
    "singapore": 0.60,
    "uae": 0.65,
}

# ── v6: Title velocity penalty ─────────────────────────────────────────────────
# JD: "title-chasers switching companies every 1.5 years" is named anti-pattern
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
        return 0.0  # Not a short-tenure hopper

    levels_seen = set()
    for r in history:
        title_lower = r.get("title", "").lower()
        for level_kw, level_num in TITLE_LEVEL_MAP.items():
            if level_kw in title_lower:
                levels_seen.add(level_num)
                break

    return 0.05 if len(levels_seen) >= 3 else 0.0


# ── v6: Pre-LLM experience check ───────────────────────────────────────────────
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


def compute_career_depth_score(candidate: dict) -> float:
    """YoE band + tenure-weighted company score + stability + industry recency bonus."""
    profile  = candidate["profile"]
    history  = candidate["career_history"]
    yoe      = profile["years_of_experience"]

    # YoE score — 4–9yr is 1.0 (JD sweet spot; 4yr added per JD language)
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
    else:  # > 11
        yoe_score = max(0.40, 0.80 - (yoe - 11) * 0.04)

    # Tenure-weighted company score
    total_months = sum(r["duration_months"] for r in history if r["duration_months"] > 0)
    if total_months == 0:
        company_score = COMPANY_DEFAULT
    else:
        weighted_sum = sum(
            COMPANY_SCORES.get(r["company"], COMPANY_DEFAULT) * r["duration_months"]
            for r in history if r["duration_months"] > 0
        )
        company_score = weighted_sum / total_months

    # Stability — reward longer average tenure
    valid_tenures = [r["duration_months"] for r in history if r["duration_months"] > 0]
    if valid_tenures:
        avg_tenure = sum(valid_tenures) / len(valid_tenures)
        # Scale: 0 months=0.30, 18 months=0.65, 24 months=0.80, 36+ months=1.0
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

    # Industry recency bonus — most recent 2 roles
    recent_roles = sorted(history, key=lambda r: r.get("start_date", ""), reverse=True)[:2]
    industry_bonus = 0.0
    for r in recent_roles:
        if r.get("industry", "") in HIGH_RELEVANCE_INDUSTRIES:
            industry_bonus = min(industry_bonus + 0.08, 0.15)

    raw = (0.40 * yoe_score + 0.35 * company_score + 0.20 * stability + 0.05 * industry_bonus)
    return float(np.clip(raw, 0.0, 1.0))


def compute_title_score(candidate: dict) -> float:
    """Soft gradient title score — full dataset coverage, no hard zeroes."""
    title = candidate["profile"]["current_title"]
    return TITLE_SCORES.get(title, TITLE_DEFAULT)


def compute_education_score(candidate: dict) -> float:
    """Tier + field relevance, capped at 1.0."""
    edu = candidate.get("education", [])
    if not edu:
        return 0.30  # floor for missing edu — not a disqualifier

    best = 0.0
    for entry in edu:
        institution = entry.get("institution", "")
        field       = entry.get("field_of_study", "").lower()
        degree_type = entry.get("degree", "").lower()

        # Institution tier
        tier = 0.50  # default for unknown institution
        for kw, val in EDU_TIER.items():
            if kw.lower() in institution.lower():
                tier = val
                break

        # Field boost
        field_boost = 0.0
        for kw, boost in EDU_FIELD_BOOST.items():
            if kw in field:
                field_boost = max(field_boost, boost)

        # PhD gets a small bonus (for research depth — noted in JD as nice-to-have)
        degree_bonus = 0.05 if "phd" in degree_type or "doctorate" in degree_type else 0.0

        score = min(tier + field_boost + degree_bonus, 1.0)
        best  = max(best, score)

    return float(np.clip(best, 0.0, 1.0))


def compute_location_score(candidate: dict) -> float:
    """Additive location component — not a multiplier, not a gate."""
    country = candidate["profile"].get("current_location", {}).get("country", "").lower()
    return LOCATION_SCORES.get(country, 0.40)


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
        recency = 0.10  # >180 days — essentially unavailable

    otw = 1.0 if sig["open_to_work_flag"] else 0.50
    return float(0.60 * recency + 0.40 * otw)


def compute_engagement_score(sig: dict) -> float:
    """Recruiter response rate + interview completion + offer acceptance.
    oar=-1 (no offer history) → 0.50 neutral, not 0.0 (verified safe in EDA).
    """
    rr  = sig["recruiter_response_rate"]
    icr = sig["interview_completion_rate"]
    raw_oar = sig["offer_acceptance_rate"]
    oar = 0.50 if raw_oar < 0 else raw_oar  # -1 → neutral

    return float(np.clip(0.50 * rr + 0.30 * icr + 0.20 * oar, 0.0, 1.0))


def compute_momentum_score(sig: dict) -> float:
    """30-day platform activity — saves, applications, views, search appearances."""
    saves  = min(sig["saved_by_recruiters_30d"] / 8.0, 1.0)
    apps   = min(sig["applications_submitted_30d"] / 5.0, 1.0)
    views  = min(sig["profile_views_received_30d"] / 20.0, 1.0)
    search = min(sig["search_appearance_30d"] / 100.0, 1.0)
    return float(0.30 * saves + 0.25 * apps + 0.25 * views + 0.20 * search)


def compute_github_score(sig: dict) -> float:
    """GitHub activity — -1 maps to 0.15 floor (not zero; absence ≠ no coding)."""
    gh = sig["github_activity_score"]
    if gh < 0:
        return 0.15  # floor — we don't penalize hard for missing GitHub
    return float(np.clip(gh / 80.0, 0.0, 1.0))


def compute_notice_score(sig: dict) -> float:
    """Notice period alignment — JD prizes sub-30d, can buy out up to 30d."""
    days = sig["notice_period_days"]
    if days <= 30:
        return 1.00
    elif days <= 60:
        return 0.75
    elif days <= 90:
        return 0.50
    elif days <= 120:
        return 0.30
    else:
        return 0.15


def compute_behavioral_score(candidate: dict) -> float:
    """Weighted behavioral score — availability acts as gate modifier."""
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


# ── Consistency gate ───────────────────────────────────────────────────────────

def compute_consistency(candidate: dict) -> float:
    """Hard gate — profiles with consistency < 0.70 score 0.0.
    EDA: 99.96% of candidates score 1.0 — this catches the 43 genuine outliers.
    """
    profile  = candidate["profile"]
    history  = candidate["career_history"]
    skills   = candidate["skills"]
    sig      = candidate["redrob_signals"]

    score = 1.0
    total_months  = sum(r["duration_months"] for r in history)
    stated_months = profile["years_of_experience"] * 12

    if total_months > stated_months + 36:
        score -= 0.50
    elif total_months > stated_months + 18:
        score -= 0.25

    expert_zero = [s for s in skills
                   if s["proficiency"] in ("expert", "advanced") and s.get("duration_months", 0) == 0]
    if len(expert_zero) >= 3:
        score -= 0.35
    elif len(expert_zero) >= 1:
        score -= 0.10

    inflated = [s for s in skills
                if s["proficiency"] == "expert"
                and s["endorsements"] == 0
                and sig["skill_assessment_scores"].get(s["name"], 50) < 40]
    if len(inflated) >= 4:
        score -= 0.25

    completeness = sig["profile_completeness_score"]
    if completeness < 20 and profile["years_of_experience"] > 10:
        score -= 0.15

    return max(0.0, score)


# ── Reasoning generator ────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, scores: dict) -> str:
    """Generates specific, non-templated reasoning string for the candidate."""
    profile  = candidate["profile"]
    history  = candidate["career_history"]
    skills   = candidate["skills"]

    name      = profile.get("name", "Candidate")
    title     = profile.get("current_title", "Unknown")
    yoe       = profile["years_of_experience"]
    country   = profile.get("current_location", {}).get("country", "")
    sig       = candidate["redrob_signals"]

    # Top skills (by relevance proxy — those with advanced/expert + endorsements)
    top_skills = sorted(
        [s["name"] for s in skills
         if s["proficiency"] in ("expert", "advanced") and s["endorsements"] > 0],
        key=lambda s: -next((sk["endorsements"] for sk in skills if sk["name"] == s), 0)
    )[:3]

    # Most recent company
    recent_company = history[0]["company"] if history else "N/A"
    recent_industry = history[0].get("industry", "") if history else ""

    # Build reasoning
    parts = []

    # Opening: title + yoe
    parts.append(f"{title} with {yoe}yr exp at {recent_company}")
    if country:
        parts.append(f"({country})")

    # Semantic signal
    sem = scores.get("semantic", 0)
    if sem >= 0.75:
        parts.append("| Strong profile-JD alignment (semantic)")
    elif sem >= 0.50:
        parts.append("| Moderate profile-JD alignment")
    else:
        parts.append("| Low semantic alignment")

    # Skill signal
    skill = scores.get("skill", 0)
    if top_skills:
        parts.append(f"| Key skills: {', '.join(top_skills)}")
    elif skill >= 0.50:
        parts.append("| Relevant skills present")
    else:
        parts.append("| Limited relevant skills")

    # Production signal
    prod = scores.get("production", 0)
    if prod >= 0.70:
        parts.append("| Strong production evidence")
    elif prod >= 0.40:
        parts.append("| Some production evidence")
    else:
        parts.append("| Limited production evidence")

    # Company / industry
    company_score = COMPANY_SCORES.get(recent_company, COMPANY_DEFAULT)
    if company_score >= 0.85:
        parts.append(f"| Premium product company background ({recent_company})")
    elif company_score <= 0.52:
        parts.append(f"| Services background ({recent_company}) — check career history")
    if recent_industry in HIGH_RELEVANCE_INDUSTRIES:
        parts.append(f"| Relevant industry: {recent_industry}")

    # Title velocity warning
    tvp = scores.get("title_velocity_penalty", 0)
    if tvp > 0:
        parts.append("| NOTE: Short avg tenure with multiple title jumps detected")

    # Behavioral
    beh = scores.get("behavioral", 0)
    avail = compute_availability_score(sig)
    if avail <= 0.30:
        parts.append("| ⚠️ LOW AVAILABILITY — inactive >90 days")
    elif avail >= 0.85:
        parts.append("| Highly available (recently active)")

    notice = sig["notice_period_days"]
    if notice <= 30:
        parts.append(f"| Notice: {notice}d ✓")
    elif notice > 90:
        parts.append(f"| Notice: {notice}d (long)")

    otw = "open" if sig["open_to_work_flag"] else "not flagged open"
    parts.append(f"| OTW: {otw}")

    return " ".join(parts)


# ── Master scorer ──────────────────────────────────────────────────────────────

def score_candidate(candidate: dict) -> tuple[float, str]:
    """Returns (final_score, reasoning_string) for a single candidate."""
    # Hard gate: consistency
    if compute_consistency(candidate) < 0.70:
        return 0.0, "FILTERED: inconsistent profile signals (consistency < 0.70)"

    # Get embedding index
    cid = candidate["candidate_id"]
    idx = CID_TO_IDX.get(cid)
    if idx is None:
        return 0.0, f"FILTERED: candidate_id {cid} not found in embedding artifacts"

    # Component scores
    semantic   = compute_semantic_score(idx)
    skill      = compute_skill_score(candidate)
    production = compute_production_score(candidate)
    career     = compute_career_depth_score(candidate)
    title      = compute_title_score(candidate)
    education  = compute_education_score(candidate)
    location   = compute_location_score(candidate)
    behavioral = compute_behavioral_score(candidate)

    # v6: title velocity penalty applied to career component
    tvp             = _compute_title_velocity_penalty(candidate)
    career_adjusted = max(0.0, career - tvp)

    # Base score (80% of final)
    base = (0.28 * semantic        +
            0.22 * skill           +
            0.18 * production      +
            0.15 * career_adjusted +
            0.12 * title           +
            0.05 * education       +
            0.05 * location)
    # No −0.05 offset — natural ceiling from components

    # Final score
    final = round(0.80 * base + 0.20 * behavioral, 4)

    reasoning = generate_reasoning(candidate, {
        "semantic": semantic, "skill": skill, "production": production,
        "career": career_adjusted, "title": title, "behavioral": behavioral,
        "title_velocity_penalty": tvp,
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
    print("Redrob AI Candidate Ranker — rank.py (v6)")
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
            })

            if n_processed % 10000 == 0:
                print(f"  Processed {n_processed:,} candidates ...")

    print(f"\n  Total processed:  {n_processed:,}")
    print(f"  Filtered (score=0): {filtered_count:,} ({filtered_count/n_processed*100:.2f}%)")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["candidate_id", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅  Submission written to: {out_path.resolve()}")
    print(f"\nTop-10 candidates:")
    print(f"{'Rank':<6} {'Candidate ID':<40} {'Score':<8} Reasoning snippet")
    print("-" * 100)
    for rank, r in enumerate(results[:10], 1):
        snippet = r["reasoning"][:70] + "..." if len(r["reasoning"]) > 70 else r["reasoning"]
        print(f"{rank:<6} {r['candidate_id']:<40} {r['score']:<8.4f} {snippet}")

    # Score distribution of top-1000
    top1000_scores = [r["score"] for r in results[:1000]]
    print(f"\nScore distribution (top 1000):")
    print(f"  #1 score:   {results[0]['score']:.4f}")
    print(f"  #10 score:  {results[9]['score']:.4f}")
    print(f"  #50 score:  {results[49]['score']:.4f}")
    print(f"  #100 score: {results[99]['score']:.4f}")
    top10_gap = results[0]["score"] - results[9]["score"]
    print(f"  Top-10 gap: {top10_gap:.4f}  ({'✅ > 0.03' if top10_gap > 0.03 else '⚠️ < 0.03 — check spread'})")

    print(f"\nNext: python validate_submission.py {args.out}")
    print(f"      python audit.py --submission {args.out} --top 50")


if __name__ == "__main__":
    main()
