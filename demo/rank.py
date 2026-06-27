#!/usr/bin/env python3
"""
rank.py  —  Redrob AI Candidate Ranker  (v7 Optimized for Demo)
Scores all candidates using pre-computed artifacts located in demo/artifacts_sample.
"""
import argparse, csv, json, pickle, sys
from datetime import datetime, date
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

# ── Artifact Paths (Locked to artifacts_sample for Demo) ────────────────────────
ARTIFACTS = Path(__file__).parent / "artifacts_sample"

# ── Load pre-computed artifacts (at module load, not per-candidate) ────────────
print(f"Loading pre-computed demo artifacts from {ARTIFACTS} ...")
jd_emb        = np.load(ARTIFACTS / "jd_emb.npy")
jd_skill_embs = np.load(ARTIFACTS / "jd_skill_embs.npy")
summary_embs  = np.load(ARTIFACTS / "summary_embs.npy")
career_embs   = np.load(ARTIFACTS / "career_embs.npy")
cid_order     = np.load(ARTIFACTS / "cid_order.npy", allow_pickle=True).tolist()

with open(ARTIFACTS / "skill_embed_map.pkl", "rb") as f:
    skill_embed_map = pickle.load(f)

with open(ARTIFACTS / "normalization_constants.json") as f:
    CONSTS = json.load(f)

# Constants
SEM_P1       = CONSTS["semantic_p1"]
SEM_P99      = CONSTS["semantic_p99"]
SKILL_DENOM  = CONSTS["skill_denom"]
PROD_P99     = CONSTS["production_p99"]
PROD_RESCALE = CONSTS["production_rescale"]

# Build O(1) lookup
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

EVAL_SIGNALS = [
    "ndcg", "mrr", "map", "offline benchmark", "online experiment",
    "ab test", "a/b test", "evaluation framework", "ranking quality",
    "precision at k", "recall at k", "recruiter feedback"
]

RETRIEVAL_SIGNALS = [
    "bm25", "dense retrieval", "hybrid retrieval", "learning-to-rank",
    "learning to rank", "reranking", "re-ranking", "faiss", "hnsw",
    "ann search", "embedding model", "retrieval quality"
]

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

COMPANY_SCORES = {
    "Pied Piper": 1.0, "Hooli": 1.0,
    "Wayne Enterprises": 0.85, "Stark Industries": 0.85,
    "Acme Corp": 0.75, "Globex Inc": 0.75,
    "Initech": 0.70, "Dunder Mifflin": 0.65,
    "Swiggy": 0.95, "Razorpay": 0.95, "CRED": 0.95,
    "Zomato": 0.90, "Paytm": 0.88, "PhonePe": 0.90,
    "Flipkart": 0.90, "Meesho": 0.88, "Ola": 0.85,
    "Byju's": 0.80, "Unacademy": 0.78, "Nykaa": 0.80,
    "Dunzo": 0.78, "Slice": 0.80, "Zepto": 0.82,
    "Google": 0.98, "Meta": 0.98,
    "Amazon": 0.95, "Microsoft": 0.95, "Uber": 0.95, "LinkedIn": 0.95, "Airbnb": 0.95,
    "NVIDIA": 0.95, "Databricks": 0.95, "Snowflake": 0.95, "Stripe": 0.95,
    "Atlassian": 0.95, "Salesforce": 0.95, "Adobe": 0.95, "Netflix": 0.95,
    "TCS": 0.50, "Infosys": 0.50, "Wipro": 0.50,
    "Accenture": 0.50, "Cognizant": 0.50, "Capgemini": 0.50,
    "HCL": 0.50, "Tech Mahindra": 0.52, "Mphasis": 0.52,
    "Genpact": 0.55, "Genpact AI": 0.55,
}
COMPANY_DEFAULT = 0.65

SERVICE_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mphasis", "Genpact", "Genpact AI",
    "IBM", "Hexaware", "Mindtree", "L&T Infotech", "Persistent Systems",
}

PRODUCT_COMPANIES = {
    c for c in COMPANY_SCORES if c not in SERVICE_COMPANIES and COMPANY_SCORES[c] >= 0.70
}

DOMAIN_SIGNALS = [
    "candidate corpus", "candidate search", "candidate-jd matching",
    "recruiter engagement", "time-to-shortlist", "recruiter-facing"
]

HYBRID_SIGNALS = [
    "hybrid retrieval", "bm25", "dense retrieval", "reranking", "llm-based re-ranker"
]

OPS_SIGNALS = [
    "embedding drift", "index refresh", "rollback", "versioning", "retrieval quality regression"
]

OWNERSHIP_SEARCH = [
    "ranking layer", "designed ranker", "owned ranking",
    "migration from keyword", "hybrid retrieval architecture", "feedback loop"
]

BORING_SIGNALS = [
    "rollback", "dashboard", "versioning", "latency budget", "monitoring",
    "index refresh", "embedding drift"
]

HIGH_RELEVANCE_INDUSTRIES = {
    "AI/ML", "Conversational AI", "HealthTech AI", "AI Services",
    "AdTech", "Fintech", "Food Delivery", "E-commerce",
    "EdTech", "SaaS", "Insurance Tech", "HealthTech",
}

LOCATION_SCORES = {
    "india": 1.0,
    "usa": 0.60, "united states": 0.60,
    "canada": 0.35,
    "uk": 0.40,
    "australia": 0.55,
    "germany": 0.55,
    "singapore": 0.60,
    "uae": 0.65,
}

TITLE_LEVEL_MAP = {
    "intern": 0, "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3, "lead": 3,
    "staff": 4, "principal": 4, "architect": 4,
    "director": 5, "vp": 6, "head": 5,
}

def _compute_title_velocity_penalty(candidate: dict) -> float:
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

def compute_semantic_score(candidate_idx: int) -> float:
    raw = (0.35 * float(np.dot(jd_emb, summary_embs[candidate_idx])) +
           0.65 * float(np.dot(jd_emb, career_embs[candidate_idx])))
    rescaled = (raw - SEM_P1) / max(SEM_P99 - SEM_P1, 1e-6)
    return float(np.clip(rescaled, 0.0, 1.0))

def compute_skill_score(candidate: dict) -> float:
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

    has_ai_skills = any(
        kw in full for kw in ["embedding", "retrieval", "ranking", "nlp", "machine learning"]
    )
    temporal_penalty = 0.0
    if has_ai_skills and not _has_pre_llm_ai_experience(candidate):
        temporal_penalty = 0.05

    return max(0.0, min(base - res + gh_c - temporal_penalty, 1.0))

def compute_production_score(candidate: dict) -> float:
    raw = compute_production_raw(candidate)
    if PROD_RESCALE:
        return float(np.clip(raw / max(PROD_P99, 1e-6), 0.0, 1.0))
    return raw

def compute_product_ratio_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    product_months = 0
    service_months = 0
    for role in history:
        months = role.get("duration_months", 0)
        company = role.get("company", "")
        if company in SERVICE_COMPANIES:
            service_months += months
        elif company in PRODUCT_COMPANIES or COMPANY_SCORES.get(company, 0) >= 0.75:
            product_months += months
    total = product_months + service_months
    if total == 0:
        return 0.5
    return product_months / total

def compute_career_depth_score(candidate: dict) -> float:
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

    raw = (0.45 * yoe_score + 0.15 * product_ratio + 0.05 * company_score + 0.25 * stability + 0.10 * industry_bonus)
    return float(np.clip(raw, 0.0, 1.0))

def compute_title_score(candidate: dict) -> float:
    title = candidate["profile"]["current_title"]
    return TITLE_SCORES.get(title, TITLE_DEFAULT)

def compute_education_score(candidate: dict) -> float:
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
    country = candidate["profile"].get("country", "").lower()
    return LOCATION_SCORES.get(country, 0.40)

def compute_ownership_score(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for kw in OWNERSHIP_SEARCH if kw in full)
    return min(hits / 5.0, 1.0)

def compute_eval_score(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for kw in EVAL_SIGNALS if kw in full)
    return min(hits / 6.0, 1.0)

def compute_retrieval_score(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for kw in RETRIEVAL_SIGNALS if kw in full)
    return min(hits / 6.0, 1.0)

def compute_availability_score(sig: dict) -> float:
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
    rr  = sig["recruiter_response_rate"]
    icr = sig["interview_completion_rate"]
    raw_oar = sig["offer_acceptance_rate"]
    oar = 0.50 if raw_oar < 0 else raw_oar
    return float(np.clip(0.50 * rr + 0.30 * icr + 0.20 * oar, 0.0, 1.0))

def compute_momentum_score(sig: dict) -> float:
    saves  = min(sig["saved_by_recruiters_30d"] / 8.0, 1.0)
    apps   = min(sig["applications_submitted_30d"] / 5.0, 1.0)
    views  = min(sig["profile_views_received_30d"] / 20.0, 1.0)
    search = min(sig["search_appearance_30d"] / 100.0, 1.0)
    return float(0.30 * saves + 0.25 * apps + 0.25 * views + 0.20 * search)

def compute_github_score(sig: dict) -> float:
    gh = sig["github_activity_score"]
    if gh < 0:
        return 0.15
    return float(np.clip(gh / 80.0, 0.0, 1.0))

def compute_notice_score(sig: dict) -> float:
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

def compute_domain_bonus(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in DOMAIN_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.06

def compute_hybrid_bonus(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in HYBRID_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.04

def compute_ops_bonus(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in OPS_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.03

def compute_boring_bonus(candidate: dict) -> float:
    texts = [candidate["profile"].get("summary", "")]
    texts += [r.get("description", "") for r in candidate.get("career_history", [])]
    full = " ".join(texts).lower()
    hits = sum(1 for sig in BORING_SIGNALS if sig in full)
    return min(hits / 4, 1.0) * 0.04

def evaluate_integrity_rules(candidate: dict) -> tuple[bool, list[str], list[str]]:
    profile = candidate["profile"]
    history = candidate["career_history"]
    skills  = candidate["skills"]
    edu     = candidate.get("education", [])
    sig     = candidate["redrob_signals"]

    critical_triggers = []
    major_triggers = []
    minor_triggers = []

    total_months  = sum(r["duration_months"] for r in history if r["duration_months"] > 0)
    stated_months = profile["years_of_experience"] * 12
    
    if stated_months > total_months + 36:
        critical_triggers.append(
            f"Experience mismatch: claimed {profile['years_of_experience']}yr experience but only "
            f"{total_months//12:.0f}yr documented in career history"
        )
    elif total_months > stated_months + 36:
        critical_triggers.append(
            f"Experience mismatch: documented history ({total_months//12:.0f}yr) "
            f"exceeds stated experience ({profile['years_of_experience']}yr) by more than 3 years"
        )

    should_reject = len(critical_triggers) >= 1 or len(major_triggers) >= 2
    reject_reasons = critical_triggers + major_triggers
    
    return should_reject, reject_reasons, minor_triggers

def compute_consistency(candidate: dict) -> float:
    rejected, _, _ = evaluate_integrity_rules(candidate)
    return 0.0 if rejected else 1.0

def generate_reasoning(candidate: dict, scores: dict) -> str:
    profile  = candidate["profile"]
    history  = candidate["career_history"]
    skills   = candidate["skills"]
    sig       = candidate["redrob_signals"]

    title = profile.get("current_title", "Unknown")
    yoe   = profile["years_of_experience"]

    search_text = (
        profile.get("summary", "") + " "
        + " ".join(r.get("description", "") for r in history)
    ).lower()
    skill_names_lower = " ".join(s["name"].lower() for s in skills)
    full_text = search_text + " " + skill_names_lower

    eval_score       = scores.get("evaluation", 0.0)
    ownership_score  = scores.get("ownership", 0.0)
    production_score = scores.get("production", 0.0)

    reasons = [f"{title} with {yoe} YoE"]
    if production_score > 0.5:
         reasons.append("Strong production evidence")
    return " | ".join(reasons)

def score_candidate(candidate: dict) -> tuple[float, str]:
    cid = candidate.get("candidate_id")
    if cid not in CID_TO_IDX:
        return 0.0, f"FILTERED: candidate_id {cid} not found in embedding artifacts"

    idx = CID_TO_IDX[cid]
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
    location        = compute_location_score(candidate)

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

    final = round(0.75 * base + 0.25 * behavioral + domain_bonus + hybrid_bonus + ops_bonus + boring_bonus, 4)
    final = min(final, 1.0)
    
    scores = {
         "evaluation": evaluation,
         "ownership": ownership,
         "production": production,
         "semantic": semantic
    }
    
    reason = generate_reasoning(candidate, scores)
    return final, reason

if __name__ == "__main__":
    pass
