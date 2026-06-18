"""
Diagnostic script — validates every concern raised in the v3 critique.
Run: python diagnose.py
"""
import json, collections
from datetime import date, datetime

path = r'c:\Users\hp\Desktop\hack2Skilll\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'

COMPANY_SCORES_DICT = {
    "Pied Piper": 1.0, "Hooli": 1.0, "Swiggy": 0.95, "Razorpay": 0.95, "CRED": 0.95,
    "Wayne Enterprises": 0.80, "Stark Industries": 0.80,
    "Acme Corp": 0.70, "Globex Inc": 0.70, "Initech": 0.65, "Dunder Mifflin": 0.60,
    "TCS": 0.50, "Infosys": 0.50, "Wipro": 0.50,
    "Accenture": 0.50, "Cognizant": 0.50, "Capgemini": 0.50
}

REFERENCE_DATE = date(2026, 6, 17)

# --- Counters ---
total = 0
consistency_filtered = 0     # Issue 2: how many hit < 0.40 threshold?
company_default_hits = 0     # Issue 4: how many career roles hit default 0.65?
company_total_roles  = 0
industry_values      = collections.Counter()
unknown_titles       = collections.Counter()
reasoning_empty_skills = 0   # Issue 3: silent failure in reasoning
offer_negative = 0           # Minor: offer_acceptance_rate = -1
interview_negative = 0       # Minor: interview_completion_rate < 0

KNOWN_TITLE_SCORES = {
    "ML Engineer","AI Engineer","Senior Machine Learning Engineer","NLP Engineer",
    "Research Scientist","Data Scientist","Backend Engineer","Analytics Engineer",
    "Data Engineer","Software Engineer","Full Stack Developer","Cloud Engineer",
    "DevOps Engineer","Data Analyst","Business Analyst","Project Manager",
    "Operations Manager","Marketing Manager","HR Manager","Content Writer",
    "Graphic Designer","Civil Engineer","Mechanical Engineer","Accountant",
    "Sales Executive","Customer Support","Junior ML Engineer"
}

AI_KEYWORDS = [
    "nlp","python","embed","retriev","vector","llm","rag","faiss","milvus",
    "lora","transformer","rank","ml","machine learning","search","pytorch","tensorflow"
]

REASONING_SKILL_KEYWORDS = [
    "nlp","python","embed","retriev","vector","llm","rag","faiss","milvus",
    "lora","transformer","rank","machine learning","search","pytorch","tensorflow",
    "sentence","information retrieval","dense","ann","recommend","scikit","hugging"
]

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        total += 1

        profile  = c['profile']
        history  = c['career_history']
        skills   = c['skills']
        sig      = c['redrob_signals']

        # === Issue 2: Consistency threshold check ===
        total_months = sum(r['duration_months'] for r in history)
        stated_months = profile['years_of_experience'] * 12
        expert_zero = [s for s in skills
                       if s['proficiency'] in ('expert','advanced') and s['duration_months'] == 0]
        inflated = [s for s in skills
                    if s['proficiency'] == 'expert'
                    and s['endorsements'] == 0
                    and sig['skill_assessment_scores'].get(s['name'], 50) < 40]
        completeness = sig['profile_completeness_score']

        cs = 1.0
        if total_months > stated_months + 36: cs -= 0.50
        elif total_months > stated_months + 18: cs -= 0.25
        if len(expert_zero) >= 3: cs -= 0.35
        elif len(expert_zero) >= 1: cs -= 0.10
        if len(inflated) >= 4: cs -= 0.25
        if completeness < 20 and profile['years_of_experience'] > 10: cs -= 0.15
        cs = max(0.0, cs)
        if cs < 0.40:
            consistency_filtered += 1

        # === Issue 4: Company default hit rate ===
        for role in history:
            company_total_roles += 1
            if role['company'] not in COMPANY_SCORES_DICT:
                company_default_hits += 1

        # === Industry values ===
        for role in history:
            industry_values[role['industry']] += 1

        # === Issue 3: Reasoning silent failure ===
        matched_ai_skills = [
            s['name'] for s in skills
            if any(kw in s['name'].lower() for kw in REASONING_SKILL_KEYWORDS)
        ]
        if not matched_ai_skills:
            reasoning_empty_skills += 1

        # === Minor: offer/interview ranges ===
        if sig['offer_acceptance_rate'] < 0:
            offer_negative += 1
        if sig['interview_completion_rate'] < 0:
            interview_negative += 1

        # === Title coverage ===
        t = profile['current_title']
        if t not in KNOWN_TITLE_SCORES:
            unknown_titles[t] += 1

# ===== REPORT =====
print(f"\nTotal candidates: {total}")

print(f"\n=== ISSUE 2: Consistency Threshold ===")
print(f"  Filtered (score < 0.40): {consistency_filtered} ({consistency_filtered/total*100:.2f}%)")
print(f"  At 2-3% safe zone: {'OK' if consistency_filtered/total <= 0.03 else 'TOO AGGRESSIVE'}")

print(f"\n=== ISSUE 4: Company Default Hit Rate ===")
print(f"  Total career roles processed: {company_total_roles}")
print(f"  Roles hitting DEFAULT (unknown company): {company_default_hits}")
print(f"  Default rate: {company_default_hits/company_total_roles*100:.1f}%")
print(f"  Verdict: {'OK - most companies are known' if company_default_hits/company_total_roles < 0.50 else 'PROBLEM - >50% hitting default'}")

print(f"\n=== ISSUE 3: Reasoning Skill Silent Failure ===")
print(f"  Candidates with ZERO matching AI skills in reasoning: {reasoning_empty_skills}")
print(f"  ({reasoning_empty_skills/total*100:.1f}% would show empty skills line)")

print(f"\n=== ACTUAL INDUSTRY VALUES (top 20) ===")
for ind, n in industry_values.most_common(20):
    print(f"  '{ind}': {n}")

print(f"\n=== TITLE COVERAGE: Unknown Titles (top 20) ===")
for t, n in unknown_titles.most_common(20):
    print(f"  '{t}': {n}")

print(f"\n=== MINOR CHECKS ===")
print(f"  offer_acceptance_rate < 0 (expected): {offer_negative} ({offer_negative/total*100:.1f}%)")
print(f"  interview_completion_rate < 0 (should be 0): {interview_negative}")
