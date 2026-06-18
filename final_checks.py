"""
Final pre-plan validation checks:
1. Consistency score distribution (is there meaningful variance?)
2. oar=-1 neutralization: does it shift top-20?
Run: python final_checks.py
"""
import json, numpy as np
from datetime import date, datetime

path = r'c:\Users\hp\Desktop\hack2Skilll\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
REFERENCE_DATE = date(2026, 6, 17)

# ── Consistency score function ──────────────────────────────────────────────
def compute_consistency(c):
    profile, history, skills, sig = (
        c['profile'], c['career_history'], c['skills'], c['redrob_signals'])
    score = 1.0
    total_months  = sum(r['duration_months'] for r in history)
    stated_months = profile['years_of_experience'] * 12
    if total_months > stated_months + 36: score -= 0.50
    elif total_months > stated_months + 18: score -= 0.25
    expert_zero = [s for s in skills
                   if s['proficiency'] in ('expert','advanced') and s['duration_months']==0]
    if len(expert_zero) >= 3: score -= 0.35
    elif len(expert_zero) >= 1: score -= 0.10
    inflated = [s for s in skills
                if s['proficiency']=='expert' and s['endorsements']==0
                and sig['skill_assessment_scores'].get(s['name'],50) < 40]
    if len(inflated) >= 4: score -= 0.25
    if sig['profile_completeness_score'] < 20 and profile['years_of_experience'] > 10:
        score -= 0.15
    return max(0.0, score)

# ── Engagement score (old: oar=max(raw,0))  vs  (new: oar=-1→0.5) ──────────
def compute_engagement_old(sig):
    rr  = sig['recruiter_response_rate']
    icr = sig['interview_completion_rate']
    oar = max(sig['offer_acceptance_rate'], 0.0)
    return 0.50*rr + 0.30*icr + 0.20*oar

def compute_engagement_new(sig):
    rr  = sig['recruiter_response_rate']
    icr = sig['interview_completion_rate']
    raw = sig['offer_acceptance_rate']
    oar = 0.50 if raw < 0 else raw   # neutral for no history
    return 0.50*rr + 0.30*icr + 0.20*oar

# ── Quick behavioral score for top-20 test ──────────────────────────────────
def availability(sig):
    last = datetime.strptime(sig['last_active_date'], '%Y-%m-%d').date()
    d    = (REFERENCE_DATE - last).days
    rec  = 1.0 if d<=14 else 0.90 if d<=30 else 0.75 if d<=60 else \
           0.55 if d<=90 else 0.30 if d<=180 else 0.10
    otw  = 1.0 if sig['open_to_work_flag'] else 0.50
    return 0.60*rec + 0.40*otw

def behavioral_old(sig):
    av  = availability(sig)
    eng = compute_engagement_old(sig)
    gh  = sig['github_activity_score']
    ghs = 0.15 if gh==-1 else min(gh/80,1.0)
    n   = sig['notice_period_days']
    ns  = 1.0 if n<=30 else 0.75 if n<=60 else 0.50 if n<=90 else 0.30 if n<=120 else 0.15
    apps   = min(sig['applications_submitted_30d']/5,1.0)
    views  = min(sig['profile_views_received_30d']/20,1.0)
    saves  = min(sig['saved_by_recruiters_30d']/8,1.0)
    search = min(sig['search_appearance_30d']/100,1.0)
    mom    = 0.30*saves+0.25*apps+0.25*views+0.20*search
    return 0.30*av+0.25*eng+0.20*mom+0.15*ghs+0.10*ns

def behavioral_new(sig):
    av  = availability(sig)
    eng = compute_engagement_new(sig)
    gh  = sig['github_activity_score']
    ghs = 0.15 if gh==-1 else min(gh/80,1.0)
    n   = sig['notice_period_days']
    ns  = 1.0 if n<=30 else 0.75 if n<=60 else 0.50 if n<=90 else 0.30 if n<=120 else 0.15
    apps   = min(sig['applications_submitted_30d']/5,1.0)
    views  = min(sig['profile_views_received_30d']/20,1.0)
    saves  = min(sig['saved_by_recruiters_30d']/8,1.0)
    search = min(sig['search_appearance_30d']/100,1.0)
    mom    = 0.30*saves+0.25*apps+0.25*views+0.20*search
    return 0.30*av+0.25*eng+0.20*mom+0.15*ghs+0.10*ns

# ── Main ─────────────────────────────────────────────────────────────────────
print("Loading 100K candidates …")
consistency_scores = []
beh_old_list, beh_new_list = [], []
cids = []

with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        c   = json.loads(line)
        sig = c['redrob_signals']
        consistency_scores.append(compute_consistency(c))
        beh_old_list.append(behavioral_old(sig))
        beh_new_list.append(behavioral_new(sig))
        cids.append(c['candidate_id'])

cs = np.array(consistency_scores)
print(f"\n=== CONSISTENCY SCORE DISTRIBUTION ===")
print(f"  Min:    {cs.min():.3f}")
print(f"  P1:     {np.percentile(cs,1):.3f}")
print(f"  P5:     {np.percentile(cs,5):.3f}")
print(f"  P10:    {np.percentile(cs,10):.3f}")
print(f"  P25:    {np.percentile(cs,25):.3f}")
print(f"  Median: {np.median(cs):.3f}")
print(f"  Mean:   {cs.mean():.3f}")
print(f"  Max:    {cs.max():.3f}")
print(f"  Std:    {cs.std():.3f}")
print(f"\n  Candidates below 0.60: {(cs<0.60).sum()} ({(cs<0.60).mean()*100:.2f}%)")
print(f"  Candidates below 0.70: {(cs<0.70).sum()} ({(cs<0.70).mean()*100:.2f}%)")
print(f"  Candidates at 1.00:    {(cs==1.00).sum()} ({(cs==1.00).mean()*100:.1f}%)")
print(f"  Variance: {cs.var():.6f}")
print(f"  → Signal strength: {'USEFUL (good spread)' if cs.std() > 0.05 else 'WEAK (near-flat)'}")

bo = np.array(beh_old_list)
bn = np.array(beh_new_list)

# Top-20 by old behavioural score
top20_old_idx = np.argsort(bo)[::-1][:20]
top20_new_idx = np.argsort(bn)[::-1][:20]
top20_old = set(cids[i] for i in top20_old_idx)
top20_new = set(cids[i] for i in top20_new_idx)
overlap   = len(top20_old & top20_new)

print(f"\n=== oar=-1 FIX IMPACT ON BEHAVIOURAL TOP-20 ===")
print(f"  Old mean engagement: {bo.mean():.4f}")
print(f"  New mean engagement: {bn.mean():.4f}")
print(f"  Mean shift: +{(bn-bo).mean():.4f}")
print(f"  Top-20 overlap (old vs new): {overlap}/20")
if overlap >= 18:
    print(f"  → Safe to apply fix (≥90% overlap)")
elif overlap >= 15:
    print(f"  → Mostly safe — check 5 movers manually")
else:
    print(f"  → Caution — significant reshuffling, review before committing")

print(f"\n  Who moved OUT of top-20 after fix:")
for cid in top20_old - top20_new:
    i   = cids.index(cid)
    sig_raw = None  # we only have scores, not raw data here
    print(f"    {cid}  old_beh={bo[i]:.4f}  new_beh={bn[i]:.4f}")

print(f"\n  Who moved IN to top-20 after fix:")
for cid in top20_new - top20_old:
    i = cids.index(cid)
    print(f"    {cid}  old_beh={bo[i]:.4f}  new_beh={bn[i]:.4f}")
