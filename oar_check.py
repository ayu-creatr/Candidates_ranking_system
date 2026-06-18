import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, numpy as np
from datetime import date, datetime

path = r'c:\Users\hp\Desktop\hack2Skilll\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
REFERENCE_DATE = date(2026, 6, 17)

def availability(sig):
    last = datetime.strptime(sig['last_active_date'], '%Y-%m-%d').date()
    d = (REFERENCE_DATE - last).days
    rec = 1.0 if d<=14 else 0.90 if d<=30 else 0.75 if d<=60 else 0.55 if d<=90 else 0.30 if d<=180 else 0.10
    return 0.60*rec + 0.40*(1.0 if sig['open_to_work_flag'] else 0.50)

def engagement_old(sig):
    return 0.50*sig['recruiter_response_rate'] + 0.30*sig['interview_completion_rate'] + 0.20*max(sig['offer_acceptance_rate'], 0.0)

def engagement_new(sig):
    oar = 0.50 if sig['offer_acceptance_rate'] < 0 else sig['offer_acceptance_rate']
    return 0.50*sig['recruiter_response_rate'] + 0.30*sig['interview_completion_rate'] + 0.20*oar

def behavioral(sig, eng_fn):
    av  = availability(sig)
    eng = eng_fn(sig)
    gh  = sig['github_activity_score']
    ghs = 0.15 if gh == -1 else min(gh/80, 1.0)
    n   = sig['notice_period_days']
    ns  = 1.0 if n<=30 else 0.75 if n<=60 else 0.50 if n<=90 else 0.30 if n<=120 else 0.15
    saves = min(sig['saved_by_recruiters_30d']/8,   1.0)
    apps  = min(sig['applications_submitted_30d']/5, 1.0)
    views = min(sig['profile_views_received_30d']/20,1.0)
    srch  = min(sig['search_appearance_30d']/100,    1.0)
    mom = 0.30*saves + 0.25*apps + 0.25*views + 0.20*srch
    return 0.30*av + 0.25*eng + 0.20*mom + 0.15*ghs + 0.10*ns

bo, bn, cids = [], [], []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c   = json.loads(line)
        sig = c['redrob_signals']
        bo.append(behavioral(sig, engagement_old))
        bn.append(behavioral(sig, engagement_new))
        cids.append(c['candidate_id'])

bo   = np.array(bo)
bn   = np.array(bn)
cids = np.array(cids)

top20_old = set(cids[np.argsort(bo)[::-1][:20]])
top20_new = set(cids[np.argsort(bn)[::-1][:20]])
overlap   = len(top20_old & top20_new)

print("=== oar=-1 FIX IMPACT ON BEHAVIOURAL SCORES ===")
print(f"Old mean behavioral : {bo.mean():.4f}")
print(f"New mean behavioral : {bn.mean():.4f}")
print(f"Mean shift          : +{(bn-bo).mean():.4f}")
print(f"Top-20 overlap      : {overlap}/20")

if overlap >= 18:
    verdict = "SAFE to apply (>=90% overlap)"
elif overlap >= 15:
    verdict = "Mostly safe - check 5 movers manually"
else:
    verdict = "CAUTION - significant reshuffling"
print(f"Verdict             : {verdict}")

moved_out = top20_old - top20_new
moved_in  = top20_new - top20_old

if moved_out:
    print("\nMoved OUT of top-20:")
    for cid in moved_out:
        i = np.where(cids == cid)[0][0]
        print(f"  {cid}  old={bo[i]:.4f}  new={bn[i]:.4f}")

if moved_in:
    print("\nMoved INTO top-20:")
    for cid in moved_in:
        i = np.where(cids == cid)[0][0]
        print(f"  {cid}  old={bo[i]:.4f}  new={bn[i]:.4f}")
