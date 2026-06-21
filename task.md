# Implementation Task — Redrob AI Candidate Ranker

## Phase 0 — Environment Setup
- [x] Audit existing files (diagnose.py, final_checks.py — diagnostic only, no ranker)
- [x] Check installed packages
- [x] Fix numpy version conflict (numpy 2.4.3 breaks sklearn/scipy → downgrade to 1.26.4)
- [x] Install sentence-transformers==2.7.0
- [x] Install scikit-learn==1.4.2
- [ ] Verify all imports work

## Phase 1 — precompute.py
- [x] Write precompute.py (full spec from plan)
- [ ] Run: `python precompute.py --candidates candidates.jsonl`
- [ ] Confirm artifacts/ folder is populated
- [ ] Read printed ranges: semantic P1/P99, skill P95, production P99

## Phase 2 — rank.py (v7 Updates)
- [x] Update country extraction to use `profile.country` (P0)
- [x] Define `OWNERSHIP_SIGNALS` and implement `compute_ownership_score` (P1)
- [x] Define `EVAL_SIGNALS` and implement `compute_eval_score` (P2)
- [x] Expand `COMPANY_SCORES` dictionary (P3 & P7)
- [x] Enrich reasoning strings with matched JD keywords (P4)
- [x] Implement research title low-production penalty (P5)
- [x] Rebalance weights and scale base score by 1.02 (P6)
- [x] Modify CSV writing to output exactly top-100 candidates with rank, score, reasoning, sorted descending with tie-breakers

## Phase 3 — audit.py (v7 Updates)
- [x] Update country details extraction to use `profile.country`
- [x] Update sample audit to track and log stats for `ownership` and `eval` scores

## Phase 4 — Full Run + Submission (v7 Verification)
- [x] Run `python audit.py --candidates candidates.jsonl --n 1000`
- [x] Run `python rank.py --candidates candidates.jsonl --out submission.csv`
- [x] Run `python validate_submission.py submission.csv`
- [x] Run audit on full output top-50 (Fixed CP1252 character map CP1252 error and IndexError)
- [ ] Verify score gaps and inspect top candidates manually
