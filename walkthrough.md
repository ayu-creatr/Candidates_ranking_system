# Walkthrough — Redrob AI Candidate Ranker (v7)

We have successfully implemented the Version 7 updates for the candidate ranker, addressing your prioritized issues and aligning the submission formatting to comply with the official validation suite.

## Summary of Changes

### 1. Fix Country Extraction (Priority 0)
- Changed the country extraction logic in `rank.py` and `audit.py` to pull directly from the root field `profile.country` instead of the non-existent nested path `current_location.country`.
- **Result:** Location scores now correctly calculate for the entire candidate pool (e.g. mean location score of `0.8958` in the sample check instead of a flat `0.4000` default).

### 2. Ownership Score (Priority 1)
- Implemented `compute_ownership_score(candidate)` using signals: `owned`, `owner`, `end to end`, `e2e`, `architected`, `designed and built`, `built from scratch`, `responsible for`, `led implementation`, `technical lead`, `drove migration`.
- Assigned a **6%** weight to this component in the base score.
- Implemented non-linear mapping: `0 hits -> 0.0`, `1 hit -> 0.5`, `2 hits -> 0.8`, `3+ hits -> 1.0`.

### 3. Evaluation Framework Score (Priority 2)
- Implemented `compute_eval_score(candidate)` using signals: `ndcg`, `mrr`, `map`, `offline benchmark`, `online experiment`, `ab test`, `a/b test`, `evaluation framework`, `ranking quality`, `precision at k`, `recall at k`, `recruiter feedback`.
- Assigned a **3%** weight to this component in the base score.
- Implemented non-linear mapping: `0 hits -> 0.0`, `1 hit -> 0.5`, `2 hits -> 0.8`, `3+ hits -> 1.0`.

### 4. Company Dictionary Expansion (Priority 3 & 7)
- Added premium and product companies to `COMPANY_SCORES`:
  - `Google: 0.98`, `Meta: 0.98`
  - `Amazon: 0.95`, `Microsoft: 0.95`, `Uber: 0.95`, `LinkedIn: 0.95`, `Airbnb: 0.95`
  - `NVIDIA: 0.95`, `Databricks: 0.95`, `Snowflake: 0.95`, `Stripe: 0.95`, `Atlassian: 0.95`, `Salesforce: 0.95`, `Adobe: 0.95`, `Netflix: 0.95`
- Prevents top engineering talent from defaulting to a neutral `0.65` score.

### 5. Better Reasoning Generator (Priority 4)
- Updated `generate_reasoning` to scan candidate text and skills for matching elite JD keywords.
- Appends `| JD keywords: <matched terms>` to the reasoning string if found (e.g. `BM25`, `NDCG`, `Hybrid retrieval`).

### 6. Production-Aware Research Penalty (Priority 5)
- Added a conditional post-processing check: if the candidate has `"research"` in their current title AND their production score is `< 0.40`, a `0.95` multiplier is applied to their final score.
- Appends `| NOTE: Research title with low production evidence penalty applied` to the reasoning string when triggered.

### 7. Rebalanced Weights (Priority 6)
- Rebalanced base score weights:
  - Semantic: `24%`
  - Skill: `20%`
  - Production: `20%`
  - Career: `15%`
  - Title: `10%`
  - Ownership: `6%`
  - Eval: `3%`
  - Education: `2%`
  - Location: `2%`
- Bounded the base score in `[0, 1]` by dividing the weighted sum by `1.02`.

### 8. Official Validator Alignment & Formatting
- Standardized the submission output in `rank.py`:
  - Selects only the **top 100** candidates.
  - Sorts descending by final score, and resolves score ties deterministically by `candidate_id` ascending.
  - Writes the required header: `candidate_id,rank,score,reasoning` (with `rank` from 1 to 100).
- Fixed terminal/CLI encoding crashes on Windows by setting `sys.stdout` encoding to `utf-8`.
- Fixed `IndexError: index 999 is out of bounds` in `audit.py` when auditing the 100-row file by making score prints conditional on submission length.

---

## Verification & Validation Results

### 1. Sample Audit Check (`python audit.py --candidates ... --n 1000`)
- **Status:** All component scores are verified within the bounds `[0, 1]`.
- **Score Spread:** The top-10 gap is `0.2042` (which is `✅ > 0.03`).

### 2. Official Validator Check (`python validate_submission.py submission.csv`)
- **Result:** `Submission is valid.`
- **Rows:** Exactly 100 data rows.
- **Ranks:** 1 to 100 appeared exactly once.
- **Order:** Scores are verified non-increasing.
