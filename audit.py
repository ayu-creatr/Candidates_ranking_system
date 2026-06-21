#!/usr/bin/env python3
"""
audit.py  —  Redrob AI Candidate Ranker  (v6)

Two modes:
  1. Sample audit (Phase 2a): checks component ranges on a random sample
     python audit.py --n 1000

  2. Submission audit (Phase 4): inspects top-N of a full submission CSV
     python audit.py --submission submission.csv --top 50
"""
import argparse, csv, json, random, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

# ─── Imports from rank.py ─────────────────────────────────────────────────────
# We import rank.py's scoring functions so we use exactly the same logic
# No duplication — this is intentional: audit validates rank.py's output
try:
    import rank as R
except ImportError:
    print("ERROR: audit.py must be run from the same directory as rank.py and artifacts/")
    sys.exit(1)


def audit_sample(candidates_path: Path, n: int):
    """Phase 2a — sample N candidates, check all component scores are [0, 1]."""
    print(f"\n{'='*60}")
    print(f"AUDIT — Sample mode (n={n})")
    print(f"{'='*60}")

    # Load all candidates, take a random sample
    all_candidates = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_candidates.append(json.loads(line))

    print(f"Loaded {len(all_candidates):,} candidates, sampling {n} ...")
    sample = random.sample(all_candidates, min(n, len(all_candidates)))

    # Per-component score tracking
    scores = {
        "semantic": [], "skill": [], "production": [],
        "career": [], "title": [], "ownership": [], "eval": [],
        "education": [], "location": [], "behavioral": [],
        "consistency": [], "final": [],
    }
    out_of_range = []
    filtered_count = 0

    for c in sample:
        cid = c["candidate_id"]
        idx = R.CID_TO_IDX.get(cid)

        cs = R.compute_consistency(c)
        scores["consistency"].append(cs)

        if cs < 0.70:
            filtered_count += 1
            scores["final"].append(0.0)
            for k in ["semantic", "skill", "production", "career", "title", "ownership", "eval", "education", "location", "behavioral"]:
                scores[k].append(None)
            continue

        if idx is None:
            filtered_count += 1
            scores["final"].append(0.0)
            continue

        sem  = R.compute_semantic_score(idx)
        sk   = R.compute_skill_score(c)
        prod = R.compute_production_score(c)
        car  = R.compute_career_depth_score(c)
        tvp  = R._compute_title_velocity_penalty(c)
        car_adj = max(0.0, car - tvp)
        tit  = R.compute_title_score(c)
        own  = R.compute_ownership_score(c)
        ev   = R.compute_eval_score(c)
        edu  = R.compute_education_score(c)
        loc  = R.compute_location_score(c)
        beh  = R.compute_behavioral_score(c)

        final, _ = R.score_candidate(c)

        scores["semantic"].append(sem)
        scores["skill"].append(sk)
        scores["production"].append(prod)
        scores["career"].append(car_adj)
        scores["title"].append(tit)
        scores["ownership"].append(own)
        scores["eval"].append(ev)
        scores["education"].append(edu)
        scores["location"].append(loc)
        scores["behavioral"].append(beh)
        scores["final"].append(final)

        # Check for out-of-range
        for name, val in [("semantic", sem), ("skill", sk), ("production", prod),
                           ("career", car_adj), ("title", tit), ("ownership", own),
                           ("eval", ev), ("education", edu), ("location", loc),
                           ("behavioral", beh), ("final", final)]:
            if not (0.0 <= val <= 1.0):
                out_of_range.append(f"  {cid}: {name}={val:.4f} OUT OF RANGE")

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'Component':<15} {'Mean':>8} {'Min':>8} {'Max':>8} {'P25':>8} {'P75':>8} {'Status'}")
    print("-" * 70)

    all_ok = True
    for name in ["semantic", "skill", "production", "career", "title", "ownership", "eval", "education", "location", "behavioral", "final"]:
        vals = [v for v in scores[name] if v is not None]
        if not vals:
            print(f"{name:<15} {'N/A':>8}")
            continue
        arr = np.array(vals)
        ok  = (arr.min() >= 0.0 and arr.max() <= 1.0)
        if not ok:
            all_ok = False
        status = "✅" if ok else "❌ OUT OF RANGE"
        print(f"{name:<15} {arr.mean():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} "
              f"{np.percentile(arr,25):>8.4f} {np.percentile(arr,75):>8.4f}  {status}")

    print(f"\nFiltered by hard gates: {filtered_count}/{n} ({filtered_count/n*100:.1f}%)")

    if out_of_range:
        print(f"\n❌ OUT-OF-RANGE DETECTED:")
        for msg in out_of_range:
            print(msg)
    else:
        print(f"\n✅ All components in [0, 1] — normalization OK")

    # ── Score distribution check ──────────────────────────────────────────────
    final_vals = np.array([v for v in scores["final"] if v > 0.0])
    if len(final_vals) > 0:
        final_vals_sorted = np.sort(final_vals)[::-1]
        print(f"\nScore distribution (non-zero scores in sample):")
        print(f"  N non-zero:  {len(final_vals_sorted)}")
        print(f"  #1:          {final_vals_sorted[0]:.4f}")
        if len(final_vals_sorted) >= 10:
            top10_gap = final_vals_sorted[0] - final_vals_sorted[9]
            print(f"  #10:         {final_vals_sorted[9]:.4f}")
            print(f"  Top-10 gap:  {top10_gap:.4f}  "
                  f"({'✅ > 0.03' if top10_gap > 0.03 else '⚠️ < 0.03 — narrow spread'})")
        print(f"  P50:         {np.median(final_vals_sorted):.4f}")
        print(f"  P90:         {np.percentile(final_vals_sorted, 90):.4f}")

    print(f"\n{'='*60}")
    print("Phase 2a complete. If all ✅, proceed to:")
    print("  python rank.py --candidates candidates.jsonl --out submission.csv")


def audit_submission(submission_path: Path, candidates_path: Path, top_n: int):
    """Phase 4 — inspect top-N of a full submission CSV."""
    print(f"\n{'='*60}")
    print(f"AUDIT — Submission mode (top {top_n})")
    print(f"{'='*60}")

    # Load submission
    rows = []
    with open(submission_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "candidate_id": row["candidate_id"],
                "score":        float(row["score"]),
                "reasoning":    row.get("reasoning", ""),
            })

    rows.sort(key=lambda x: x["score"], reverse=True)
    print(f"Submission has {len(rows):,} rows.")

    # Score stats
    all_scores = np.array([r["score"] for r in rows])
    print(f"\nFull submission score distribution:")
    if len(all_scores) >= 1:
        print(f"  #1:    {all_scores[0]:.4f}")
    if len(all_scores) >= 10:
        print(f"  #10:   {all_scores[9]:.4f}")
    if len(all_scores) >= 100:
        print(f"  #100:  {all_scores[99]:.4f}")
    if len(all_scores) >= 1000:
        print(f"  #1000: {all_scores[999]:.4f}")
    top10_gap = (all_scores[0] - all_scores[9]) if len(all_scores) >= 10 else 0.0
    print(f"  Top-10 gap: {top10_gap:.4f}  "
          f"({'✅' if top10_gap > 0.03 else '⚠️ narrow'}")

    zero_count = (all_scores == 0.0).sum()
    print(f"  Zero-score (filtered): {zero_count:,} ({zero_count/len(all_scores)*100:.1f}%)")

    # Show top-N
    print(f"\n{'─'*100}")
    print(f"TOP {top_n} CANDIDATES:")
    print(f"{'─'*100}")

    top_rows = rows[:top_n]

    # Try to look up candidate details if candidates.jsonl is available
    cid_details = {}
    if candidates_path and candidates_path.exists():
        top_ids = {r["candidate_id"] for r in top_rows}
        with open(candidates_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                if c["candidate_id"] in top_ids:
                    p = c["profile"]
                    sig = c["redrob_signals"]
                    cid_details[c["candidate_id"]] = {
                        "title":  p.get("current_title", "?"),
                        "yoe":    p.get("years_of_experience", 0),
                        "country": p.get("country", "?"),
                        "company": c["career_history"][0]["company"] if c["career_history"] else "?",
                        "github": sig.get("github_activity_score", -1),
                        "otw":    sig.get("open_to_work_flag", False),
                        "notice": sig.get("notice_period_days", 999),
                    }
                if len(cid_details) == len(top_ids):
                    break

    print(f"{'#':<4} {'Score':<8} {'Title':<35} {'YoE':>4} {'Country':<10} "
          f"{'Company':<22} {'GH':>4} {'OTW':<5} {'Notice':>6}")
    print("-" * 105)

    for rank, r in enumerate(top_rows, 1):
        cid  = r["candidate_id"]
        d    = cid_details.get(cid, {})
        title   = d.get("title", "?")[:33]
        yoe     = d.get("yoe", "?")
        country = d.get("country", "?")[:9]
        company = d.get("company", "?")[:20]
        gh      = d.get("github", -1)
        gh_str  = str(gh) if gh >= 0 else "N/A"
        otw     = "✓" if d.get("otw", False) else "✗"
        notice  = d.get("notice", 999)
        notice_str = f"{notice}d" if notice < 999 else "?"

        print(f"{rank:<4} {r['score']:<8.4f} {title:<35} {str(yoe):>4} {country:<10} "
              f"{company:<22} {gh_str:>4} {otw:<5} {notice_str:>6}")

    print(f"\n{'─'*100}")
    print("Manual review checklist for top-10:")
    print("  [ ] All top-10 are AI/ML engineers (not Business Analysts or HRs)")
    print("  [ ] At least 7/10 have India as country")
    print("  [ ] No obvious honeypots (perfect keyword-stuffed profiles)")
    print("  [ ] Reasoning strings are specific, not templated")
    print("  [ ] Score gap between #1 and #10 > 0.03")

    print(f"\nAudit complete. Submission: {submission_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit tool for Redrob AI Candidate Ranker.")
    parser.add_argument("--candidates", default=None,
                        help="Path to candidates.jsonl (required for --n mode; optional for --submission)")
    parser.add_argument("--n", type=int, default=0,
                        help="Sample N candidates for Phase 2a audit")
    parser.add_argument("--submission", default=None,
                        help="Path to submission CSV for Phase 4 audit")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top candidates to inspect in submission audit")
    args = parser.parse_args()

    if args.n > 0:
        if not args.candidates:
            print("ERROR: --candidates is required for sample audit mode (--n)")
            sys.exit(1)
        audit_sample(Path(args.candidates), args.n)

    elif args.submission:
        candidates_path = Path(args.candidates) if args.candidates else None
        audit_submission(Path(args.submission), candidates_path, args.top)

    else:
        print("Usage:")
        print("  Phase 2a (sample check):   python audit.py --candidates candidates.jsonl --n 1000")
        print("  Phase 4  (submission check): python audit.py --submission submission.csv --candidates candidates.jsonl --top 50")


if __name__ == "__main__":
    main()
