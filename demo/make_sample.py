import json
from pathlib import Path

# Paths
CANDIDATES_PATH = Path("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
SAMPLE_OUT_PATH = Path("sample_1k.jsonl")

print("Extracting 1,000 candidates for Streamlit sandbox...")
sample_candidates = []

with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        sample_candidates.append(line.strip())

with open(SAMPLE_OUT_PATH, "w", encoding="utf-8") as f_out:
    for candidate in sample_candidates:
        f_out.write(candidate + "\n")

print(f"SUCCESS: Extracted 1,000 candidates to {SAMPLE_OUT_PATH}")
