"""Check INCLUDED saved Part C outputs without downloading Cranfield or importing third-party packages."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent / "results"
with (root / "part_c_per_query.csv").open(newline="", encoding="utf-8") as f:
    per_query = list(csv.DictReader(f))
with (root / "part_c_ranked_results.csv").open(newline="", encoding="utf-8") as f:
    rankings = list(csv.DictReader(f))
summary = json.loads((root / "part_c_summary.json").read_text(encoding="utf-8"))
models = ("TF-IDF", "BM25", "LM-Dirichlet")
assert len(per_query) == 675, f"Expected 675 per-query rows, got {len(per_query)}"
assert len(rankings) == 6750, f"Expected 6750 rankings, got {len(rankings)}"
for model in models:
    rows = [r for r in per_query if r["Model"] == model]
    assert len(rows) == 225
    expected = summary["aggregate"][model]
    pairs = (("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1"),
             ("P3", "P3"), ("AP", "MAP@10"), ("RR", "MRR@10"), ("nDCG10", "nDCG10"))
    for source, target in pairs:
        actual = mean(float(r[source]) for r in rows)
        assert abs(actual - expected[target]) < 1e-9, f"Mismatch: {model} {target}"
    print(f"{model}: 225 queries; saved metrics agree with summary")
print("PASS: Saved Part C CSV/JSON results are internally consistent.")
print("Note: This checks saved files; it does not execute the retrieval models.")
