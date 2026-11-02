# CSC 618 — Week 10, Part C: Real Dataset Retrieval Experiment

This is the **standalone Part C** experiment. It is independent of the Self Study 1 and Part B folders. It reuses a **copy of `evaluation.py`** so that imports work without any other part of the assignment.

**Models:** TF-IDF, BM25, and a Dirichlet-smoothed language model.  
**Dataset:** public Cranfield collection accessed by `ir_datasets.load("cranfield")`.  
**Evaluation:** 1,400 documents; 225 queries; 1,837 query-document judgments; top 10 returned per query.  
**Metrics:** Precision@10, Recall@10, F1@10, Precision@3, AP@10, RR@10, nDCG@10 and averages (including MAP@10 and MRR@10).

## Files

- `part_c.py`: downloads/loads Cranfield, runs the three retrieval models, writes the results.
- `evaluation.py`: shared metric implementations copied into this standalone folder.
- `requirements.txt`: three third-party Python libraries.
- `run_windows.bat` and `run_mac.sh`: optional shortcuts **after** installing requirements.
- `results/part_c_per_query.csv`: supplied output (675 query/model rows).
- `results/part_c_ranked_results.csv`: supplied output (6,750 document-ranking rows).
- `results/part_c_summary.json`: supplied dataset details and aggregate scores.
- `verify_results.py`: checks the structure and aggregate metrics of the **included saved files**; this does not rerun retrieval.

## How to run in VS Code — Windows PowerShell

1. Download the repository ZIP (or clone the repository). Extract it.
2. In VS Code, choose **File > Open Folder**, then open `CSC618_Part_C_Standalone`.
3. Choose **Terminal > New Terminal**. Make sure the terminal path is the folder that contains `part_c.py`.
4. Check that Python is available:

   ```powershell
   py --version
   ```

5. Recommended: create a separate environment and install all required libraries:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

6. Run the complete experiment:

   ```powershell
   .\.venv\Scripts\python.exe part_c.py
   ```

   Alternatively, without a virtual environment:

   ```powershell
   py -m pip install -r requirements.txt
   py part_c.py
   ```

7. The program prints JSON with `metadata` and `aggregate`, and writes the three files in `results/`.
8. To inspect the provided saved results without downloading the dataset, run:

   ```powershell
   py verify_results.py
   ```

## How to run — macOS/Linux

Open `CSC618_Part_C_Standalone` in VS Code and use the terminal:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python part_c.py
python3 verify_results.py
```

**First run requires internet access** to download the public Cranfield dataset; subsequent runs usually use the cached dataset. Runtime depends on your computer, and `ir_datasets` may ask for dataset-download confirmation or report a download problem. See https://ir-datasets.com/cranfield.html for the source and dataset information.

## Results from the supplied output files

| Model | P@10 | R@10 | F1@10 | P@3 | MAP@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.2284 | 0.3771 | 0.2568 | 0.3452 | 0.2242 | 0.5046 | 0.2813 |
| BM25 | 0.2258 | 0.3785 | 0.2559 | 0.3393 | 0.2229 | 0.4992 | 0.2847 |
| LM-Dirichlet | 0.2067 | 0.3580 | 0.2369 | 0.3141 | 0.2040 | 0.4779 | 0.2638 |

Results are **from the provided CSV/JSON exports**. The package build verified their consistency, but **the full Cranfield retrieval run was not independently executed in the packaging environment** because `ir_datasets` was unavailable there. Run `part_c.py` yourself before describing the outputs as reproduced on your PC.

## What to show in the Part C Word report (minimal evidence)

- **Code screenshot:** `part_c.py` lines 63–82: BM25, language model, and ranked-results evaluation (one or two clear images).
- **Execution screenshot:** after successfully running `part_c.py`, capture the actual terminal JSON showing all three models' aggregate results.
- **Optional additional image:** Query 1 rows from `results/part_c_per_query.csv`; the editable Word results tables are already sufficient if screenshots are limited.

For a walkthrough of dataset loading and TF-IDF, the other useful code ranges are `part_c.py` lines 28–54 and 45–62. Do not replace results with made-up numbers.

## Academic reporting note

Cranfield has graded qrels. For binary scores, this implementation treats grades above zero as relevant and unjudged documents as not relevant. The nDCG calculation uses full grades. The three retrieval methods are evaluated at the same top-10 cutoff. Results on this historical research benchmark do not automatically generalize to other collections.
