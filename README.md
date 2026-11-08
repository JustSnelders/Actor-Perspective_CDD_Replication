# Replication: Multi-perspective Concept Drift Detection (BPIC'17)

This repository replicates the process-level concept drift detection on BPIC'17 (objective O1: Table 1, the average
and maximum signal change of Sect. 5.2, and Table 2) of:

> E. L. Klijn, F. Mannhardt, D. Fahland. *Multi-perspective Concept Drift Detection: Including the Actor
> Perspective.* CAiSE 2024, LNCS 14663, pp. 141–157. https://doi.org/10.1007/978-3-031-61057-8_9

This replication uses the authors' own unmodified code and configuration.

## Contents

| Path | What it is |
|---|---|
| `authors_code/` | Unmodified copy of the authors' repository [ekg-bpic17-concept-drift-detection-multi-perspective](https://github.com/multi-dimensional-process-mining/ekg-bpic17-concept-drift-detection-multi-perspective), commit `1a5788018507ca36c4e4da47b7050f448abddd3c` (3 Apr 2024). |
| `scripts/run_replication.py` | Wrapper that runs the authors' `main.py` steps with four runtime adaptations to current library versions and two bug fixes, without editing the authors' files. |
| `scripts/compare_results.py` | Compares change points and average/maximum signal changes with the paper's Table 1 and Sect. 5.2. |
| `scripts/table2_detailed.py` | Recomputes the paper's Table 2 for the replication's change points. |
| `scripts/plot_task_signals.py` | (Optional) Re-draws the paper's Fig. 3 (tasks T02, T03, T10). |

Only `scripts/run_replication.py` is needed to run the replication, the other scripts turn its raw outputs into the comparisons with the paper. A folder `work/` is filled by the wrapper and holds the input CSV, cached intermediate results and the raw pipeline outputs. The results are stored in `results/`.


## Requirements

- Windows, tested on Windows 11.
- Python 3.13
- Neo4j Desktop 2 2026.04, APOC Core and APOC Extended plugins.
- About 16 GB RAM (Neo4j heap 10 GB) and about 2 GB of free disk space.

## Step by step guide

1. Get this repository, open location in a terminal

2. Create a Python environment, install requirements.txt
   ```bash
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Neo4j as detailed in the author's Github repo (https://github.com/multi-dimensional-process-mining/ekg-bpic17-concept-drift-detection-multi-perspective)
   - Set up the DBMS ("Create a new graph database" steps)
   - Rename `config.example.yaml` to `config.yaml` and set configuration ("Set configuration file" steps)
   - Run the DBMS in Neo4j

4. Download the BPIC'17 CSV of Esser & Fahland, *Event Data and Queries for
   Multi-Dimensional Event Data in the Neo4j Graph Database*, Zenodo, https://doi.org/10.5281/zenodo.3865222, and
   save it as `work/data/BPI_Challenge_2017.csv`

5. Build the event knowledge graph and the task instances (authors' steps 0–2, about 25 minutes):
   ```bash
   python scripts/run_replication.py build
   ```

6. Cluster the tasks and detect drift (authors' steps 3, 5, 9, 10, about 30 minutes), then copy the outputs
   to `results/`:
   ```bash
   python scripts/run_replication.py detect export
   ```

7. Run scripts to compare results with the paper:
   ```bash
   python scripts/compare_results.py
   python scripts/table2_detailed.py
   python scripts/plot_task_signals.py
   ```

All steps are deterministic; running them again gives the same results. Steps that were interrupted can be
restarted, as the per-window subgraphs are cached in `work/output_intermediate/`.

## Results

`results/` contains:

- `table1_comparison.csv`: change points per feature set and penalty, original vs. replication (paper's Table 1)
- `signal_change_comparison.csv`: average and maximum signal change at penalty 3, original vs. replication
  (paper's Sect. 5.2), including the values per change point
- `table2_comparison.csv`: signal change of tasks and variants with the fractions and strongest changes of their
  activities and activity pairs (paper's Table 2)
- `fig_task_signals.pdf`: counterpart of the paper's Fig. 3
- `cps_*.csv`, `signal_magnitudes_3_*.csv`, `signal_change_magnitude_avg_mean_3.csv`: raw outputs of the
  authors' pipeline (change points per feature set, signal changes per feature and change point, averages)
