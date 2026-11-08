import ast
from pathlib import Path
import pandas as pd

# Determine filepaths
ROOT = Path(__file__).resolve().parent.parent  # repository root
BASE = ROOT / "work" / "output_final" / "BPIC17"
RESULTS = ROOT / "results"

# Name feature sets in dictionary
FEATURE_SETS = {"task_relative": "F_T^N", "task_variant_relative": "F_V^N",
                "activity_relative": "F_A^N", "activity_handover_case_relative": "F_A^c"}

# Represent table 1 of the original paper
ORIGINAL_TABLE1 = {
    2: {"task_relative": [11, 137, 145, 200, 328, 366, 386], "task_variant_relative": [9, 146, 196, 330, 368],
        "activity_relative": [11, 145, 183, 191, 327, 367, 385],
        "activity_handover_case_relative": [11, 198, 329, 366, 384]},
    3: {"task_relative": [11, 200, 328, 366], "task_variant_relative": [9, 196, 330, 368],
        "activity_relative": [11, 198, 327, 365], "activity_handover_case_relative": [11, 198, 329, 366]},
    4: {"task_relative": [11, 200, 328, 366], "task_variant_relative": [196, 330],
        "activity_relative": [11, 198, 365], "activity_handover_case_relative": [11, 199, 366]},
}

# Represent average and maximum signal change at penalty 3 of original paper (under "Mean Change Signal")
ORIGINAL_SIGNAL = {"task_relative": (1.2, 17.4), "task_variant_relative": (0.2, 17.3),
                   "activity_relative": (0.8, 6.5), "activity_handover_case_relative": (0.1, 6.7)}

# Pair replication change points with original paper change points within 4 days
def match_cps(replication, original, tol=4):
    pairs, rest = [], list(original)
    unmatched = []
    for cp in replication:
        cands = sorted(rest, key=lambda p: abs(p - cp))
        if cands and abs(cands[0] - cp) <= tol:
            pairs.append((cp, cands[0]))
            rest.remove(cands[0])
        else:
            unmatched.append(cp)
    return pairs, unmatched, rest

# Main function
def main():
    # Create results folder
    RESULTS.mkdir(exist_ok=True)

    # Read author's output file of detected change points
    cps = pd.read_csv(BASE / "task_concept_drift_detection" / "process_level_drift" / "cps_all_features.csv",
                      index_col=0)

    # Build comparison with the original paper's table 1
    rows = []
    for pen in (2, 3, 4):
        for fs in FEATURE_SETS:
            repl = ast.literal_eval(cps.loc[fs, f"1_{pen}"])
            orig = ORIGINAL_TABLE1[pen][fs]
            pairs, only_repl, only_orig = match_cps(repl, orig)
            rows.append({"penalty": pen, "feature_set": FEATURE_SETS[fs], "original": orig, "replication": repl,
                         "identical": repl == orig, "matched_within_4_days": len(pairs),
                         "max_shift_days": max([abs(a - b) for a, b in pairs], default=None),
                         "only_replication": only_repl, "only_original": only_orig})
    
    # Output comparison table
    df1 = pd.DataFrame(rows)
    df1.to_csv(RESULTS / "table1_comparison.csv", index=False)
    print(df1.to_string())

    # Read authors' output file
    sig = pd.read_csv(BASE / "task_concept_drift_evaluation" / "signal_magnitude" /
                      "signal_change_magnitude_avg_mean_3.csv", index_col=0, header=[0, 1])

    # Loop over the feature sets
    rows = []
    for fs in FEATURE_SETS:
        per_cp = sig[fs]

        # Calculate average signal and maximum signal change of replication
        repl_avg = per_cp.loc["mean"].astype(float).mean()
        repl_max = per_cp.loc["max"].astype(float).max()

        # Compare results with original
        orig_avg, orig_max = ORIGINAL_SIGNAL[fs]
        rows.append({"feature_set": FEATURE_SETS[fs], "original_avg": orig_avg, "replication_avg": round(repl_avg, 2),
                     "diff_avg": round(repl_avg - orig_avg, 2), "original_max": orig_max,
                     "replication_max": round(repl_max, 2), "diff_max": round(repl_max - orig_max, 2),
                     "per_cp_avg": [round(v, 2) for v in per_cp.loc["mean"].astype(float)],
                     "per_cp_max": [round(v, 2) for v in per_cp.loc["max"].astype(float)],
                     "change_point_intervals": list(per_cp.columns)})
    
    # Output resulting table
    df2 = pd.DataFrame(rows)
    df2.to_csv(RESULTS / "signal_change_comparison.csv", index=False)
    print(df2.to_string())


if __name__ == "__main__":
    main()
