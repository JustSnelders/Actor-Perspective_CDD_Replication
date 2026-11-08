from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from neo4j import GraphDatabase

# Define path directory
ROOT = Path(__file__).resolve().parent.parent

# Define penalty, tasks, variants
PENALTY = 3
TASKS = {1: ["m04", "T03", "T07"], 2: ["m01", "m03", "m04", "T01", "T02", "T11"], 3: ["m02", "m04", "m05"],
         4: ["m05", "T11"]}
VARIANTS = {2: [4, 6, 8, 13], 3: [4, 27, 12, 43]}

# Represent table 2 of the original paper
ORIGINAL = {(1, "m04"): -11.19, (1, "T03"): 10.13, (1, "T07"): -14.71,
         (2, "m01"): 3.79, (2, "m03"): -5.41, (2, "m04"): -2.06, (2, "T01"): 1.37, (2, "T02"): 2.18,
         (2, "T11"): 1.28, (2, "v04"): -1.13, (2, "v06"): 1.18, (2, "v08"): 2.40, (2, "v13"): -1.61,
         (3, "m02"): -13.02, (3, "m04"): 1.38, (3, "m05"): 15.27, (3, "v04"): -4.79, (3, "v27"): 4.10,
         (3, "v12"): -2.14, (3, "v43"): 1.90, (4, "m05"): -11.88, (4, "T11"): 17.38}

# Find change point intervals in authors' signal magnitude files and label them
def cp_keys(df):
    """'0_11_vs_11_198' style keys of the magnitude CSV, ordered by change point."""
    keys = sorted({c for c, _ in df.columns if "_vs_" in c}, key=lambda k: int(k.split("_")[1]))
    return {i + 1: k for i, k in enumerate(keys)}

# Classify activities (pairs) into 2 groups: move WITH the task, move AGAINST the task
def classify(changes, direction):
    same = {k: [v, 0] for k, v in changes.items() if abs(v) > 0.1 and (v > 0) == (direction > 0)}
    opp = {k: [v, 0] for k, v in changes.items() if abs(v) > 0.1 and (v > 0) != (direction > 0)}
    return same, opp

# Pick strongest change of the classified activity groups
def extreme(d, direction, same):
    if not d:
        return np.nan
    vals = [v[0] for v in d.values()]
    rising = direction > 0
    return (max(vals) if rising else min(vals)) if same else (min(vals) if rising else max(vals))

# Compute all comparison values for one row of table 5 (one task/variant at one cp)
def analyse(sequences, change, act_changes, df_changes):
    # Split into same and opposite trend
    a_same, a_opp = classify(act_changes, change)
    d_same, d_opp = classify(df_changes, change)

    # Compute fractions per variant
    fr = {"act_same": [], "act_opp": [], "df_same": [], "df_opp": []}
    for v in sequences:
        # Build consecutive activity pairs of the variant
        pairs = [f"{v[i]}_{v[i + 1]}" for i in range(len(v) - 1)]

        # Compute shares for variants with >= 2 activities
        if pairs:
            fr["df_same"].append(sum(p in d_same for p in pairs) / len(pairs))
            fr["df_opp"].append(sum(p in d_opp for p in pairs) / len(pairs))

        # Compute shares for individual activities
        fr["act_same"].append(sum(a in a_same for a in v) / len(v))
        fr["act_opp"].append(sum(a in a_opp for a in v) / len(v))
    
    # Average over the variants
    res = {f"rho_{k}": (np.mean(x) if x else np.nan) for k, x in fr.items()}
    res.update({"max_act_same": extreme(a_same, change, True), "max_act_opp": extreme(a_opp, change, False),
                "max_df_same": extreme(d_same, change, True), "max_df_opp": extreme(d_opp, change, False)})
    return res

# Main function
def main():
    # Load signal changes of all 4 feature sets and find cp intervals
    sm = ROOT / "work" / "output_final" / "BPIC17" / "task_concept_drift_evaluation" / "signal_magnitude"
    read = lambda fs: pd.read_csv(sm / f"signal_magnitudes_{PENALTY}_{fs}.csv", index_col=0, header=[0, 1])
    df_task, df_var = read("task_relative"), read("task_variant_relative")
    df_act, df_actdf = read("activity_relative"), read("activity_handover_case_relative")
    k_task, k_var, k_act, k_actdf = cp_keys(df_task), cp_keys(df_var), cp_keys(df_act), cp_keys(df_actdf)
    print("intervals:", k_task, k_var, k_act, k_actdf, sep="\n  ")

    # Read config.yaml
    c = yaml.safe_load(open(ROOT / "config.yaml"))

    # Connect to Neo4j database
    driver = GraphDatabase.driver(c["uri"], auth=(c["user"], c["password"]))

    # Find all task instances, return their activity sequence
    q_var = "MATCH (ti:TaskInstance) WHERE ti[$prop] = $val RETURN DISTINCT ti.variant AS variant"

    # Trace tasks to events and return distinct activities as name+lifecycle
    q_act = ("MATCH (ti:TaskInstance)-[:CONTAINS]->(e:Event) WHERE ti[$prop] = $val "
             "RETURN DISTINCT e.activity+'+'+e.lifecycle AS a")

    # Find two events e1, e2 in same task instance, where e2 directly follows e1
    q_df = ("MATCH (e1:Event)<-[:CONTAINS]-(ti:TaskInstance)-[:CONTAINS]->(e2:Event) WHERE ti[$prop] = $val "
            "AND (e1)-[:DF_CASEAWO]->(e2) AND (e1)-[:DF_RESOURCE]->(e2) "
            "RETURN DISTINCT e1.activity+'+'+e1.lifecycle AS a1, e2.activity+'+'+e2.lifecycle AS a2")

    # Build one row of table 5 for each of the 22 tasks and variants in paper's table 2
    rows = []
    with driver.session(database=c["db_name"]) as s:
        items = [(cp, t, "cluster", t) for cp, ts in TASKS.items() for t in ts] + \
                [(cp, f"v{str(v).zfill(2)}", "ID", v) for cp, vs in VARIANTS.items() for v in vs]
        for cp, label, prop, val in items:
            if cp not in k_task:
                continue
            if prop == "cluster":
                change = df_task.loc[f"task_{val}_relative_freq", (k_task[cp], "mean")]
            else:
                change = df_var.loc[f"task_variant_{val}_relative_freq", (k_var[cp], "mean")]
            seqs = [r["variant"] for r in s.run(q_var, prop=prop, val=val)]
            acts = [r["a"] for r in s.run(q_act, prop=prop, val=val)]
            dfs = [(r["a1"], r["a2"]) for r in s.run(q_df, prop=prop, val=val)]
            act_changes = {a: df_act.loc[f"activity_{a}_relative_freq", (k_act[cp], "mean")] for a in acts
                           if f"activity_{a}_relative_freq" in df_act.index}
            df_changes = {f"{a}_{b}": df_actdf.loc[f"activity_{a}_{b}_relative_freq", (k_actdf[cp], "mean")]
                          for a, b in dfs if f"activity_{a}_{b}_relative_freq" in df_actdf.index}
            row = {"cp": cp, "item": label, "original_change": ORIGINAL.get((cp, label), np.nan), "replication_change": change}
            row.update(analyse(seqs, change, act_changes, df_changes))
            rows.append(row)

    # Turn collected rows into table and save results
    out = pd.DataFrame(rows)
    out["diff"] = out["replication_change"] - out["original_change"]
    out["same_direction"] = np.sign(out["replication_change"]) == np.sign(out["original_change"])
    (ROOT / "results").mkdir(exist_ok=True)
    out.round(2).to_csv(ROOT / "results" / "table2_comparison.csv", index=False)
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
