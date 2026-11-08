import os
import shutil
import sys
import time
from pathlib import Path

# Determine filepaths
ROOT = Path(__file__).resolve().parent.parent
AUTHORS = ROOT / "authors_code"
WORK = ROOT / "work"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config.yaml"
ANALYSIS_CONFIG = ROOT / "config_analysis.yaml"

# Specify pipeline steps
STEP_GROUPS = {"build": ["clear", "populate", "tasks"], "detect": ["cluster", "process", "magnitude", "overall"]}
FLAGS = {"cluster": "step_cluster_tasks", "visualize": "step_visualize_clusters",
         "process": "step_process_level_drift_detection", "actor": "step_actor_drift_detection",
         "collab": "step_collab_drift_detection", "compare": "step_compare_to_process_drift",
         "magnitude": "step_calculate_magnitude_signal_changes",
         "overall": "step_calculate_overall_average_max_signal_change",
         "detailed": "step_detailed_change_signal_analysis"}
ALL_STEPS = ["clear", "populate", "tasks", *FLAGS]
USAGE = ("usage: python scripts/run_replication.py <step> [<step> ...]\n"
         "steps: build, detect, export, or single steps: " + ", ".join(ALL_STEPS))

# Bridge gap between old version functionalities and current version of libraries
def apply_runtime_adaptations():
    import pandas as pd
    from neo4j import Session
    from promg.modules.data_importer import Importer
    from promg.modules.db_management import DBManagement
    from promg.cypher_queries.db_managment_ql import DBManagementQueryLibrary as dbm_ql

    # Store pandas text cols as objects 
    pd.set_option("future.infer_string", False)

    # Add old PromG function call aliases
    if not hasattr(Session, "write_transaction"): 
        Session.write_transaction = Session.execute_write
        Session.read_transaction = Session.execute_read

    # Return status row after clearing database
    def _clear_db(self, replace=True):  # 3
        self.connection.exec_query(dbm_ql.get_replace_db_query, **{"db_name": self.connection.db_name})
        return True
    
    DBManagement.clear_db = _clear_db

    # Deal with permission errors when deleting temporary files in PromG
    def _delete_log_grouped_by_labels(self, file_name):  # 4
        path = Path(self.get_import_directory(), file_name)
        for _ in range(10):
            try:
                if os.path.exists(path):
                    os.remove(path)
                return
            except PermissionError:
                time.sleep(1)
        print(f"[run_replication] could not delete temporary file {path}; left in place")

    Importer._delete_log_grouped_by_labels = _delete_log_grouped_by_labels

# Wrapper function importing the author's original clustering, fixing the bugs
def _encode_and_cluster_specific(self, min_variant_length, manual_clusters, num_clusters,
                                 include_remainder, clustering_description):  # 5
    """Same logic as TaskClustering.encode_and_cluster_specific in authors_code, except for the label bookkeeping."""
    import pandas as pd
    from modules.task_clustering import variant_encoder_selection
    from modules.task_clustering.task_clustering import cluster

    # Build path of the cache file
    out = f"{self.intermediate_output_directory}\\variants_clustered_{self.dataset_name}_{clustering_description}.pkl"
    if os.path.exists(out):
        self.df_variants_clustered = pd.read_pickle(out)
        return self.df_variants_clustered

    # Pick authors' variant encoder
    e = variant_encoder_selection.get_variant_encoder(self.dataset_name)

    # Store each variant by its manual clusters rather than the other way around
    manual_ids = {i: key for key, ids in manual_clusters.items() for i in ids}

    # Cluster using the authors' clustering method
    df_to_cluster = self.df_variants[self.df_variants['variant_length'] >= min_variant_length]
    df_clusters = cluster(e.encode(df_to_cluster), num_clusters, self.metric, self.linkage)
    df = pd.concat([self.df_variants, df_clusters], axis=1)

    # Give each variant task label
    labels, a_id = [], 1
    for _, row in df.iterrows():
        # Assign manual cluster ids first
        if row['ID'] in manual_ids:
            labels.append(manual_ids[row['ID']])

        # Handle unclustered variants according to authors' config (include_remainder)
        elif pd.isna(row['cluster']):
            if include_remainder and row['variant_length'] == 1:
                labels.append(f"A{str(a_id).zfill(2)}")
                a_id += 1
            else:
                labels.append(None)

        # If clustered, the cluster number becomes the task name
        else:
            labels.append(f"T{str(int(row['cluster'])).zfill(2)}")

    # Replace cluster numbers with task labels
    df['sk_label'] = df['cluster']
    df['cluster'] = pd.Series(labels, index=df.index, dtype=object)
    self.df_variants_clustered = df

    # Output resulting df
    df.to_pickle(out)
    return df

# Read database config in config.yaml
def load_config():
    from promg import Configuration
    config = Configuration.init_conf_with_config_file(path=CONFIG)
    for attr in ("semantic_header_path", "dataset_description_path"):
        p = Path(getattr(config, attr))
        if not p.is_absolute():
            setattr(config, attr, ROOT / p)
    return config

# Export the results
def export_results():
    RESULTS.mkdir(exist_ok=True)
    seen = {}
    for src in sorted((WORK / "output_final").rglob("*")):
        if src.is_file():
            if src.name in seen:
                raise RuntimeError(f"duplicate result name {src.name}: {seen[src.name]} and {src}")
            seen[src.name] = src
            shutil.copy2(src, RESULTS / src.name)
    print(f"[run_replication] exported {len(seen)} files to {RESULTS}")

# Main pipeline function
def main(argv):
    # Turn cmd commands into list of steps
    steps = []
    for s in argv:
        steps += STEP_GROUPS.get(s, [s])
    unknown = [s for s in steps if s not in ALL_STEPS + ["export"]]
    if not steps or unknown:
        sys.exit(USAGE if not steps else f"unknown step(s): {unknown}\n{USAGE}")
    pipeline_steps = [s for s in steps if s != "export"]

    if pipeline_steps:
        # Prepare the environment
        if not CONFIG.exists():
            sys.exit("config.yaml not found: copy config.example.yaml to config.yaml and fill in your Neo4j settings")
        WORK.mkdir(exist_ok=True)
        os.chdir(WORK)
        sys.path.insert(0, str(AUTHORS))
        apply_runtime_adaptations()

        # Point analysis configuration to the root file
        import analysis_configuration
        analysis_configuration.AnalysisConfiguration.__init__.__defaults__ = (ANALYSIS_CONFIG,)

        # Import author's main.py as a module 'original'
        import main as original

        # Fetch the two classes that contain bugs
        from modules.task_clustering.task_clustering import TaskClustering
        from modules.task_concept_drift_detection.concept_drift_evaluation import ConceptDriftEvaluation
        
        # Replace clustering method with int bug with corrected version
        TaskClustering.encode_and_cluster_specific = _encode_and_cluster_specific

        # Fix author bug that points to unexisting method (inconsistent method call)
        ConceptDriftEvaluation.calculate_overall_average_max_signal_change = \
            ConceptDriftEvaluation.cd_calculate_overall_average_max_signal_change

        # Run the authors' pipeline with the specified steps
        for step, flag in FLAGS.items():
            setattr(original, flag, step in pipeline_steps)
        start = time.time()
        original.main(config=load_config(),
                      step_clear_db="clear" in pipeline_steps,
                      step_populate_graph="populate" in pipeline_steps,
                      step_build_tasks="tasks" in pipeline_steps)
        print(f"[run_replication] steps {pipeline_steps} finished in {time.time() - start:.0f} s")

    # Export results for each step
    if "export" in steps:
        export_results()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
