import yaml
from pathlib import Path


class AnalysisConfiguration:
    def __init__(self, path=Path("config_analysis.yaml")):
        config = yaml.safe_load(open(path))

        self.dataset_name = config["dataset_name"]
        self.intermediate_output_directory = config["intermediate_output_directory"]
        self.final_output_directory = config["final_output_directory"]

        self.cluster_min_variant_freq = config['min_variant_freq']
        self.num_clusters = config['num_clusters']
        self.cluster_min_variant_length = config['cluster_min_variant_length']
        self.manual_clusters = config['manual_clusters']
        self.cluster_include_remainder = config['cluster_include_remainder']
        self.leftover_cluster = config['leftover_cluster']
        self.clustering_description = config['clustering_description']

        self.window_sizes = config['window_sizes']
        self.penalties = config['penalties']
        self.process_drift_feature_sets = config['process_drift_feature_sets']
        self.actor_drift_feature_sets = config['actor_drift_feature_sets']
        self.min_actor_frequency = config['min_actor_frequency']
        self.min_collab_frequency = config['min_collab_frequency']

        self.comp_window_size = config['comp_window_size']
        self.comp_feature_set_name_process_level = config['comp_feature_set_name_process_level']
        self.comp_process_drift_penalty = config['comp_process_drift_penalty']
        self.comp_subgroup_to_compare = config['comp_subgroup_to_compare']
        self.comp_subgroup_feature_set_name = config['comp_subgroup_feature_set_name']
        self.comp_subgroup_drift_penalty = config['comp_subgroup_drift_penalty']

        self.mc_window_size = config['mc_window_size']
        self.mc_penalty = config['mc_penalty']

        self.dc_penalty = config['dc_penalty']
        self.dc_task_dict = config['dc_task_dict']
        self.dc_variant_dict = config['dc_variant_dict']
