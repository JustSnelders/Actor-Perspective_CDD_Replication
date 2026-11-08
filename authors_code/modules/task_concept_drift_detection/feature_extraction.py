from datetime import timedelta
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from os import path
import os
from PerformanceRecorder import PerformanceRecorder

from promg import DatabaseConnection
from promg.data_managers.semantic_header import ConstructedNodes

from queries.task_cd_detection import ConceptDriftDetectionTasksQueryLibrary as ql
from queries import query_result_parser as qp


class FeatureExtraction:
    def __init__(self, db_connection, dataset_name: str, case: ConstructedNodes, actor: ConstructedNodes, exclude_cluster: str):
        self.connection = db_connection
        self.intermediate_output_directory = f"output_intermediate\\{dataset_name}\\subgraphs\\"
        self.case = case
        self.actor = actor
        # TODO: retrieve event class from semantic header?
        os.makedirs(self.intermediate_output_directory, exist_ok=True)
        self.num_windows = None
        self.exclude_cluster = exclude_cluster
        self.task_subgraphs_nodes = []
        self.task_subgraphs_edges = []
        self.event_subgraphs_nodes = []
        self.event_subgraphs_edges = []
        self.durations = []
        self.task_node_based_features = ["distinct_task_count", "distinct_task_variant_count",
                                         "count_per_task", "count_per_task_variant",
                                         "count_per_task_relative", "count_per_task_variant_relative",
                                         "case_volume", "total_task_count"]
        self.event_node_based_features = ["distinct_activity_count", "count_per_activity",
                                          "count_per_activity_relative",
                                          "total_activity_count"]
        self.task_edge_based_features = ["distinct_task_handover_count_actor",
                                         "distinct_task_variant_handover_count_actor",
                                         "distinct_task_handover_count_case",
                                         "distinct_task_variant_handover_count_case",
                                         "count_per_task_handover_actor_relative",
                                         "count_per_task_variant_handover_actor_relative",
                                         "count_per_task_handover_actor", "count_per_task_variant_handover_actor",
                                         "count_per_task_handover_case_relative",
                                         "count_per_task_variant_handover_case_relative",
                                         "count_per_task_handover_case", "count_per_task_variant_handover_case",
                                         "total_task_handover_count_actor",
                                         "total_task_handover_count_case",
                                         "average_number_handovers_case", "average_number_handovers_actor"]
        self.event_edge_based_features = ["distinct_activity_handover_count_actor",
                                          "distinct_activity_handover_count_case",
                                          "count_per_activity_handover_actor_relative",
                                          "count_per_activity_handover_actor",
                                          "count_per_activity_handover_case_relative",
                                          "count_per_activity_handover_case"]
        self.performance_features = ["average_case_duration", "case_throughput_speed", "case_throughput_velocity"]
        self.pr = PerformanceRecorder(dataset_name, 'extracting_features')

    def retrieve_subgraphs_for_feature_extraction(self, window_size: int, feature_set=""):
        graph_start_date = qp.parse_timestamp(self.connection.exec_query(ql.q_get_start_timestamp), 'start_time')
        graph_end_date = qp.parse_timestamp(self.connection.exec_query(ql.q_get_end_timestamp), 'end_time')

        # Calculate the number of windows
        self.num_windows = (graph_end_date - graph_start_date).days // window_size

        for window in range(0, self.num_windows):

            if any(feature in feature_set for feature in self.task_node_based_features) or feature_set == "":
                if path.exists(f"{self.intermediate_output_directory}task_node_ws{window_size}_w{window}.pkl"):
                    df_task_subgraphs_nodes = pd.read_pickle(
                        f"{self.intermediate_output_directory}task_node_ws{window_size}_w{window}.pkl")
                else:
                    df_task_subgraphs_nodes = qp.parse_to_dataframe(
                        self.connection.exec_query(ql.q_retrieve_task_subgraph_nodes, **{
                            "start_date": (graph_start_date + timedelta(days=window * window_size)).strftime("%Y-%m-%d"),
                            "end_date": (graph_start_date + timedelta(days=(window + 1) * window_size)).strftime("%Y-%m-%d"),
                            "case": self.case, "resource": self.actor, "exclude_cluster": self.exclude_cluster}))
                    df_task_subgraphs_nodes.to_pickle(
                        f"{self.intermediate_output_directory}task_node_ws{window_size}_w{window}.pkl")
                self.task_subgraphs_nodes.append(df_task_subgraphs_nodes)
                self.pr.record_performance(f"extract_task_subgraphs_nodes_{window}/{self.num_windows}")

            if any(feature in feature_set for feature in self.event_node_based_features) or feature_set == "":
                if path.exists(f"{self.intermediate_output_directory}event_node_ws{window_size}_w{window}.pkl"):
                    df_event_subgraphs_nodes = pd.read_pickle(
                        f"{self.intermediate_output_directory}event_node_ws{window_size}_w{window}.pkl")
                else:
                    df_event_subgraphs_nodes = qp.parse_to_dataframe(
                        self.connection.exec_query(ql.q_retrieve_event_subgraph_nodes, **{
                            "start_date": (graph_start_date + timedelta(days=window * window_size)).strftime("%Y-%m-%d"),
                            "end_date": (graph_start_date + timedelta(days=(window + 1) * window_size)).strftime("%Y-%m-%d"),
                            "case": self.case, "resource": self.actor}))
                    df_event_subgraphs_nodes.to_pickle(
                        f"{self.intermediate_output_directory}event_node_ws{window_size}_w{window}.pkl")
                self.event_subgraphs_nodes.append(df_event_subgraphs_nodes)
                self.pr.record_performance(f"extract_event_subgraphs_nodes_{window}/{self.num_windows}")

            if any(feature in feature_set for feature in self.task_edge_based_features) or feature_set == "":
                if path.exists(f"{self.intermediate_output_directory}task_edge_ws{window_size}_w{window}.pkl"):
                    df_task_subgraphs_edges = pd.read_pickle(
                        f"{self.intermediate_output_directory}task_edge_ws{window_size}_w{window}.pkl")
                else:
                    df_task_subgraphs_edges = qp.parse_to_dataframe(
                        self.connection.exec_query(ql.q_retrieve_task_subgraph_edges, **{
                            "start_date": (graph_start_date + timedelta(days=window * window_size)).strftime("%Y-%m-%d"),
                            "end_date": (graph_start_date + timedelta(days=(window + 1) * window_size)).strftime("%Y-%m-%d"),
                            "case": self.case, "resource": self.actor, "exclude_cluster": self.exclude_cluster}))
                    df_task_subgraphs_edges.to_pickle(
                        f"{self.intermediate_output_directory}task_edge_ws{window_size}_w{window}.pkl")
                self.task_subgraphs_edges.append(df_task_subgraphs_edges)
                self.pr.record_performance(f"extract_task_subgraphs_edges_{window}/{self.num_windows}")

            if any(feature in feature_set for feature in self.event_edge_based_features) or feature_set == "":
                if path.exists(f"{self.intermediate_output_directory}event_edge_ws{window_size}_w{window}.pkl"):
                    df_event_subgraphs_edges = pd.read_pickle(
                        f"{self.intermediate_output_directory}event_edge_ws{window_size}_w{window}.pkl")
                else:
                    df_event_subgraphs_edges = qp.parse_to_dataframe(
                        self.connection.exec_query(ql.q_retrieve_event_subgraph_edges, **{
                            "start_date": (graph_start_date + timedelta(days=window * window_size)).strftime("%Y-%m-%d"),
                            "end_date": (graph_start_date + timedelta(days=(window + 1) * window_size)).strftime("%Y-%m-%d"),
                            "case": self.case, "resource": self.actor}))
                    df_event_subgraphs_edges.to_pickle(
                        f"{self.intermediate_output_directory}event_edge_ws{window_size}_w{window}.pkl")
                self.event_subgraphs_edges.append(df_event_subgraphs_edges)
                self.pr.record_performance(f"extract_event_subgraphs_edges_{window}/{self.num_windows}")

            if any(feature in feature_set for feature in self.performance_features) or feature_set == "":
                if path.exists(f"{self.intermediate_output_directory}durations_ws{window_size}_w{window}.pkl"):
                    df_global = pd.read_pickle(
                        f"{self.intermediate_output_directory}durations_ws{window_size}_w{window}.pkl")
                else:
                    df_global = qp.parse_to_dataframe(
                        self.connection.exec_query(ql.q_retrieve_case_durations, **{
                            "start_date": (graph_start_date + timedelta(days=window * window_size)).strftime("%Y-%m-%d"),
                            "end_date": (graph_start_date + timedelta(days=(window + 1) * window_size)).strftime("%Y-%m-%d"),
                            "case": self.case}), timedelta_cols={'duration': 'minutes'})
                    df_global.to_pickle(
                        f"{self.intermediate_output_directory}durations_ws{window_size}_w{window}.pkl")
                self.durations.append(df_global)

    def apply_feature_extraction(self, features, actor="", actor_1="", actor_2=""):
        '''
        Adapted from Adams et al. (2021) https://github.com/niklasadams/explainable_concept_drift_pm
        '''
        feature_vectors = [[] for i in range(0, self.num_windows)]
        for feature in features:
            results = []
            # task node based features
            if feature == "case_volume":
                results = extract_number_of_cases(self.task_subgraphs_nodes, actor)
            if feature == "distinct_task_count":
                results = extract_distinct_performance_instance_count(self.task_subgraphs_nodes, 'task', actor)
            if feature == "total_task_count":
                results = extract_total_performance_instance_count(self.task_subgraphs_nodes, 'task', actor)
            if feature == "distinct_task_variant_count":
                results = extract_distinct_performance_instance_count(self.task_subgraphs_nodes, 'task_variant', actor)
            if feature == "count_per_task":
                results = extract_count_per_performance_instance(self.task_subgraphs_nodes, 'task', actor)
            if feature == "count_per_task_relative":
                results = extract_count_per_performance_instance_normalized(self.task_subgraphs_nodes, 'task', actor)
            if feature == "count_per_task_variant":
                results = extract_count_per_performance_instance(self.task_subgraphs_nodes, 'task_variant', actor)
            if feature == "count_per_task_variant_relative":
                results = extract_count_per_performance_instance_normalized(self.task_subgraphs_nodes, 'task_variant',
                                                                            actor)

            # task edge based features -- actor
            if feature == "distinct_task_handover_count_actor":
                results = extract_distinct_handover_count(self.task_subgraphs_edges, 'task', actor_1, actor_2,
                                                          self.actor.type)
            if feature == "distinct_task_variant_handover_count_actor":
                results = extract_distinct_handover_count(self.task_subgraphs_edges, 'task_variant', actor_1,
                                                          actor_2, self.actor.type)
            if feature == "total_task_handover_count_actor":
                results = extract_total_handover_count(self.task_subgraphs_edges, 'task', actor_1, actor_2, self.actor.type)

            if feature == "count_per_task_handover_actor_relative":
                results = extract_count_per_handover_normalized(self.task_subgraphs_edges, 'task', actor_1, actor_2,
                                                                self.actor.type)
            if feature == "count_per_task_handover_actor":
                results = extract_count_per_handover(self.task_subgraphs_edges, 'task', actor_1, actor_2, self.actor.type)
            if feature == "count_per_task_variant_handover_actor_relative":
                results = extract_count_per_handover_normalized(self.task_subgraphs_edges, 'task_variant', actor_1,
                                                                actor_2, self.actor.type)
            if feature == "count_per_task_variant_handover_actor":
                results = extract_count_per_handover(self.task_subgraphs_edges, 'task_variant', actor_1, actor_2,
                                                     self.actor.type)

            # task edge based features -- case
            if feature == "distinct_task_handover_count_case":
                results = extract_distinct_handover_count(self.task_subgraphs_edges, 'task', actor_1, actor_2, self.case.type)
            if feature == "distinct_task_variant_handover_count_case":
                results = extract_distinct_handover_count(self.task_subgraphs_edges, 'task_variant', actor_1, actor_2,
                                                          self.case.type)
            if feature == "total_task_handover_count_case":
                results = extract_total_handover_count(self.task_subgraphs_edges, 'task', actor_1, actor_2, self.case.type)
            if feature == "count_per_task_handover_case_relative":
                results = extract_count_per_handover_normalized(self.task_subgraphs_edges, 'task', actor_1, actor_2,
                                                                self.case.type)
            if feature == "count_per_task_handover_case":
                results = extract_count_per_handover(self.task_subgraphs_edges, 'task', actor_1, actor_2, self.case.type)
            if feature == "count_per_task_variant_handover_case_relative":
                results = extract_count_per_handover_normalized(self.task_subgraphs_edges, 'task_variant', actor_1,
                                                                actor_2, self.case.type)
            if feature == "count_per_task_variant_handover_case":
                results = extract_count_per_handover(self.task_subgraphs_edges, 'task_variant', actor_1, actor_2,
                                                     self.case.type)

            # event node based features
            if feature == "distinct_activity_count":
                results = extract_distinct_performance_instance_count(self.event_subgraphs_nodes, 'activity',
                                                                      actor)
            if feature == "count_per_activity_relative":
                results = extract_count_per_performance_instance_normalized(self.event_subgraphs_nodes,
                                                                            'activity', actor)
            if feature == "count_per_activity":
                results = extract_count_per_performance_instance(self.event_subgraphs_nodes, 'activity',
                                                                 actor)
            if feature == "total_activity_count":
                results = extract_total_performance_instance_count(self.event_subgraphs_nodes, 'activity',
                                                                   actor)

            # event edge based features -- actor
            if feature == "distinct_activity_handover_count_actor":
                results = extract_distinct_handover_count(self.event_subgraphs_edges, 'activity', actor_1,
                                                          actor_2, self.actor.type)
            if feature == "count_per_activity_handover_actor_relative":
                results = extract_count_per_handover_normalized(self.event_subgraphs_edges, 'activity',
                                                                actor_1, actor_2, self.actor.type)
            if feature == "count_per_activity_handover_actor":
                results = extract_count_per_handover(self.event_subgraphs_edges, 'activity',
                                                     actor_1, actor_2, self.actor.type)
            # event edge based features -- case
            if feature == "distinct_activity_handover_count_case":
                results = extract_distinct_handover_count(self.event_subgraphs_edges, 'activity', actor_1,
                                                          actor_2, self.case.type)
            if feature == "count_per_activity_handover_case_relative":
                results = extract_count_per_handover_normalized(self.event_subgraphs_edges, 'activity',
                                                                actor_1, actor_2, self.case.type)
            if feature == "count_per_activity_handover_case":
                results = extract_count_per_handover(self.event_subgraphs_edges, 'activity',
                                                     actor_1, actor_2, self.case.type)

            # global features
            if feature == "average_case_duration":
                results = extract_average_case_duration(self.durations)
            if feature == "case_throughput_speed":
                results = extract_case_throughput_speed(self.durations)
            if feature == "case_throughput_velocity":
                results = extract_case_throughput_velocity(self.durations)

            # more feature
            for i in range(0, len(results)):
                for result in results[i]:
                    feature_vectors[i].append(result)
        # set non existent features to zero
        feature_names__ = []
        for window in range(0, self.num_windows):
            for feature in range(0, len(feature_vectors[window])):
                feature_names__.append(feature_vectors[window][feature][0])

        feature_names = list(set(feature_names__))

        feature_lists = []
        for feature_name in range(0, len(feature_names)):
            feature_list = []
            for window in range(0, self.num_windows):
                existing_features = [i[0] for i in feature_vectors[window]]
                if feature_names[feature_name] in existing_features:
                    # find index
                    idx = existing_features.index(feature_names[feature_name])
                    feature_list.append(feature_vectors[window][idx][1])
                else:
                    feature_list.append(0)
            feature_lists.append(feature_list)

        # self.pr.record_performance(f"apply_extraction_{self.num_windows}")
        self.pr.record_total_performance()
        self.pr.save_to_file()

        return feature_names, np.asarray(feature_lists).transpose()

    def pca_reduction(self, features_np, dimensions, normalize=False, normalize_function='max'):
        '''Reduces a time series of features
        Adapted from Adams et al. (2021) https://github.com/niklasadams/explainable_concept_drift_pm
        features: Two dimensional array of features
        dimensions: Target dimensionality. Use 'mle' for automated choice of dimensionality
          Automated choice of dimensionality can sometimes fail, a manual choice of dimensionality
          is then needed. If the Feautres are more than the time series is long, the automated
          choice can not be applied. We sole this by first reducing the features to the length
          of the time series and then reducing automatically.
        normalize: Whether the feautres should be normalized before reduction. The
          features should be normalized, if they have very different scales.
        normalize_function: Choose 'max' or 'sum'
        '''
        # print(features_np.shape)
        # print(features_np)
        if normalize:
            row_sums = features_np.sum(axis=0) + 0.0001
            if normalize_function == 'max':
                row_sums = features_np.max(axis=0) + 0.0001
            # print(row_sums)
            new_matrix = features_np / row_sums[np.newaxis, :]
            features_np = new_matrix
            # print(features_np)
        tmp_features = features_np
        if dimensions == 'mle':
            if features_np.shape[1] > features_np.shape[0]:
                pca = PCA(n_components=features_np.shape[0], svd_solver="full")
                pca.fit(features_np)
                tmp_features = pca.transform(features_np)
        pca = PCA(n_components=dimensions, svd_solver="full")
        pca.fit(tmp_features)
        reduced_features = pca.transform(tmp_features)

        if reduced_features.shape[1] == 0:
            pca = PCA(n_components=1, svd_solver="full")
            pca.fit(tmp_features)
            reduced_features = pca.transform(tmp_features)
        # print("Original features: ", features_np.shape)
        # print("Reduced features shape: ", reduced_features.shape)
        return reduced_features


def extract_distinct_performance_instance_count(subgraphs_nodes, node_type, actor):
    results = []
    for df_subgraph in subgraphs_nodes:
        all_instances = []
        for index, row in df_subgraph.iterrows():
            if actor == "":
                all_instances.append(row[node_type])
            else:
                if row['actor'] == actor:
                    all_instances.append(row[node_type])
        results.append([(f'distinct_{node_type}_count', len(set(all_instances)))])
    return results


def extract_number_of_cases(task_subgraphs_nodes, actor):
    results = []
    for df_subgraph in task_subgraphs_nodes:
        all_cases = []
        for index, row in df_subgraph.iterrows():
            if actor == "":
                all_cases.append(row['case'])
            else:
                if row['actor'] == actor:
                    all_cases.append(row['case'])
        results.append([('case_count', len(set(all_cases)))])
    return results


def extract_count_per_performance_instance(subgraphs_nodes, node_type, actor):
    results = []
    for df_subgraph in subgraphs_nodes:
        count_per_instance = {}
        total_count = 0
        for index, row in df_subgraph.iterrows():
            if actor == "":
                if not row[node_type] in count_per_instance.keys():
                    count_per_instance[row[node_type]] = 0
                count_per_instance[row[node_type]] += 1
            else:
                if row['actor'] == actor:
                    if not row[node_type] in count_per_instance.keys():
                        count_per_instance[row[node_type]] = 0
                    count_per_instance[row[node_type]] += 1
        subgraph_results = [(f'{node_type}_{str(instance)}_freq', count_per_instance[instance]) for instance in
                            count_per_instance.keys()]
        results.append(subgraph_results)
    return results


def extract_count_per_performance_instance_normalized(subgraphs_nodes, node_type, actor):
    results = []
    for df_subgraph in subgraphs_nodes:
        count_per_instance = {}
        total_count = 0
        for index, row in df_subgraph.iterrows():
            if actor == "":
                if not row[node_type] in count_per_instance.keys():
                    count_per_instance[row[node_type]] = 0
                count_per_instance[row[node_type]] += 1
                total_count += 1
            else:
                if row['actor'] == actor:
                    if not row[node_type] in count_per_instance.keys():
                        count_per_instance[row[node_type]] = 0
                    count_per_instance[row[node_type]] += 1
                    total_count += 1
        subgraph_results = [
            (f'{node_type}_{str(instance)}_relative_freq', count_per_instance[instance] / total_count * 100) for
            instance in
            count_per_instance.keys()]
        results.append(subgraph_results)
    return results


def extract_total_performance_instance_count(subgraphs_nodes, node_type, actor):
    results = []
    for df_subgraph in subgraphs_nodes:
        all_instances = []
        for index, row in df_subgraph.iterrows():
            if actor == "":
                all_instances.append(row[node_type])
            else:
                if row['actor'] == actor:
                    all_instances.append(row[node_type])
        results.append([(f'total_{node_type}_count', len(all_instances))])
    return results


def extract_distinct_handover_count(subgraphs_edges, node_type, actor_1="", actor_2="", entity_type=""):
    results = []
    for df_subgraph in subgraphs_edges:
        if entity_type != "":
            df_subgraph = df_subgraph[df_subgraph["entity_type"] == entity_type]
        all_handovers = []
        for index, row in df_subgraph.iterrows():
            if actor_1 == "" and actor_2 == "":
                all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
            elif actor_1 == "":
                if row['actor_2'] == actor_2:
                    all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
            elif actor_2 == "":
                if row['actor_1'] == actor_1:
                    all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
            else:
                if row['actor_1'] == actor_1 and row['actor_2'] == actor_2:
                    all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
        results.append([(f"distinct_{node_type}_handover_count", len(set(all_handovers)))])
    return results


def extract_total_handover_count(subgraphs_edges, node_type, actor_1="", actor_2="", entity_type=""):
    results = []
    for df_subgraph in subgraphs_edges:
        if entity_type != "":
            df_subgraph = df_subgraph[df_subgraph["entity_type"] == entity_type]
        all_handovers = []
        for index, row in df_subgraph.iterrows():
            if actor_1 == "" and actor_2 == "":
                all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
            elif actor_1 == "":
                if row['actor_2'] == actor_2:
                    all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
            elif actor_2 == "":
                if row['actor_1'] == actor_1:
                    all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
            else:
                if row['actor_1'] == actor_1 and row['actor_2'] == actor_2:
                    all_handovers.append(f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}")
        results.append([(f"total_{node_type}_handover_count", len(all_handovers))])
    return results


def extract_count_per_handover(subgraphs_edges, node_type, actor_1="", actor_2="", entity_type=""):
    results = []
    for df_subgraph in subgraphs_edges:
        if entity_type != "":
            df_subgraph = df_subgraph[df_subgraph["entity_type"] == entity_type]
        count_per_handover = {}
        for index, row in df_subgraph.iterrows():
            if actor_1 == "" and actor_2 == "":
                if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
            elif actor_1 == "":
                if row['actor_2'] == actor_2:
                    if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                        count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
            elif actor_2 == "":
                if row['actor_1'] == actor_1:
                    if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                        count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
            else:
                if row['actor_1'] == actor_1 and row['actor_2'] == actor_2:
                    if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                        count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
        subgraph_results = [(f'{node_type}_{str(handover)}_count', count_per_handover[handover]) for handover in
                            count_per_handover.keys()]
        results.append(subgraph_results)
    return results


def extract_count_per_handover_normalized(subgraphs_edges, node_type, actor_1="", actor_2="", entity_type=""):
    results = []
    for df_subgraph in subgraphs_edges:
        if entity_type != "" and not df_subgraph.empty:
            df_subgraph = df_subgraph[df_subgraph["entity_type"] == entity_type]
        count_per_handover = {}
        total_count = 0
        for index, row in df_subgraph.iterrows():
            if actor_1 == "" and actor_2 == "":
                if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
                total_count += 1
            elif actor_1 == "":
                if row['actor_2'] == actor_2:
                    if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                        count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
                    total_count += 1
            elif actor_2 == "":
                if row['actor_1'] == actor_1:
                    if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                        count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
                    total_count += 1
            else:
                if row['actor_1'] == actor_1 and row['actor_2'] == actor_2:
                    if f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}" not in count_per_handover.keys():
                        count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] = 0
                    count_per_handover[f"{row[f'{node_type}_1']}_{row[f'{node_type}_2']}"] += 1
                    total_count += 1
        subgraph_results = [
            (f'{node_type}_{str(handover)}_relative_freq', count_per_handover[handover] / total_count * 100)
            for handover in count_per_handover.keys()]
        results.append(subgraph_results)
    return results


def extract_average_case_duration(durations):
    results = []
    for df_window_durations in durations:
        results.append([(f"average_case_duration", df_window_durations['duration'].mean() / 3600 / 24)])
    return results


def extract_case_throughput_speed(durations):
    results = []
    for df_window_durations in durations:
        average_duration_days = df_window_durations['duration'].mean() / 3600 / 24
        case_throughput_speed = len(df_window_durations.index) / average_duration_days
        results.append([(f"case_throughput_speed", case_throughput_speed)])
    return results


def extract_case_throughput_velocity(durations):
    results = []
    for df_window_durations in durations:
        average_duration_days = df_window_durations['duration'].mean() / 3600 / 24
        case_throughput_velocity = len(df_window_durations.index) / average_duration_days / average_duration_days
        results.append([(f"case_throughput_velocity", case_throughput_velocity)])
    return results
