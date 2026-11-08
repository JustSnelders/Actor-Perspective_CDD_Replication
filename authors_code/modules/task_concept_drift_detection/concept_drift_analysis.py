import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from itertools import product
from os import path
import os
import glob

from promg import DatabaseConnection
from promg.data_managers.semantic_header import ConstructedNodes, SemanticHeader

from queries.task_cd_detection import ConceptDriftDetectionTasksQueryLibrary as ql
from modules.task_concept_drift_detection.feature_extraction import FeatureExtraction
from modules.task_concept_drift_detection import change_point_detection
from modules.task_concept_drift_detection import change_point_visualization
from queries import query_result_parser as qp


def get_feature_extractor_objects(db_connection, dataset_name: str, case: ConstructedNodes, resource: ConstructedNodes,
                                  feature_names: list, window_sizes: list, exclude_cluster: str = ""):
    list_f_extr = []
    for window_size in window_sizes:
        f_extr = FeatureExtraction(db_connection=db_connection, dataset_name=dataset_name, case=case, actor=resource,
                                   exclude_cluster=exclude_cluster)
        f_extr.retrieve_subgraphs_for_feature_extraction(window_size=window_size, feature_set=feature_names)
        list_f_extr.append(f_extr)
    return list_f_extr


def strip_inactive_windows(features):
    index_where_inactive = list(np.where(np.all(features == 0, axis=1)))
    original_index = list(range(0, len(features)))
    original_index_stripped = sorted(np.setdiff1d(original_index, index_where_inactive))
    features_stripped = features[np.where(~np.all(features == 0, axis=1))]
    return features_stripped, original_index_stripped, index_where_inactive[0]


def strip_inactive_windows_collab(features, actor_1_activity, actor_2_activity):
    index_where_inactive_actor_1 = list(np.where(np.all(actor_1_activity == 0, axis=1)))
    index_where_inactive_actor_2 = list(np.where(np.all(actor_2_activity == 0, axis=1)))
    list_either_inactive = np.concatenate([index_where_inactive_actor_1, index_where_inactive_actor_2],
                                          axis=1).flatten()
    index_where_either_inactive = sorted(list(set(list_either_inactive)))
    original_index = list(range(0, len(features)))
    original_index_stripped = sorted(np.setdiff1d(original_index, index_where_either_inactive))
    features_stripped = np.delete(features, index_where_either_inactive, 0)
    return features_stripped, original_index_stripped, index_where_either_inactive


def retrieve_original_cps(cps, original_index_stripped):
    actual_cps = []
    for cp in cps:
        actual_cps.append(original_index_stripped[cp])
    return actual_cps


def remove_duplicate_collab_pairs(collab_pairs):
    collab_pairs_distinct = []
    for collab_pair in collab_pairs:
        sorted_collab_pair = sorted(collab_pair)
        if sorted_collab_pair not in collab_pairs_distinct:
            collab_pairs_distinct.append(sorted_collab_pair)
    return collab_pairs_distinct


def split_consecutive(array, step_size=1):
    if np.any(array):
        idx = np.r_[0, np.where(np.diff(array) != step_size)[0] + 1, len(array)]
        list_consecutive = [array[i:j] for i, j in zip(idx, idx[1:])]
        return list_consecutive
    else:
        return None


class ConceptDriftDetection:
    def __init__(self, db_connection, semantic_header, dataset_name, resource: str, case: str):
        self.connection = db_connection
        self.dataset_name = dataset_name
        self.output_directory = f"output_final\\{dataset_name}\\task_concept_drift_detection"
        self.resource: ConstructedNodes = semantic_header.get_entity(resource)
        self.case: ConstructedNodes = semantic_header.get_entity(case)

    def detect_process_level_drift(self, window_sizes, penalties, feature_sets, exclude_cluster,
                                   plot_drift=False):
        all_features = [item for sublist in list(feature_sets.values()) for item in sublist]
        list_f_extr = get_feature_extractor_objects(db_connection=self.connection,
                                                    dataset_name=self.dataset_name, case=self.case,
                                                    resource=self.resource, feature_names=all_features,
                                                    window_sizes=window_sizes, exclude_cluster=exclude_cluster)

        # create output directory for process drift detection
        process_level_drift_directory = os.path.join(self.output_directory, "process_level_drift")
        os.makedirs(process_level_drift_directory, exist_ok=True)

        # initialize dataframe to store change points
        cp_settings = ["{}_{}".format(a_, b_) for a_, b_ in product(window_sizes, penalties)]
        indices = [fs_name for fs_name, _ in feature_sets.items()]
        df_process_level_drift_points = pd.DataFrame(index=indices, columns=cp_settings)
        if path.exists(f"{process_level_drift_directory}\\cps_all_features.csv"):
            df_process_level_drift_points_old = pd.read_csv(f"{process_level_drift_directory}\\cps_all_features.csv",
                                                            index_col=0)
            df_process_level_drift_points = pd.concat(
                [df_process_level_drift_points, df_process_level_drift_points_old],
                ignore_index=False).dropna(how='all').sort_index(axis=1)

        for feature_set_name, feature_list in feature_sets.items():
            df_process_level_feature_drift_points = pd.DataFrame(index=[feature_set_name], columns=cp_settings)
            if path.exists(f"{process_level_drift_directory}\\{feature_set_name}\\cps_{feature_set_name}.csv"):
                df_process_level_feature_drift_points_old = pd.read_csv(
                    f"{process_level_drift_directory}\\{feature_set_name}\\cps_{feature_set_name}.csv", index_col=0)
                df_process_level_feature_drift_points = pd.concat(
                    [df_process_level_feature_drift_points, df_process_level_feature_drift_points_old],
                    ignore_index=False).dropna(how='all').sort_index(axis=1)

            # create output directory for specified feature set
            process_level_drift_feature_directory = os.path.join(process_level_drift_directory, feature_set_name)
            os.makedirs(process_level_drift_feature_directory, exist_ok=True)

            # initialize dictionary to store change points for specified feature set
            dict_process_cps = {}
            for index, window_size in enumerate(window_sizes):
                feature_names, feature_vector = list_f_extr[index].apply_feature_extraction(feature_list)
                reduced_feature_vector = list_f_extr[index].pca_reduction(feature_vector, 'mle', normalize=True,
                                                                          normalize_function="max")
                for pen in penalties:
                    # detect change points for specified penalty and write to dataframe
                    cp = change_point_detection.rpt_pelt(reduced_feature_vector, pen=pen)
                    dict_process_cps[pen] = cp
                    # print(f"Change points {feature_set_name}: {cp}")
                    df_process_level_drift_points.loc[feature_set_name, f"{window_size}_{pen}"] = str(cp)
                    df_process_level_feature_drift_points.loc[feature_set_name, f"{window_size}_{pen}"] = str(cp)
                if plot_drift:
                    change_point_visualization.plot_trends(np_feature_vectors=feature_vector,
                                                           feature_names=feature_names, window_size=window_size,
                                                           output_directory=process_level_drift_feature_directory,
                                                           dict_change_points=dict_process_cps, min_freq=5)
            df_process_level_feature_drift_points.to_csv(
                f"{process_level_drift_directory}\\{feature_set_name}\\cps_{feature_set_name}.csv")
        df_process_level_drift_points.to_csv(f"{process_level_drift_directory}\\cps_all_features.csv")

    def detect_actor_drift(self, window_sizes, penalties, feature_sets, min_actor_freq, exclude_cluster,
                           plot_drift=False):
        actor_list = qp.parse_to_list(self.connection.exec_query(ql.q_retrieve_actor_list, **{
            "resource": self.resource, "min_freq": min_actor_freq}), "actor")

        all_features = [item for sublist in list(feature_sets.values()) for item in sublist]
        list_f_extr = get_feature_extractor_objects(db_connection=self.connection,
                                                    dataset_name=self.dataset_name,
                                                    case=self.case,
                                                    resource=self.resource,
                                                    feature_names=all_features,
                                                    window_sizes=window_sizes,
                                                    exclude_cluster=exclude_cluster)

        for feature_set_name, feature_list in feature_sets.items():
            print(f"Feature set: {feature_set_name}")

            # create output directory for (actor drift detection X specified feature set)
            actor_drift_feature_directory = os.path.join(self.output_directory, f"actor_drift\\{feature_set_name}")
            os.makedirs(actor_drift_feature_directory, exist_ok=True)

            # initialize dataframe to store change points for specified feature set
            cp_settings = ["{}_{}".format(a_, b_) for a_, b_ in product(window_sizes, penalties)]
            df_all_actor_drift_points = pd.DataFrame(index=actor_list, columns=cp_settings)
            if path.exists(f"{actor_drift_feature_directory}\\all_actor_cps_{feature_set_name}.csv"):
                df_all_actor_drift_points_old = pd.read_csv(
                    f"{actor_drift_feature_directory}\\all_actor_cps_{feature_set_name}.csv",
                    index_col=0)
                df_all_actor_drift_points = pd.concat(
                    [df_all_actor_drift_points, df_all_actor_drift_points_old], ignore_index=False).dropna(
                    how='all').sort_index(axis=1)

            print(f"Detecting change points for {len(actor_list)} actors...")
            for actor in tqdm(actor_list):
                # create directory to store actor-specific change points (and plots)
                actor_drift_feature_subdirectory = os.path.join(actor_drift_feature_directory, actor)
                os.makedirs(actor_drift_feature_subdirectory, exist_ok=True)

                df_actor_drift_points = pd.DataFrame(index=[actor], columns=cp_settings)
                if path.exists(f"{actor_drift_feature_directory}\\{actor}\\{actor}_cps_{feature_set_name}.csv"):
                    df_actor_drift_points_old = pd.read_csv(
                        f"{actor_drift_feature_directory}\\{actor}\\{actor}_cps_{feature_set_name}.csv",
                        index_col=0)
                    df_actor_drift_points = pd.concat(
                        [df_actor_drift_points, df_actor_drift_points_old], ignore_index=False).dropna(
                        how='all').sort_index(axis=1)

                # initialize dictionary to store change points for specified feature set
                dict_actor_cps = {}
                for index, window_size in enumerate(window_sizes):
                    # generate mv time series for specified features/actor/window size
                    actor_feature_names, actor_feature_vector = list_f_extr[index].apply_feature_extraction(
                        feature_list, actor=actor, actor_1=actor, actor_2=actor)
                    actor_feature_vector_stripped, time_window_mapping, windows_inactive = strip_inactive_windows(
                        actor_feature_vector)
                    reduced_actor_feature_vector = list_f_extr[index].pca_reduction(actor_feature_vector_stripped,
                                                                                    'mle',
                                                                                    normalize=True,
                                                                                    normalize_function="max")
                    for pen in penalties:
                        # detect change points for specified penalty and write to dataframe
                        cp = change_point_detection.rpt_pelt(reduced_actor_feature_vector, pen=pen)
                        # print(f"Change points {actor} {feature_set[1]} (pen={pen}): {cp}")
                        cp = retrieve_original_cps(cp, time_window_mapping)
                        df_all_actor_drift_points.loc[actor, f"{window_size}_{pen}"] = str(cp)
                        df_actor_drift_points.loc[actor, f"{window_size}_{pen}"] = str(cp)
                        dict_actor_cps[pen] = cp
                    if plot_drift:
                        change_point_visualization.plot_trends(
                            np_feature_vectors=actor_feature_vector, feature_names=actor_feature_names,
                            window_size=window_size, output_directory=actor_drift_feature_subdirectory,
                            dict_change_points=dict_actor_cps, subgroup=actor, min_freq=5,
                            highlight_ranges=split_consecutive(windows_inactive))
                df_actor_drift_points.to_csv(
                    f"{actor_drift_feature_directory}\\{actor}\\{actor}_cps_{feature_set_name}.csv")
            df_all_actor_drift_points.to_csv(f"{actor_drift_feature_directory}\\all_actor_cps_{feature_set_name}.csv")

    def detect_collab_drift(self, window_sizes, penalties, min_collab_freq, detailed_analysis,
                            exclude_cluster, plot_drift=False):
        collab_list = qp.parse_to_2d_list(self.connection.exec_query(ql.q_retrieve_collab_list, **{
            "resource": self.resource, "case": self.case, "min_freq": min_collab_freq}),
                                          "actor_1", "actor_2")
        # collab_list = [["User_1", "User_2"]]
        collab_pairs_distinct = remove_duplicate_collab_pairs(collab_list)
        if detailed_analysis:
            feature_sets = {"task_handovers_case": ["count_per_task_handover_case"]}
        else:
            feature_sets = {"total_task_handovers_case": ["total_task_handover_count_case"]}

        all_features = [item for sublist in list(feature_sets.values()) for item in sublist]
        all_features += ["total_task_count"]
        list_f_extr = get_feature_extractor_objects(db_connection=self.connection,
                                                    dataset_name=self.dataset_name,
                                                    case=self.case, resource=self.resource,
                                                    feature_names=all_features,
                                                    window_sizes=window_sizes,
                                                    exclude_cluster=exclude_cluster)

        # set up indices and columns for dataframes
        cp_settings = ["{}_{}".format(a_, b_) for a_, b_ in product(window_sizes, penalties)]
        index_collab_pairs = [f"{pair[0]}_{pair[1]}" for pair in collab_pairs_distinct]

        # retrieve activity per time window for all actors in collab_pairs
        list_actors = [actor for collab_pair in collab_pairs_distinct for actor in collab_pair]
        dicts_actor_activity_per_ws = []
        for i, ws in enumerate(window_sizes):
            dict_actor_activity = {}
            for a in list_actors:
                _, actor_activity = list_f_extr[i].apply_feature_extraction(["total_task_count"], actor=a)
                dict_actor_activity[a] = actor_activity
            dicts_actor_activity_per_ws.append(dict_actor_activity)

        for feature_set_name, feature_list in feature_sets.items():
            print(f"Feature set: {feature_set_name}")

            # create analysis directory for (collab drift detection X specified feature set)
            collab_drift_feature_directory = os.path.join(self.output_directory, f"collab_drift\\{feature_set_name}")
            os.makedirs(collab_drift_feature_directory, exist_ok=True)

            # initialize dataframe to store change points for specified feature set
            df_all_collab_drift_points = pd.DataFrame(index=index_collab_pairs, columns=cp_settings)
            if path.exists(f"{collab_drift_feature_directory}\\collab_cp_{feature_set_name}.csv"):
                df_all_collab_drift_points_old = pd.read_csv(
                    f"{collab_drift_feature_directory}\\collab_cp_{feature_set_name}.csv",
                    index_col=0)
                df_all_collab_drift_points = pd.concat(
                    [df_all_collab_drift_points, df_all_collab_drift_points_old], ignore_index=False).dropna(
                    how='all').sort_index(axis=1)

            for collab_pair in tqdm(collab_pairs_distinct):
                print(collab_pair)

                # create analysis directory for actor to plot
                collab_drift_feature_subdirectory = os.path.join(collab_drift_feature_directory,
                                                                 f"{collab_pair[0]}_{collab_pair[1]}")
                os.makedirs(collab_drift_feature_subdirectory, exist_ok=True)

                collab_pair_reversed = collab_pair[::-1]
                df_collab_drift_points = pd.DataFrame(index=[f"{collab_pair[0]}_{collab_pair[1]}"], columns=cp_settings)
                if path.exists(
                        f"{collab_drift_feature_directory}\\{collab_pair[0]}_{collab_pair[1]}\\{collab_pair[0]}_{collab_pair[1]}_cps_{feature_set_name}.csv"):
                    df_collab_drift_points_old = pd.read_csv(
                        f"{collab_drift_feature_directory}\\{collab_pair[0]}_{collab_pair[1]}\\{collab_pair[0]}_{collab_pair[1]}_cps_{feature_set_name}.csv",
                        index_col=0)
                    df_collab_drift_points = pd.concat(
                        [df_collab_drift_points, df_collab_drift_points_old], ignore_index=False).dropna(
                        how='all').sort_index(axis=1)

                dict_collab_cps = {}
                for index, window_size in enumerate(window_sizes):
                    actor_1_activity = dicts_actor_activity_per_ws[index][collab_pair[0]]
                    actor_2_activity = dicts_actor_activity_per_ws[index][collab_pair[1]]

                    # generate mv time series for specified features/collab_pair/window size
                    collab_feature_names, collab_feature_vector = list_f_extr[index].apply_feature_extraction(
                        feature_list, actor_1=collab_pair[0], actor_2=collab_pair[1])
                    collab_feature_names = [f"dir1_{f_name}" for f_name in collab_feature_names]
                    collab_reverse_feature_names, collab_reverse_feature_vector = list_f_extr[
                        index].apply_feature_extraction(
                        feature_list, actor_1=collab_pair_reversed[0], actor_2=collab_pair_reversed[1])
                    collab_reverse_feature_names = [f"dir2_{f_name}" for f_name in collab_reverse_feature_names]
                    collab_total_feature_vector = np.concatenate((collab_feature_vector, collab_reverse_feature_vector),
                                                                 axis=1)
                    collab_total_feature_vector_stripped, time_window_mapping, windows_inactive = \
                        strip_inactive_windows_collab(collab_total_feature_vector, actor_1_activity, actor_2_activity)

                    collab_all_feature_names = collab_feature_names + collab_reverse_feature_names
                    reduced_collab_feature_vector = list_f_extr[index].pca_reduction(
                        collab_total_feature_vector_stripped,
                        'mle', normalize=True,
                        normalize_function="max")
                    for pen in penalties:
                        # detect change points for specified penalty and write to dataframe
                        cp = change_point_detection.rpt_pelt(reduced_collab_feature_vector, pen=pen)
                        cp = retrieve_original_cps(cp, time_window_mapping)
                        dict_collab_cps[pen] = cp
                        print(
                            f"Change points {collab_pair[0]}_{collab_pair[1]} {feature_set_name} (pen={pen}): {cp}")
                        df_all_collab_drift_points.loc[
                            f"{collab_pair[0]}_{collab_pair[1]}", f"{window_size}_{pen}"] = str(
                            cp)
                        df_collab_drift_points.loc[f"{collab_pair[0]}_{collab_pair[1]}", f"{window_size}_{pen}"] = str(
                            cp)
                    if plot_drift:
                        change_point_visualization.plot_trends(
                            np_feature_vectors=collab_total_feature_vector, feature_names=collab_all_feature_names,
                            window_size=window_size, output_directory=collab_drift_feature_subdirectory,
                            dict_change_points=dict_collab_cps, subgroup=f"{collab_pair[0]}_{collab_pair[1]}",
                            min_freq=10,
                            highlight_ranges=split_consecutive(windows_inactive))
                # df_collab_drift_points.to_csv(
                #     f"{collab_drift_feature_directory}\\{collab_pair[0]}_{collab_pair[1]}\\{collab_pair[0]}_{collab_pair[1]}_cps_{feature_set_name}.csv")
            df_all_collab_drift_points.to_csv(f"{collab_drift_feature_directory}\\collab_cp_{feature_set_name}.csv")
