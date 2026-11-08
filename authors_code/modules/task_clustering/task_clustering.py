import os
import pandas as pd
from os import path
from sklearn.cluster import AgglomerativeClustering

from promg import DatabaseConnection
from promg.data_managers.semantic_header import ConstructedNodes, SemanticHeader

from modules.task_clustering import variant_encoder_selection
from queries.task_clustering import TaskClusteringQueryLibrary as ql
from queries import query_result_parser as qp


class TaskClustering:

    def __init__(self, db_connection, semantic_header, dataset_name, resource: str, case: str, min_variant_freq: int):
        self.connection = db_connection
        self.dataset_name = dataset_name
        self.resource: ConstructedNodes = semantic_header.get_entity(resource)
        self.case: ConstructedNodes = semantic_header.get_entity(case)

        self.min_variant_freq = min_variant_freq
        self.metric = "euclidean"
        self.linkage = "ward"

        self.intermediate_output_directory = f"output_intermediate\\{dataset_name}\\task_clustering"
        os.makedirs(self.intermediate_output_directory, exist_ok=True)

        if path.exists(f"{self.intermediate_output_directory}\\variants_{dataset_name}_F{min_variant_freq}.pkl"):
            self.df_variants = pd.read_pickle(
                f"{self.intermediate_output_directory}\\variants_{dataset_name}_F{min_variant_freq}.pkl")
        else:
            self.df_variants = qp.parse_to_dataframe(
                self.connection.exec_query(ql.q_retrieve_variants, **{"min_variant_freq": min_variant_freq}))
            self.df_variants.to_pickle(
                f"{self.intermediate_output_directory}\\variants_{dataset_name}_F{min_variant_freq}.pkl")

    # TODO: Add regular encode_and_cluster_function

    def encode_and_cluster_specific(self, min_variant_length, manual_clusters, num_clusters,
                                    include_remainder, clustering_description):
        if path.exists(
                f"{self.intermediate_output_directory}\\variants_clustered_{self.dataset_name}_{clustering_description}.pkl"):
            self.df_variants_clustered = pd.read_pickle(
                f"{self.intermediate_output_directory}\\variants_clustered_{self.dataset_name}_{clustering_description}.pkl")
        else:
            # encode variants
            e = variant_encoder_selection.get_variant_encoder(self.dataset_name)
            manually_clustered_variants = [item for sublist in list(manual_clusters.values()) for item in sublist]
            df_variants_to_cluster = self.df_variants[self.df_variants['variant_length'] >= min_variant_length]
            df_variants_encoded = e.encode(df_variants_to_cluster)

            # cluster encoded variants
            df_clusters = cluster(df_variants_encoded, num_clusters, self.metric, self.linkage)
            self.df_variants_clustered = pd.concat([self.df_variants, df_clusters], axis=1)

            # fix labeling of variants that were not considered in the clustering
            self.df_variants_clustered['cluster'] = self.df_variants_clustered['cluster'].astype(str)
            A_id = 1
            for index, row in self.df_variants_clustered.iterrows():
                if row['ID'] in manually_clustered_variants:
                    variant_key = \
                        [key for key, corresponding_list in manual_clusters.items() if row['ID'] in corresponding_list][
                            0]
                    self.df_variants_clustered.loc[index, 'cluster'] = variant_key
                elif pd.isna(row['cluster']):
                    if include_remainder:
                        if row['variant_length'] == 1:
                            self.df_variants_clustered.loc[index, 'cluster'] = f"A{str(A_id).zfill(2)}"
                            A_id += 1
                else:
                    self.df_variants_clustered.loc[index, 'cluster'] = f"T{str(int(row['cluster'])).zfill(2)}"

            os.makedirs(self.intermediate_output_directory, exist_ok=True)
            self.df_variants_clustered.to_pickle(
                f"{self.intermediate_output_directory}\\variants_clustered_{self.dataset_name}_{clustering_description}.pkl")
        return self.df_variants_clustered

    def construct_clusters(self):
        for index, row in self.df_variants_clustered.iterrows():
            if not pd.isna(row['cluster']):
                self.connection.exec_query(
                    ql.q_write_clusters_to_task_instance, **{
                        "variant": row['variant'],
                        "cluster": row['cluster']})
        self.connection.exec_query(ql.q_create_cluster_nodes)
        self.connection.exec_query(ql.q_link_task_instances_to_clusters)

    def remove_clusters(self):
        self.connection.exec_query(ql.q_remove_cluster_nodes)
        self.connection.exec_query(ql.q_remove_cluster_property)


def cluster(df_variants_encoded, num_clusters, metric, linkage):
    clusters = AgglomerativeClustering(n_clusters=num_clusters, metric=metric, linkage=linkage) \
        .fit_predict(df_variants_encoded.values)
    index = list(df_variants_encoded.index.values)
    df_clusters = pd.DataFrame(index=index, data={'cluster': clusters})
    return df_clusters
