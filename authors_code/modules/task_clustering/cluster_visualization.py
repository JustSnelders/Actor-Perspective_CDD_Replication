import os
import pandas as pd
import pydot
from graphviz import Digraph
from _collections import OrderedDict

from promg import DatabaseConnection

from modules.task_clustering.vis_configuration import VisualizationConfiguration

from queries.task_clustering import TaskClusteringQueryLibrary as ql
from queries import query_result_parser as qp


class ClusterVisualizer:

    def __init__(self, db_connection, dataset_name):
        self.connection = db_connection
        self.vc = VisualizationConfiguration()

        self.df_task_variants = qp.parse_to_dataframe(self.connection.exec_query(ql.q_retrieve_clusters))

        self.output_directory = f"output_final\\{dataset_name}\\cluster_visualizations"
        os.makedirs(self.output_directory, exist_ok=True)

    def visualize_clusters(self, print_variants_not_in_cluster=False):
        df_variants_clustered = self.df_task_variants[~self.df_task_variants['cluster'].isna()].copy()

        cluster_list = list(df_variants_clustered['cluster'].unique())
        for cluster in cluster_list:
            df_variants_in_cluster = df_variants_clustered[df_variants_clustered['cluster'] == cluster].copy()
            df_variants_in_cluster.sort_values(by=['ID', 'frequency'], ascending=[True, False], inplace=True)
            dot_grouped_variants = get_dot_grouped_variants_colored(df_variants_in_cluster,
                                                                    node_properties=self.vc.node_properties)
            (graph,) = pydot.graph_from_dot_data(dot_grouped_variants.source)
            graph.write_png(f"{self.output_directory}\\variants_in_cluster_{cluster}.png")

        if print_variants_not_in_cluster:
            output_directory_not_clustered = os.path.join(self.output_directory, "variants_not_in_cluster")
            os.makedirs(output_directory_not_clustered, exist_ok=True)

            df_variants_not_in_cluster = self.df_task_variants[self.df_task_variants['cluster'].isna()].copy()
            df_variants_not_in_cluster.reset_index(level=0, inplace=True)

            for index, row in df_variants_not_in_cluster.iterrows():
                variant = df_variants_not_in_cluster.loc[index, 'variant']
                variant_id = df_variants_not_in_cluster.loc[index, 'ID']
                dot_single_variant = get_dot_single_variant_colored(variant_id=variant_id, variant=variant,
                                                                    node_properties=self.vc.node_properties)
                (graph,) = pydot.graph_from_dot_data(dot_single_variant.source)
                graph.write_png(f"{output_directory_not_clustered}\\variant_{variant_id}.png")

        print_legend(self.vc.node_properties, self.output_directory)


def get_dot_grouped_variants_colored(df_grouped_variants, node_properties):
    dot = Digraph(comment='Query Result')
    dot.attr("graph", rankdir="LR", margin="0", nodesep="0.25", ranksep="0.05")
    dot.attr("node", fixedsize="true", fontname="Helvetica", fontsize="10", margin="0")
    for index, row in df_grouped_variants.iloc[::-1].iterrows():
        variant = row.variant
        node_id = 1
        variant_label = str(int(row['ID']))
        variant_frequency = str(int(row['frequency']))

        with dot.subgraph() as s:
            s.attr(newrank="True")
            s.node(str(index), f'{variant_label}\t{variant_frequency}\l', shape="rect", width="1.7", color="white",
                   penwidth=str(0.5))
            s.edge(str(index), f'{index}_{node_id}', style="invis")

            for pos, event in enumerate(variant[:-1]):
                s.node(f'{index}_{node_id}', "", style="filled",
                       shape=node_properties[event][0],
                       width=node_properties[event][1], height=node_properties[event][2],
                       fillcolor=node_properties[event][3], fontcolor="black", penwidth=str(0.5))
                s.edge(f'{index}_{node_id}', f'{index}_{node_id + 1}', style="invis")
                node_id += 1

            s.node(f'{index}_{node_id}', "", style="filled",
                   shape=node_properties[variant[-1]][0],
                   width=node_properties[variant[-1]][1], height=node_properties[variant[-1]][2],
                   fillcolor=node_properties[variant[-1]][3], fontcolor="black", penwidth=str(0.5))
    return dot


def get_dot_single_variant_colored(variant_id, variant, node_properties, print_rank=True):
    dot = Digraph(comment='Query Result')
    dot.attr("graph", rankdir="LR", margin="0", nodesep="0.25", ranksep="0.05")
    dot.attr("node", fixedsize="true", fontname="Helvetica", fontsize="11", margin="0")

    node_id = 1
    if print_rank:
        dot.node(str(variant_id), f'{variant_id}\l', shape="rect", width="0.3", color="white", penwidth=str(0.5))
        dot.edge(str(variant_id), f'{variant_id}_{node_id}', style="invis")

    for event in variant[:-1]:
        dot.node(f'{variant_id}_{node_id}', "", style="filled", shape=node_properties[event][0],
                 width=node_properties[event][1], height=node_properties[event][2],
                 fillcolor=node_properties[event][3], fontcolor="black", penwidth=str(0.5))
        dot.edge(f'{variant_id}_{node_id}', f'{variant_id}_{node_id + 1}', style="invis")
        node_id += 1
    dot.node(f'{variant_id}_{node_id}', "", style="filled", shape=node_properties[variant[-1]][0],
             width=node_properties[variant[-1]][1], height=node_properties[variant[-1]][2],
             fillcolor=node_properties[variant[-1]][3], fontcolor="black", penwidth=str(0.5))
    return dot


def print_legend(node_properties, output_directory):
    dot = Digraph(comment='Query Result')
    dot.attr("graph", rankdir="LR", margin="0", nodesep="0.05", ranksep="0.05")
    dot.attr("node", fixedsize="false", fontname="Helvetica", fontsize="12", margin="0")

    node_id = 1

    for key, value in OrderedDict(reversed(list(node_properties.items()))).items():
        with dot.subgraph() as s:
            s.attr(newrank="True")
            s.node(str(node_id), "", style="filled", shape=node_properties[key][0],
                   width=node_properties[key][1], height=node_properties[key][2],
                   fillcolor=node_properties[key][3], fontcolor="black", penwidth=str(0.5))
            s.edge(str(node_id), str(node_id + 1), style="invis")
            s.node(str(node_id + 1), f'{key}\l', shape="rect", width="2", color="white",
                   height=node_properties[key][2], fontcolor="black", penwidth=str(0.5))
            node_id += 2

    (graph,) = pydot.graph_from_dot_data(dot.source)
    graph.write_png(f"{output_directory}\\legend.png")
