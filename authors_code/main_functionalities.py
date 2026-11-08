from colorama import Fore
from promg import OcedPg, SemanticHeader, DatasetDescriptions
from promg.modules.db_management import DBManagement
from promg.modules.task_identification import TaskIdentification

from modules.task_clustering.cluster_visualization import ClusterVisualizer
from modules.task_clustering.task_clustering import TaskClustering
from modules.task_concept_drift_detection.concept_drift_analysis import ConceptDriftDetection
from modules.task_concept_drift_detection.concept_drift_evaluation import ConceptDriftEvaluation
from modules.task_identification.task_identification import TaskIdentification as TaskIdentificationLocal
from modules.custom_modules.delay_analysis import PerformanceAnalyzeDelays
from modules.custom_modules.df_interactions import InferDFInteractions
from modules.custom_modules.discover_dfg import DiscoverDFG


def clear_db(db_connection):
    print(Fore.RED + 'Clearing the database.' + Fore.RESET)

    db_manager = DBManagement(db_connection)
    db_manager.clear_db(replace=True)
    db_manager.set_constraints()


def load_data(db_connection, config):
    if config.use_preprocessed_files:
        print(Fore.RED + '💾 Preloaded files are used!' + Fore.RESET)
    else:
        print(Fore.RED + '📝 Importing and creating files' + Fore.RESET)

    semantic_header = SemanticHeader.create_semantic_header(config=config)
    dataset_descriptions = DatasetDescriptions(config=config)

    oced_pg = OcedPg(database_connection=db_connection,
                     semantic_header=semantic_header,
                     dataset_descriptions=dataset_descriptions,
                     store_files=False,
                     use_sample=config.use_sample,
                     use_preprocessed_files=config.use_preprocessed_files,
                     import_directory=config.import_directory)
    oced_pg.load()


def transform_data(db_connection,
                   config):
    dataset_descriptions = DatasetDescriptions(config=config)
    semantic_header = SemanticHeader.create_semantic_header(config=config)

    oced_pg = OcedPg(database_connection=db_connection,
                     semantic_header=semantic_header,
                     dataset_descriptions=dataset_descriptions,
                     store_files=False,
                     use_sample=config.use_sample,
                     use_preprocessed_files=config.use_preprocessed_files,
                     import_directory=config.import_directory)

    oced_pg.transform()
    oced_pg.create_df_edges()


def delete_parallel_df(db_connection, config):
    semantic_header = SemanticHeader.create_semantic_header(config=config)
    print(Fore.RED + 'Inferring DF over relations between objects.' + Fore.RESET)
    infer_df_interactions = InferDFInteractions(db_connection=db_connection, semantic_header=semantic_header)
    infer_df_interactions.delete_parallel_directly_follows_derived('CASE_AO', 'Application')
    infer_df_interactions.delete_parallel_directly_follows_derived('CASE_AO', 'Offer')
    infer_df_interactions.delete_parallel_directly_follows_derived('CASE_AW', 'Application')
    infer_df_interactions.delete_parallel_directly_follows_derived('CASE_AW', 'Workflow')
    infer_df_interactions.delete_parallel_directly_follows_derived('CASE_WO', 'Workflow')
    infer_df_interactions.delete_parallel_directly_follows_derived('CASE_WO', 'Offer')


def discover_model(db_connection, config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    print(Fore.RED + 'Discovering multi-object DFG.' + Fore.RESET)
    dfg = DiscoverDFG(db_connection=db_connection, semantic_header=semantic_header)
    dfg.discover_dfg_for_entity("Application", 25000, 0.0)
    dfg.discover_dfg_for_entity("Offer", 25000, 0.0)
    dfg.discover_dfg_for_entity("Workflow", 25000, 0.0)
    dfg.discover_dfg_for_entity("CASE_AO", 25000, 0.0)
    dfg.discover_dfg_for_entity("CASE_AW", 25000, 0.0)
    dfg.discover_dfg_for_entity("CASE_WO", 25000, 0.0)


def build_tasks(db_connection, config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    print(Fore.RED + 'Detecting tasks.' + Fore.RESET)
    task_identifier = TaskIdentification(
        db_connection=db_connection,
        semantic_header=semantic_header,
        resource="Resource",
        case="CaseAWO")
    task_identifier.identify_tasks()
    # task_identifier.aggregate_on_task_variant()
    task_identifier_local = TaskIdentificationLocal(db_connection=db_connection,
                                                    semantic_header=semantic_header,
                                                    resource="Resource",
                                                    case="CaseAWO")
    task_identifier_local.set_task_id()


def cluster_tasks(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    print(Fore.RED + 'Clustering tasks.' + Fore.RESET)
    task_clustering = TaskClustering(db_connection=db_connection,
                                     semantic_header=semantic_header,
                                     dataset_name=analysis_config.dataset_name,
                                     resource="Resource", case="CaseAWO",
                                     min_variant_freq=analysis_config.cluster_min_variant_freq)
    task_clustering.encode_and_cluster_specific(min_variant_length=analysis_config.cluster_min_variant_length,
                                                manual_clusters=analysis_config.manual_clusters,
                                                num_clusters=analysis_config.num_clusters,
                                                include_remainder=analysis_config.cluster_include_remainder,
                                                clustering_description=analysis_config.clustering_description)
    task_clustering.remove_clusters()
    task_clustering.construct_clusters()


def visualize_clusters_in_dot(db_connection, analysis_config):
    cluster_visualization = ClusterVisualizer(db_connection=db_connection,
                                              dataset_name=analysis_config.dataset_name)
    cluster_visualization.visualize_clusters()


def process_level_drift_detection(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    print(Fore.RED + 'Detecting process-level drift.' + Fore.RESET)
    cd_detection = ConceptDriftDetection(db_connection=db_connection,
                                         semantic_header=semantic_header,
                                         dataset_name=analysis_config.dataset_name,
                                         resource="Resource", case="CaseAWO")
    cd_detection.detect_process_level_drift(window_sizes=analysis_config.window_sizes,
                                            penalties=analysis_config.penalties,
                                            feature_sets=analysis_config.process_drift_feature_sets,
                                            exclude_cluster=analysis_config.leftover_cluster,
                                            plot_drift=False)


def actor_drift_detection(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    print(Fore.RED + 'Detecting actor drift.' + Fore.RESET)
    cd_detection = ConceptDriftDetection(db_connection=db_connection,
                                         semantic_header=semantic_header,
                                         dataset_name=analysis_config.dataset_name,
                                         resource="Resource",
                                         case="CaseAWO")
    cd_detection.detect_actor_drift(window_sizes=analysis_config.window_sizes,
                                    penalties=analysis_config.penalties,
                                    feature_sets=analysis_config.actor_drift_feature_sets,
                                    min_actor_freq=500, exclude_cluster=analysis_config.leftover_cluster,
                                    plot_drift=False)


def collab_drift_detection(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    print(Fore.RED + 'Detecting collab drift.' + Fore.RESET)
    cd_detection = ConceptDriftDetection(db_connection=db_connection,
                                         semantic_header=semantic_header,
                                         dataset_name=analysis_config.dataset_name,
                                         resource="Resource",
                                         case="CaseAWO")
    cd_detection.detect_collab_drift(window_sizes=analysis_config.window_sizes,
                                     penalties=analysis_config.penalties,
                                     min_collab_freq=300, detailed_analysis=True,
                                     exclude_cluster=analysis_config.leftover_cluster, plot_drift=False)


def cd_compare_subgroup_to_process_drift(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    cd_evaluation = ConceptDriftEvaluation(db_connection=db_connection,
                                           semantic_header=semantic_header,
                                           dataset_name=analysis_config.dataset_name,
                                           resource="Resource",
                                           case="CaseAWO")
    cd_evaluation.cd_compare_subgroup_to_process_drift(window_size=analysis_config.comp_window_size,
                                                       pen_process=analysis_config.comp_process_drift_penalty,
                                                       pen_subgroup=analysis_config.comp_subgroup_drift_penalty,
                                                       subgroup_type=analysis_config.comp_subgroup_to_compare,
                                                       feature_set_name_subgroup=analysis_config.comp_subgroup_feature_set_name,
                                                       feature_set_name_process_level=analysis_config.comp_feature_set_name_process_level)


def cd_calculate_magnitude_signal_changes(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    cd_evaluation = ConceptDriftEvaluation(db_connection=db_connection,
                                           semantic_header=semantic_header,
                                           dataset_name=analysis_config.dataset_name,
                                           resource="Resource",
                                           case="CaseAWO")
    cd_evaluation.calculate_magnitude_signal_changes(window_size=analysis_config.mc_window_size,
                                                     penalty=analysis_config.mc_penalty,
                                                     feature_sets=analysis_config.process_drift_feature_sets,
                                                     exclude_cluster=analysis_config.leftover_cluster)


def cd_calculate_overall_average_max_signal_change(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    cd_evaluation = ConceptDriftEvaluation(db_connection=db_connection,
                                           semantic_header=semantic_header,
                                           dataset_name=analysis_config.dataset_name,
                                           resource="Resource",
                                           case="CaseAWO")
    cd_evaluation.calculate_overall_average_max_signal_change(window_size=analysis_config.mc_window_size,
                                                              penalty=analysis_config.mc_penalty,
                                                              feature_sets=analysis_config.process_drift_feature_sets,
                                                              exclude_cluster=analysis_config.leftover_cluster)


def cd_compare_tasks_vs_activity_activity_pair(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    cd_evaluation = ConceptDriftEvaluation(db_connection=db_connection,
                                           semantic_header=semantic_header,
                                           dataset_name=analysis_config.dataset_name,
                                           resource="Resource",
                                           case="CaseAWO")
    cd_evaluation.compare_tasks_vs_activity_activity_pair(penalty=analysis_config.dc_penalty,
                                                          cp_task_dict=analysis_config.dc_task_dict)


def cd_compare_variant_vs_activity_activity_pair(db_connection, config, analysis_config):
    semantic_header = SemanticHeader.create_semantic_header(config)
    cd_evaluation = ConceptDriftEvaluation(db_connection=db_connection,
                                           semantic_header=semantic_header,
                                           dataset_name=analysis_config.dataset_name,
                                           resource="Resource",
                                           case="CaseAWO")
    cd_evaluation.compare_variant_vs_activity_activity_pair(penalty=analysis_config.dc_penalty,
                                                            cp_variant_dict=analysis_config.dc_variant_dict)


def infer_delays(db_connection):
    print(Fore.RED + 'Computing delay edges.' + Fore.RESET)
    delays = PerformanceAnalyzeDelays(db_connection)
    # delays.enrich_with_delay_edges()
    # delays.analyze_delays()
    delays.visualize_delays(10000)


def print_statistics(db_connection):
    db_manager = DBManagement(db_connection)
    db_manager.print_statistics()
