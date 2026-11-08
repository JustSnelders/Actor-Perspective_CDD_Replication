from datetime import datetime
from promg import DatabaseConnection
from promg import Configuration
from promg import Performance

from main_functionalities import clear_db, load_data, transform_data, build_tasks, \
    print_statistics, cluster_tasks, visualize_clusters_in_dot, process_level_drift_detection, \
    actor_drift_detection, collab_drift_detection, cd_compare_subgroup_to_process_drift, \
    cd_compare_tasks_vs_activity_activity_pair, cd_compare_variant_vs_activity_activity_pair, \
    cd_calculate_magnitude_signal_changes, cd_calculate_overall_average_max_signal_change
from analysis_configuration import AnalysisConfiguration

analysis_config = AnalysisConfiguration()


# steps for task clustering, each can be switched on/off
step_cluster_tasks = False
step_visualize_clusters = False

# steps for concept drift detection, each can be switched on/off
step_process_level_drift_detection = False
step_actor_drift_detection = False
step_collab_drift_detection = True

# steps for evaluation, can only be turned is the above steps have run
step_compare_to_process_drift = False
step_calculate_magnitude_signal_changes = False
step_calculate_overall_average_max_signal_change = False
step_detailed_change_signal_analysis = False


def main(config,
         step_clear_db=True,
         step_populate_graph=True,
         step_build_tasks=True,
         ) -> None:
    """
    Main function, read all the logs, clear and create the graph, perform checks
    @return: None
    """
    print("Started at =", datetime.now().strftime("%H:%M:%S"))

    db_connection = DatabaseConnection.set_up_connection(config=config)
    performance = Performance.set_up_performance(config=config)

    if step_clear_db:
        clear_db(db_connection)

    if step_populate_graph:
        load_data(db_connection=db_connection,
                  config=config)
        transform_data(db_connection=db_connection,
                       config=config)

    if step_build_tasks:
        build_tasks(db_connection=db_connection, config=config)

    if step_cluster_tasks:
        cluster_tasks(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_visualize_clusters:
        visualize_clusters_in_dot(db_connection=db_connection, analysis_config=analysis_config)

    if step_process_level_drift_detection:
        process_level_drift_detection(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_actor_drift_detection:
        actor_drift_detection(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_collab_drift_detection:
        collab_drift_detection(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_compare_to_process_drift:
        cd_compare_subgroup_to_process_drift(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_calculate_magnitude_signal_changes:
        cd_calculate_magnitude_signal_changes(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_calculate_overall_average_max_signal_change:
        cd_calculate_overall_average_max_signal_change(db_connection=db_connection, config=config, analysis_config=analysis_config)

    if step_detailed_change_signal_analysis:
        cd_compare_tasks_vs_activity_activity_pair(db_connection=db_connection, config=config, analysis_config=analysis_config)
        cd_compare_variant_vs_activity_activity_pair(db_connection=db_connection, config=config, analysis_config=analysis_config)

    performance.finish_and_save()
    print_statistics(db_connection)

    db_connection.close_connection()


if __name__ == "__main__":
    main(config=Configuration.init_conf_with_config_file(),
         step_clear_db=False,
         step_populate_graph=False,
         step_build_tasks=False)
