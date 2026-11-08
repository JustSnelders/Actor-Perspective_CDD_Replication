from promg import Performance
from modules.custom_queries.discovery_dfg import DiscoverDFGQueryLibrary as ql


class DiscoverDFG:
    def __init__(self, db_connection, semantic_header):
        self.connection = db_connection
        self.semantic_header = semantic_header

    @Performance.track()
    def discover_dfg_for_entity(self, entity_str: str, df_threshold: int = 0, relative_df_threshold: float = 0.0):
        entity = self.semantic_header.get_entity(entity_type=entity_str)

        if relative_df_threshold == 0.0:
            self.connection.exec_query(ql.aggregate_df_relations,
                                       **{
                                           "corr_label": entity.get_corr_type_strings(),
                                           "df_label": entity.get_df_label(),
                                           "dfc_label": entity.get_df_a_label(),
                                           "df_threshold": df_threshold
                                       })
        else:
            self.connection.exec_query(ql.aggregate_df_relations_heuristic,
                                       **{
                                           "corr_label": entity.get_corr_type_strings(),
                                           "df_label": entity.get_df_label(),
                                           "dfc_label": entity.get_df_a_label(),
                                           "df_threshold": df_threshold,
                                           "relative_df_threshold": relative_df_threshold
                                       })
