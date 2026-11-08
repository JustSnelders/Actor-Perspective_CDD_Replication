from promg import DatabaseConnection, SemanticHeader
from promg import Performance
from modules.custom_queries.df_interactions import InferDFInteractionsQueryLibrary as ql


class InferDFInteractions:
    def __init__(self, db_connection, semantic_header):
        self.connection = db_connection
        self.semantic_header = semantic_header

    @Performance.track()
    def delete_parallel_directly_follows_derived(self, entity_str: str, original_entity_str: str):
        entity = self.semantic_header.get_entity(entity_type=entity_str)
        original_entity = self.semantic_header.get_entity(entity_type=original_entity_str)

        self.connection.exec_query(ql.delete_parallel_directly_follows_derived,
                                   **{
                                       "entity": entity,
                                       "original_entity": original_entity
                                   })
