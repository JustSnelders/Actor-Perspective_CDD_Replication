from promg import Query


class TaskClusteringQueryLibrary:

    @staticmethod
    def q_retrieve_variants(min_variant_freq: int):
        query_str = '''
            MATCH (ti:TaskInstance)
            WITH ti.variant AS variant, ti.ID AS ID, size(ti.variant) AS variant_length
            WITH DISTINCT variant, variant_length, ID, COUNT (*) AS frequency WHERE frequency >= $min_variant_freq
            RETURN variant, variant_length, ID, frequency
            '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "min_variant_freq": min_variant_freq
                     })

    @staticmethod
    def q_retrieve_clusters():
        query_str = '''
                MATCH (ti:TaskInstance)
                WITH ti.variant AS variant, ti.ID AS ID, ti.cluster AS cluster
                WITH DISTINCT variant, ID, cluster, COUNT (*) AS frequency
                RETURN cluster, ID, variant, frequency
                '''
        return Query(query_str=query_str)

    @staticmethod
    def q_write_clusters_to_task_instance(variant: list, cluster: str):
        query_str = '''
            CALL apoc.periodic.iterate(
            "MATCH (ti:TaskInstance) WHERE ti.variant = $variant
             RETURN ti",
            "WITH ti
             SET ti.cluster = '$cluster'",
             {batchSize: $batch_size})'''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "variant": variant,
                         "cluster": cluster
                     })

    @staticmethod
    def q_create_cluster_nodes():
        query_str = '''
            CALL apoc.periodic.iterate(
            'MATCH (ti:TaskInstance) WHERE ti.cluster IS NOT NULL 
             RETURN DISTINCT ti.cluster AS cluster, count(*) AS cluster_count',
            'WITH cluster, cluster_count
             MERGE (tc:TaskCluster {Name:cluster, count:cluster_count})',
             {batchSize: $batch_size})'''
        return Query(query_str=query_str)

    @staticmethod
    def q_link_task_instances_to_clusters():
        query_str = '''
        CALL apoc.periodic.iterate(
        'MATCH (tc:TaskCluster)
         MATCH (ti:TaskInstance) WHERE ti.cluster = tc.Name
         RETURN tc, ti',
        'WITH tc, ti
         CREATE (ti)-[:OBSERVED]->(tc)',
         {batchSize: $batch_size})'''
        return Query(query_str=query_str)

    @staticmethod
    def q_remove_cluster_nodes():
        query_str = '''
            CALL apoc.periodic.iterate(
            'MATCH (tc:TaskCluster)
             RETURN tc',
            'WITH tc
             DETACH DELETE tc',
            {batchSize: $batch_size})'''
        return Query(query_str=query_str)

    @staticmethod
    def q_remove_cluster_property():
        query_str = '''
            CALL apoc.periodic.iterate(
            'MATCH (ti:TaskInstance)
             RETURN ti',
            'WITH ti
             REMOVE ti.cluster',
            {batchSize: $batch_size})'''
        return Query(query_str=query_str)
