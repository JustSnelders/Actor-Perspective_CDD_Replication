from promg import Query


class ConceptDriftDetectionTasksQueryLibrary:

    @staticmethod
    def q_get_start_timestamp():
        query_str = f'''
            MATCH (ti:TaskInstance)
            WITH min(ti.start_time) AS start_time
            RETURN start_time
            '''
        return Query(query_str=query_str)

    @staticmethod
    def q_get_end_timestamp():
        query_str = f'''
                MATCH (ti:TaskInstance)
                WITH max(ti.start_time) AS end_time
                RETURN end_time
                '''
        return Query(query_str=query_str)

    @staticmethod
    def q_retrieve_task_subgraph_nodes(start_date: str, end_date: str, case, resource, exclude_cluster=""):
        where_exclude_cluster = ""
        if not exclude_cluster == "":
            where_exclude_cluster = f"AND ti.cluster <> \"{exclude_cluster}\""
        query_str = f'''
            MATCH (r:$resource)<-[:CORR]-(ti:TaskInstance)-[:CORR]->(c:$case) 
                WHERE date("$start_date") <= date(ti.start_time) AND date(ti.end_time) <= date("$end_date") 
                AND ti.cluster IS NOT NULL $where_exclude_cluster
            WITH ti.cluster AS task, ti.ID AS task_variant, c.sysId AS case, r.sysId AS actor
            RETURN task, task_variant, case, actor
            '''

        return Query(query_str=query_str,
                     template_string_parameters={
                         "start_date": start_date,
                         "end_date": end_date,
                         "case": case.type,
                         "resource": resource.type,
                         "where_exclude_cluster": where_exclude_cluster
                     })

    @staticmethod
    def q_retrieve_event_subgraph_nodes(start_date: str, end_date: str, case, resource):
        query_str = '''
            MATCH (r:$resource)<-[:CORR]-(e:Event)-[:CORR]->(c:$case) 
                WHERE date("$start_date") <= date(e.timestamp) <= date("$end_date")
            WITH e.activity+'+'+e.lifecycle AS activity, c.sysId AS case, r.sysId AS actor
            RETURN activity, case, actor
            '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "start_date": start_date,
                         "end_date": end_date,
                         "case": case.type,
                         "resource": resource.type
                     })

    @staticmethod
    def q_retrieve_task_subgraph_edges(start_date: str, end_date: str, case, resource, exclude_cluster=""):
        where_exclude_cluster = ""
        if not exclude_cluster == "":
            where_exclude_cluster = f"AND ti1.cluster <> \"{exclude_cluster}\" AND ti2.cluster <> \"{exclude_cluster}\""
        query_str = f'''
                MATCH (ti1:TaskInstance)-[r:$df_ti_case|$df_ti_resource]->(ti2:TaskInstance) 
                    WHERE date("{start_date}") <= date(ti1.end_time) AND date(ti2.start_time) <= date("{end_date}") 
                    AND ti1.cluster IS NOT NULL AND ti2.cluster IS NOT NULL $where_exclude_cluster
                MATCH (r1:$resource_node_label)<-[:CORR]-(ti1)-[:CORR]->(c1:$case_node_label)
                MATCH (r2:$resource_node_label)<-[:CORR]-(ti2)-[:CORR]->(c2:$case_node_label)
                WITH ti1.cluster AS task_1, ti2.cluster AS task_2, ti1.ID AS task_variant_1, ti2.ID AS task_variant_2, 
                    c1.sysId AS case_1, c2.sysId AS case_2, r1.sysId AS actor_1, r2.sysId AS actor_2,
                    duration.inSeconds(ti1.end_time, ti2.start_time) AS duration, r.entityType AS entity_type
                RETURN task_1, task_2, task_variant_1, task_variant_2, case_1, case_2, actor_1, actor_2, duration, 
                    entity_type
                '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "start_date": start_date,
                         "end_date": end_date,
                         "case_node_label": case.type,
                         "resource_node_label": resource.type,
                         "df_ti_case": case.get_df_ti_label(),
                         "df_ti_resource": resource.get_df_ti_label(),
                         "where_exclude_cluster": where_exclude_cluster
                     })

    @staticmethod
    def q_retrieve_event_subgraph_edges(start_date: str, end_date: str, case, resource):
        query_str = f'''
                MATCH (e1:Event)-[r:$df_case|$df_resource]->(e2:Event) 
                    WHERE date("$start_date") <= date(e1.timestamp) 
                    AND date(e2.timestamp) <= date("$end_date")
                MATCH (r1:$resource_node_label)<-[:CORR]-(e1)-[:CORR]->(c1:$case_node_label)
                MATCH (r2:$resource_node_label)<-[:CORR]-(e2)-[:CORR]->(c2:$case_node_label)
                WITH e1.activity+'+'+e1.lifecycle AS activity_1, e2.activity+'+'+e2.lifecycle AS activity_2, 
                    c1.sysId AS case_1, c2.sysId AS case_2, r1.sysId AS actor_1, r2.sysId AS actor_2,
                    duration.inSeconds(e1.timestamp, e2.timestamp) AS duration, r.entityType AS entity_type
                RETURN activity_1, activity_2, case_1, case_2, actor_1, actor_2, duration, entity_type
                '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "start_date": start_date,
                         "end_date": end_date,
                         "case_node_label": case.type,
                         "resource_node_label": resource.type,
                         "df_case": case.get_df_label(),
                         "df_resource": resource.get_df_label()
                     })

    @staticmethod
    def q_retrieve_variants_in_cluster(aggregation_type: str, aggregation_id):
        query_str = f'''
            MATCH (ti:TaskInstance) WHERE ti.$aggregation_type = $aggregation_id
            RETURN DISTINCT ti.variant AS variant
            '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "aggregation_type": aggregation_type,
                         "aggregation_id": aggregation_id
                     })

    @staticmethod
    def q_retrieve_activity_pairs_in_cluster(aggregation_type: str, aggregation_id, case, resource):
        query_str = f'''
                MATCH (e1:Event)<-[:CONTAINS]-(ti:TaskInstance)-[:CONTAINS]->(e2:Event) WHERE ti.$aggregation_type = $aggregation_id
                AND (e1)-[:$df_case]->(e2) AND (e1)-[:$df_resource]->(e2)
                WITH e1.activity+'+'+e1.lifecycle AS action1, e2.activity+'+'+e2.lifecycle AS action2
                RETURN DISTINCT action1, action2
                '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "aggregation_type": aggregation_type,
                         "aggregation_id": aggregation_id,
                         "df_case": case.get_df_label(),
                         "df_resource": resource.get_df_label()
                     })

    @staticmethod
    def q_retrieve_activities_in_cluster(aggregation_type, aggregation_id):
        query_str = f'''
                MATCH (ti:TaskInstance)-[:CONTAINS]->(e:Event) WHERE ti.$aggregation_type = $aggregation_id
                WITH e.activity+'+'+e.lifecycle AS action
                RETURN DISTINCT action
                '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "aggregation_type": aggregation_type,
                         "aggregation_id": aggregation_id
                     })

    @staticmethod
    def q_retrieve_actor_list(resource, min_freq=0):
        query_str = f'''
            MATCH (ti:TaskInstance)-[:CORR]->(r:$resource_node_label) WHERE ti.cluster IS NOT NULL
            WITH DISTINCT r.sysId AS actor, COUNT(*) AS count WHERE count > $min_freq
            RETURN actor
            '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "min_freq": min_freq,
                         "resource_node_label": resource.type
                     })

    @staticmethod
    def q_retrieve_collab_list(resource, case, min_freq):
        # q = f'''
        #     MATCH (ti1:TaskInstance)-[:DF_TI]->(ti2:TaskInstance) WHERE ti1.rID <> ti2.rID AND ti1.rID <> "User_1"
        #         AND ti2.rID <> "User_1" AND ti1.cluster IS NOT NULL AND ti2.cluster IS NOT NULL
        #     WITH DISTINCT ti1.rID AS actor_1, ti2.rID AS actor_2, count(*) AS count WHERE count > {min_freq}
        #     RETURN actor_1, actor_2'''
        query_str = f'''
            MATCH (r1:$resource_node_label)<-[:CORR]-(ti1:TaskInstance)-[:$df_ti_case]->
                (ti2:TaskInstance)-[:CORR]->(r2:$resource_node_label) 
                WHERE r1.sysId <> r2.sysId AND ti1.cluster IS NOT NULL AND ti2.cluster IS NOT NULL
            WITH DISTINCT r1.sysId AS actor_1, r2.sysId AS actor_2, count(*) AS count WHERE count > $min_freq
            RETURN actor_1, actor_2'''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "min_freq": min_freq,
                         "resource_node_label": resource.type,
                         "df_ti_case": case.get_df_ti_label()
                     })

    @staticmethod
    def q_retrieve_case_durations(start_date, end_date, case):
        query_str = f'''
            MATCH (ti1:TaskInstance)-[:CORR]->(c:$case_node_label)<-[:CORR]-(ti2:TaskInstance)
                WHERE date("$start_date") <= date(ti2.end_time) AND date(ti1.start_time) <= date("$end_date")
                AND NOT (:TaskInstance)-[:$df_ti_case]->(ti1) 
                AND NOT (ti2)-[:$df_ti_case]->(:TaskInstance) 
            WITH duration.inSeconds(ti1.start_time, ti2.end_time) AS duration
            RETURN duration
            '''
        return Query(query_str=query_str,
                     template_string_parameters={
                         "start_date": start_date,
                         "end_date": end_date,
                         "case_node_label": case.type,
                         "df_ti_case": case.get_df_ti_label()
                     })
