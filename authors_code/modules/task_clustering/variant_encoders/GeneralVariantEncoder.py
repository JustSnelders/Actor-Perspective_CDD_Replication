import pandas as pd


class GeneralEncoder:

    def __init__(self):
        self.encoding_description = "general"

    def encode(self, df_task_variants):
        df_task_variants_encoded = pd.get_dummies(
            pd.DataFrame(df_task_variants['path'].tolist(), index=df_task_variants.index).stack()).groupby(
            level=0).sum().copy()
        for (column_name, column_data) in df_task_variants_encoded.items():
            df_task_variants_encoded.loc[df_task_variants_encoded[column_name] > 1, column_name] = 1
        return df_task_variants_encoded

    def get_encoding_description(self):
        return self.encoding_description


def merge_events(df_encoded_tasks, event):
    if len(event[0]) == 3:
        df_encoded_tasks[event[1]] = df_encoded_tasks[event[0][0]] + df_encoded_tasks[event[0][1]] + \
                                     df_encoded_tasks[event[0][2]]
        df_encoded_tasks.drop([event[0][0], event[0][1], event[0][2]], axis=1, inplace=True)
    elif len(event[0]) == 2:
        df_encoded_tasks[event[1]] = df_encoded_tasks[event[0][0]] + df_encoded_tasks[event[0][1]]
        df_encoded_tasks.drop([event[0][0], event[0][1]], axis=1, inplace=True)
    return df_encoded_tasks
