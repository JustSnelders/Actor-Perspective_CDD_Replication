import datetime
import pandas as pd


def transform_neo_duration(dataframe, column, unit='seconds'):
    for index, row in dataframe.iterrows():
        duration = row[column]
        duration_seconds = duration.hours_minutes_seconds_nanoseconds[0] * 3600 + \
                           duration.hours_minutes_seconds_nanoseconds[1] * 60 + \
                           duration.hours_minutes_seconds_nanoseconds[2]
        if unit == 'seconds':
            duration_unit = duration_seconds
        elif unit == 'minutes':
            duration_unit = duration_seconds / 60
        elif unit == 'hours':
            duration_unit = duration_seconds / 60 / 60
        elif unit == 'days':
            duration_unit = duration_seconds / 60 / 60 / 24
        dataframe.loc[index, f'{column}_{unit}'] = duration_unit
    dataframe.drop(columns=[column], inplace=True)
    return dataframe


def transform_neo_date(dataframe, column):
    dataframe[column] = pd.DatetimeIndex(dataframe[column].astype(str))
    return dataframe


def parse_to_list(query_result, label):
    result_as_list = []
    for record in query_result:
        result_as_list.append(record[label])
    return result_as_list


def parse_to_2d_list(query_result, label1, label2):
    result_as_list = []
    for record in query_result:
        result_as_list.append([record[label1], record[label2]])
    return result_as_list


def parse_timestamp(query_result, key):
    neo_timestamp = query_result[0][key]
    timestamp = datetime.datetime(neo_timestamp.year, neo_timestamp.month, neo_timestamp.day, neo_timestamp.hour,
                                  neo_timestamp.minute, int(neo_timestamp.second)).date()
    return timestamp


def parse_to_dataframe(query_result, timedelta_cols: dict = None, timestamp_cols: list = None):
    dataframe = pd.DataFrame([dict(record) for record in query_result])
    if timedelta_cols is not None:
        for timedelta_col_name, unit in timedelta_cols.items():
            transform_neo_duration(dataframe, timedelta_col_name, unit=unit)
    if timestamp_cols is not None:
        for timestamp_col in timestamp_cols:
            transform_neo_date(dataframe, timestamp_col)
    return dataframe
