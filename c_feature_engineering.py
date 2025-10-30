import pandas as pd

def create_time_features(df_merged: pd.DataFrame):
    df_merged_copy = df_merged.copy()

    # Convert the hour_minute attribute into two separate "hour" and "minute" features.
    datetime_series = pd.to_datetime(df_merged_copy['hour_minute'], format='%H:%M')
    df_merged_copy['hour'] = datetime_series.dt.hour
    df_merged_copy['minute'] = datetime_series.dt.minute
    df_merged_copy.drop(columns='hour_minute', inplace=True)
    
    return df_merged_copy

def create_age_feature(df_merged: pd.DataFrame):
    df_merged_copy = df_merged.copy()
    df_merged_copy['age'] = (df_merged_copy['year'] - df_merged_copy['year_of_birth']).astype('Int64')
    return df_merged_copy

def create_security_equipment_one_hot(df_merged: pd.DataFrame):
    df_merged_copy = df_merged.copy()
    
    SECU_FLAG = {
        1:"used_belt", 2:"used_helmet", 3:"used_child_restraint", 4:"used_reflective_vest",
        5:"used_airbag", 6:"used_gloves", 7:"used_gloves_and_airbag", 9:"used_other"
    }
    SECU_COLS = ["safety_equipment_1","safety_equipment_2","safety_equipment_3"]

    S = df_merged_copy[SECU_COLS].apply(pd.to_numeric, errors="coerce")

    for code, name in SECU_FLAG.items():
        df_merged_copy[name] = S.isin([code]).any(axis=1).astype(int)

    # Merge Airbag and Airbag + Gloves
    df_merged_copy['used_airbag'] = df_merged_copy[['used_airbag', 'used_gloves_and_airbag']].any(axis=1).astype(int)
    df_merged_copy['used_gloves'] = df_merged_copy[['used_gloves', 'used_gloves_and_airbag']].any(axis=1).astype(int)

    # Remove columns not needed anymore
    df_merged_copy.drop(columns='used_gloves_and_airbag', inplace=True)

    return df_merged_copy