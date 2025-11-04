import pandas as pd
import numpy as np

def impute_nans_own_vehicle(df: pd.DataFrame):
    # Impute missing values for the columns below. If the row is a pedestrian, exchange NaN with -1. 
    cols_to_be_cleaned_obstacle = ['initial_point_of_impact', "fixed_obstacle_struck", "mobile_obstacle_struck"]
    for col in cols_to_be_cleaned_obstacle:
        df.loc[
            (df['role'] == 'pedestrian') & (df[col].isna()),
            col
        ] = -1
    # Drop all other NaN Values (72) because imputing them with 0 does not fit here, since 0 is "None".
    # Introducing another value just for these edge cases is unnecessary imo. 
    df = df.dropna(subset=cols_to_be_cleaned_obstacle, how='any')
    
    # Impute missing values in pedestrian cases with "none" and in other cases with "unknown"
    df['vehicle_category'] = np.where((df['vehicle_category'].isna()) & (df['role'] == 'pedestrian'), 'none', df['vehicle_category'])#np.where((df['vehicle_category'].isna()) & (df['role'] != "pedestrian"), "unknown", df['vehicle_category']))
    df = df.dropna(subset=['vehicle_category'], how='any')

    # Impute missing values for the columns below. For pedestrian cases, NaN values are replaced by -1. The other NaN values are replaced by 0. (unknown)
    cols_to_be_cleaned_own = ['main_maneuver_before_accident', 'motor_type', 'impact_score']
    for col in cols_to_be_cleaned_own:
        df[col] = np.where((df[col].isna()) & (df['role'] == "pedestrian"), -1, np.where((df[col].isna()) & (df['role'] != "Pedestrian"), 0, df[col]))

    return df

def impute_nans_other_vehicle(df:pd.DataFrame):
    # Replace NaN values with -1 when no other car is present in the accident.
    df['initial_point_of_impact_other'] = np.where((df['initial_point_of_impact_other'].isna()) & (df['other_vehicle'] == 0), -1, df['initial_point_of_impact_other'])
    # Drop the remaining rows with NaN values (Same Reason as with "Initial_point_of_impact")
    df = df.dropna(subset=['initial_point_of_impact_other'], how='any')
    
    # Replace NaN values with "none" when no other car is present, otherwise with "unknown"
    df['vehicle_category_other'] = np.where((df['vehicle_category_other'].isna()) & (df['other_vehicle'] == 0), "none", df['vehicle_category'])#np.where((df['vehicle_category_other'].isna()) & (df['other_vehicle'] == 1), "unknown", df['vehicle_category_other']))
    df = df.dropna(subset=['vehicle_category_other'], how='any')

    # Replace NaN values with -1 when no other car is present otherwise with a 0
    cols_to_be_cleaned_other = ['initial_point_of_impact_other', 'main_maneuver_before_accident_other', 'motor_type_other', 'impact_score_other']
    for col in cols_to_be_cleaned_other:
        df[col] = np.where((df[col].isna()) & (df['other_vehicle'] == 0), -1, np.where((df[col].isna()) & (df['other_vehicle'] == 1), 0, df[col]))
    
    return df