import pandas as pd

columns_to_drop = [
    # IDs and high-cardinality identifiers
    'id_accident', 'id_vehicle', 'id_user', 'number_vehicle', 'department_code',
    'commune_code', 'postal_address', 'road_number', 'road_number_index',
    'road_number_letter', 'id_vehicle_other', 'number_vehicle_other',

    # little information and a lot of missing values
    'width_central_reservation', 'number_occupants_in_public_transport',
    'number_occupants_in_public_transport_other',
    'nearest_reference_marker_distance', 'nearest_reference_marker',

    # Original columns that have been engineered into new features
    'year_of_birth',  # Replaced by 'age' and 'age_group'
    'year', 'month', 'day',  # Replaced by cyclical features (sin/cos) and 'day_of_week'
    'hour_minute',  # Original string column, already dropped in c_feature_engineering
    'hour', 'minute',  # Replaced by 'time_of_day', 'hour_sin', 'hour_cos'
    'safety_equipment_1', 'safety_equipment_2', 'safety_equipment_3',  # Replaced by 'used_belt', 'used_helmet', etc.
    'vehicle_category',  # Replaced by 'vehicle_category_simplified', 'impact_score'
    'number_of_traffic_lanes', 'intersection', 'road_category', 'traffic_regime',  # Engineered into 'road_complexity_index'
    'pavement_condition', 'longitudinal_profile',  # Engineered into 'surface_quality_indicator'
    'user_category',  # Replaced by 'role'

    # Drop the original multi-class target variable.
    # We are replacing 'injury_severity' (1,2,3,4) with our new ordinal 'injury_target' (0,1,2).
    'injury_severity',

    # Other columns deemed not useful for modeling
    'injured_pedestrian_alone', 'trip_purpose', 'direction_of_travel',
    'carriageway_width',

    # Drop unrealistic/engineered secu equipment flags that are not needed
    'used_other', 'used_gloves', 'used_reflective_vest',
]


def select_features(df_table: pd.DataFrame):
    """
    Drops columns that are irrelevant for modeling from the DataFrame.
    """
    # Identify which of the columns to drop actually exist in the DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_table.columns]

    if existing_columns_to_drop:
        print(f"  Dropping {len(existing_columns_to_drop)} columns...")
        df_selected = df_table.drop(columns=existing_columns_to_drop)
        return df_selected
    else:
        print("  No columns to drop found in the DataFrame.")
        return df_table

