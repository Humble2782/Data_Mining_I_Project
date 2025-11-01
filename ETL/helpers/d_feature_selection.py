import pandas as pd

columns_to_drop = [
    # IDs
    'id_accident', 'id_vehicle', 'id_user', 'number_vehicle', 'department_code', 'id_vehicle_other',
    # Unique codes
    'commune_code', 'postal_address', 'road_number', 'road_number_index',
    'road_number_letter', 'number_vehicle_other',

    # little information and a lot of missing values
    'width_central_reservation', 'number_occupants_in_public_transport', 'number_occupants_in_public_transport_other',
    'nearest_reference_marker_distance', 'nearest_reference_marker',

    # Other columns deemed not useful for modeling
    'injured_pedestrian_alone', 'carriageway_width', 'trip_purpose', 'year', 'year_of_birth',

    # Drop unrealistic secu equipment
    'used_other', 'used_gloves', 'used_reflective_vest',

    # Replaced by feature engineering
    'safety_equipment_1', 'safety_equipment_2', 'safety_equipment_3'
]


def select_features(df_table: pd.DataFrame):
    return df_table.drop(columns=columns_to_drop)
