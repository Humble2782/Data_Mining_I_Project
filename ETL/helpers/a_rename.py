import pandas as pd

# English column name translations to better understand the data.
circumstances_column_names = {
    'Num_Acc': 'id_accident',
    'jour': 'day',
    'mois': 'month',
    'an': 'year',
    'hrmn': 'hour_minute',
    'lum': 'lighting_condition',
    'dep': 'department_code',
    'com': 'commune_code',
    'agg': 'location',
    'int': 'intersection',
    'atm': 'weather_condition',
    'col': 'type_of_collision',
    'adr': 'postal_address',
    'lat': 'latitude',
    'long': 'longitude'
}

location_column_names = {
    'Num_Acc': 'id_accident',
    'catr': 'road_category',
    'voie': 'road_number',
    'v1': 'road_number_index',
    'v2': 'road_number_letter',
    'circ': 'traffic_regime',
    'nbv': 'number_of_traffic_lanes',
    'vosp': 'reserved_lane_present',
    'prof': 'longitudinal_profile',
    'pr': 'nearest_reference_marker',
    'pr1': 'nearest_reference_marker_distance',
    'plan': 'horizontal_alignment',
    'lartpc': 'width_central_reservation',
    'larrout': 'carriageway_width',
    'surf': 'pavement_condition',
    'infra': 'infrastructure',
    'situ': 'accident_situation',
    'vma': 'speed_limit'
}

vehicles_column_names = {
    'Num_Acc': 'id_accident',
    'id_vehicule': 'id_vehicle',
    'num_veh': 'number_vehicle',
    'senc': 'direction_of_travel',
    'catv': 'vehicle_category',
    'obs': 'fixed_obstacle_struck',
    'obsm': 'mobile_obstacle_struck',
    'choc': 'initial_point_of_impact',
    'manv': 'main_maneuver_before_accident',
    'motor': 'motor_type',
    'occutc': 'number_occupants_in_public_transport'
}

users_column_names = {
    'Num_Acc': 'id_accident',
    'id_usager': 'id_user',
    'id_vehicule': 'id_vehicle',
    'num_veh': 'number_vehicle',
    'place': 'position',
    'catu': 'user_category',
    'grav': 'injury_severity',  # This is the feature we want to predict.
    'sexe': 'sex',
    'an_nais': 'year_of_birth',
    'trajet': 'trip_purpose',
    'secu1': 'safety_equipment_1',
    'secu2': 'safety_equipment_2',
    'secu3': 'safety_equipment_3',
    'locp': 'pedestrian_location',
    'actp': 'pedestrian_action',
    'etatp': 'injured_pedestrian_alone'
}


def rename_circumstances(df_circumstances: pd.DataFrame):
    return df_circumstances.rename(columns=circumstances_column_names)


def rename_locations(df_locations: pd.DataFrame):
    return df_locations.rename(columns=location_column_names)


def rename_vehicles(df_vehicles: pd.DataFrame):
    return df_vehicles.rename(columns=vehicles_column_names)


def rename_users(df_users: pd.DataFrame):
    """
    Normalizes the schema for the users table.
    Handles missing 'id_usager' in older files by creating a synthetic 'id_user'.
    """
    df_copy = df_users.copy()

    # Define the key source columns from the dictionary
    acc_key_source = 'Num_Acc'
    user_key_source = 'id_usager'

    # Get the target names from the dictionary
    acc_key_target = users_column_names[acc_key_source]
    user_key_target = users_column_names[user_key_source]

    # Handle 'id_accident' (Num_Acc) first (needed for synthetic ID)
    if acc_key_source in df_copy.columns:
        df_copy.rename(columns={acc_key_source: acc_key_target}, inplace=True)
    elif acc_key_target not in df_copy.columns:
        # If 'Num_Acc' is also not present, we can't proceed.
        raise ValueError(f"Source column '{acc_key_source}' (for 'id_accident') not found in users file.")

    # Handle 'id_user' (id_usager) - real or synthetic
    if user_key_source in df_copy.columns:
        # Case 1: 2022+ data (has id_usager)
        df_copy.rename(columns={user_key_source: user_key_target}, inplace=True)
    else:
        # Case 2: 2019/2020 data (missing id_usager)
        # Create a unique index (1, 2, 3...) for each user within an accident group
        df_copy.sort_index(inplace=True)
        df_copy['user_index'] = df_copy.groupby(acc_key_target).cumcount() + 1

        # Create a robust, unique string ID (e.g., "2019000001_U1")
        df_copy[user_key_target] = df_copy[acc_key_target].astype(str) + '_U' + df_copy['user_index'].astype(str)

        df_copy.drop(columns=['user_index'], inplace=True)

    # Handle all OTHER renames from the dictionary
    # Create a new map, excluding the ones we handled manually
    other_renames = {
        k: v for k, v in users_column_names.items()
        if k not in [acc_key_source, user_key_source]  # Exclude 'Num_Acc' and 'id_usager'
    }

    # Rename only columns that exist in the dataframe
    existing_renames = {k: v for k, v in other_renames.items() if k in df_copy.columns}
    df_copy.rename(columns=existing_renames, inplace=True)

    return df_copy

