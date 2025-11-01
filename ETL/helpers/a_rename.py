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
    return df_users.rename(columns=users_column_names)
