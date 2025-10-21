# Import libraries
import pandas as pd
import os
import glob
import re

def get_years():
    """Find all years from the filenames in the data directory."""
    files = glob.glob('data/original/*-*.csv')
    years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files)))))
    if not years:
        print("No year-specific data files found in 'data/original/' directory.")
        print("Please ensure files are named like 'lieux-2022.csv'.")
    else:
        print(f"Found data for years: {years}")
    return years

def preprocess_and_save(year, input_name, output_name, column_map):
    """ 
    Reads a CSV file, renames the columns, and saves it to a new location.
    """
    input_path = f'data/original/{input_name}-{year}.csv'
    output_path = f'data/renamed/{output_name}-{year}.csv'

    try:
        # Using encoding='latin1' for French characters
        df = pd.read_csv(input_path, sep=';', encoding='latin1', low_memory=False)
    except FileNotFoundError:
        print(f"File not found, skipping: {input_path}")
        return

    # Rename columns using the provided map
    df.rename(columns=column_map, inplace=True)

    df.to_csv(output_path, sep=';', index=False)
    print(f"Successfully preprocessed and saved: {output_path}")


if __name__ == "__main__":
    output_dir = 'data/renamed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    years = get_years()

    # Column maps
    characteristics_column_names = {
        'Num_Acc': 'id_accident', 'jour': 'day', 'mois': 'month', 'an': 'year',
        'hrmn': 'hour_minute', 'lum': 'lighting_condition', 'dep': 'department_code',
        'com': 'commune_code', 'agg': 'location', 'int': 'intersection',
        'atm': 'weather_condition', 'col': 'type_of_collision', 'adr': 'postal_address',
        'lat': 'latitude', 'long': 'longitude'
    }
    locations_column_names = {
        'Num_Acc': 'id_accident', 'catr': 'road_category', 'voie': 'road_number',
        'v1': 'road_number_index', 'v2': 'road_number_letter', 'circ': 'traffic_regime',
        'nbv': 'number_of_traffic_lanes', 'vosp': 'reserved_lane_present',
        'prof': 'longitudinal_profile', 'pr': 'nearest_reference_marker',
        'pr1': 'nearest_reference_marker_distance', 'plan': 'horizontal_alignment',
        'lartpc': 'width_central_reservation', 'larrout': 'carriageway_width',
        'surf': 'pavement_condition', 'infra': 'infrastructure', 'situ': 'accident_situation',
        'vma': 'speed_limit'
    }
    vehicles_column_names = {
        'Num_Acc': 'id_accident', 'id_vehicule': 'id_vehicle', 'num_veh': 'number_vehicle',
        'senc': 'direction_of_travel', 'catv': 'vehicle_category', 'obs': 'fixed_obstacle_struck',
        'obsm': 'mobile_obstacle_struck', 'choc': 'initial_point_of_impact',
        'manv': 'main_maneuver_before_accident', 'motor': 'motor_type',
        'occutc': 'number_occupants_in_public_transport'
    }
    users_column_names = {
        'Num_Acc': 'id_accident', 'id_usager': 'id_user', 'id_vehicule': 'id_vehicle',
        'num_veh': 'number_vehicle', 'place': 'position', 'catu': 'user_category',
        'grav': 'injury_severity', 'sexe': 'sex', 'an_nais': 'year_of_birth',
        'trajet': 'trip_purpose', 'secu1': 'safety_equipment_1', 'secu2': 'safety_equipment_2',
        'secu3': 'safety_equipment_3', 'locp': 'pedestrian_location',
        'actp': 'pedestrian_action', 'etatp': 'injured_pedestrian_alone'
    }

    # A single, consistent map for all years
    table_maps = {
        'carcteristiques': ('characteristics', characteristics_column_names),
        'lieux': ('locations', locations_column_names),
        'usagers': ('users', users_column_names),
        'vehicules': ('vehicles', vehicles_column_names)
    }

    for year in years:
        for input_name, (output_name, column_map) in table_maps.items():
            preprocess_and_save(year, input_name, output_name, column_map)