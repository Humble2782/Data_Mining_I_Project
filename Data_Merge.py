# Data_Merge.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data.

import pandas as pd
import argparse
import glob
import re

# Import the new processing functions
import feature_engineering as fe

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
    'grav': 'injury_severity', # This is the feature we want to predict.
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


def process_year(df_circumstances, df_locations, df_users, df_vehicles):
    """
    This function processes the data for a given year.
    It reads the data, calls processing functions, performs merges,
    and saves the final merged table for the specified year.
    """
    df_circumstances = df_circumstances.rename(columns=circumstances_column_names)
    df_locations = df_locations.rename(columns=location_column_names)
    df_vehicles = df_vehicles.rename(columns=vehicles_column_names)
    df_users = df_users.rename(columns=users_column_names)
    
    # --- Call processing functions from feature_engineering.py ---
    processed_users = fe.process_users(df_users, df_circumstances[['id_accident', 'year']])
    processed_vehicles = fe.process_vehicles(df_vehicles)
    selected_locations = fe.process_locations(df_locations)

    # --- Perform merges using the processed dataframes ---
    vehicle_categories_involved = fe.create_vehicle_dummies(processed_vehicles)
    df_user_vehicle = fe.process_user_vehicle_merge(processed_users, processed_vehicles)

    # --- Final merge ---
    final_table = (
        df_circumstances
        .copy()
        .merge(selected_locations, on='id_accident', how='left')
        .merge(df_user_vehicle, on='id_accident', how='left')
        .merge(vehicle_categories_involved, on='id_accident', how='left')
    )

    return final_table


def process_files(input_path: str, output_path: str):
    files_path = glob.glob('data/original/*-*.csv')
    years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files_path)))))
    
    if not years:
        print("No year-specific data files found in 'data/' directory.")
        print("Please ensure files are named like 'locations-2022.csv'.")
    else:
        print(f"Found data for years: {years}")
        for year in years:
            try:
                print(f"Processing data for year: {year}")
                df_circumstances = pd.read_csv(f'{input_path}/characteristics-{year}.csv', sep=';', decimal=',')
                df_locations = pd.read_csv(f'{input_path}/locations-{year}.csv', sep=';', decimal=',')
                df_users = pd.read_csv(f'{input_path}/users-{year}.csv', sep=';', decimal=',')
                df_vehicles = pd.read_csv(f'{input_path}/vehicles-{year}.csv', sep=';', decimal=',')

                final_table = process_year(df_circumstances, df_locations, df_users, df_vehicles)
                
                if output_path:
                    output_filename = f'{output_path}/full_table-{year}.csv'
                else: 
                    output_filename = f'data/full/full_table_{year}.csv'

                final_table.to_csv(output_filename, sep=';', index=False)
                print(f"Successfully saved merged data for year {year} to {output_filename}")

            except FileNotFoundError as e:
                print(f"Skipping year {year} due to missing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help='The file path of the source files.', type=str)
    parser.add_argument('-o', '--output_path', help='The path of the output folder.', type=str)
    args = parser.parse_args()

    process_files(args.input_path, args.output_path)