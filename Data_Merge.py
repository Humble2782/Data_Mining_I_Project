# Data_Merge.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data.

import pandas as pd
import argparse
import glob
import re

# Import the new processing functions
import feature_engineering as fe
import a_rename as re

def process_year(df_circumstances, df_locations, df_users, df_vehicles):
    """
    This function processes the data for a given year.
    It reads the data, calls processing functions, performs merges,
    and saves the final merged table for the specified year.
    """
    df_circumstances = re.rename_circumstances(df_circumstances)
    df_locations = re.rename_locations(df_locations)
    df_vehicles = re.rename_vehicles(df_vehicles)
    df_users = re.rename_users(df_users)
    
    # --- Call processing functions from feature_engineering.py ---
    processed_circumstances = fe.process_circumstances(df_circumstances)
    processed_users = fe.process_users(df_users, processed_circumstances[['id_accident', 'year']])
    processed_vehicles = fe.process_vehicles(df_vehicles)
    selected_locations = fe.process_locations(df_locations)

    # --- Perform merges using the processed dataframes ---
    vehicle_categories_involved = fe.create_vehicle_dummies(processed_vehicles)
    df_user_vehicle = fe.process_user_vehicle_merge(processed_users, processed_vehicles)

    # --- Final merge ---
    final_table = (
        processed_circumstances
        .copy()
        .merge(selected_locations, on='id_accident', how='left')
        .merge(df_user_vehicle, on='id_accident', how='left')
        .merge(vehicle_categories_involved, on='id_accident', how='left')
    )

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
        'injured_pedestrian_alone', 'carriageway_width', 'trip_purpose', 'year',

        # Drop unrealistic secu equipment
        'used_other', 'used_gloves', 'used_reflective_vest' 
    ]
    
    final_table.drop(columns=columns_to_drop, inplace=True)

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