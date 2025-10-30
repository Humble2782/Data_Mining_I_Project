# Data_Merge.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data.

import pandas as pd
import argparse
import glob
import re

# Import the new processing functions
import a_rename as na
import b_merge_tables as me
import c_feature_engineering as fe
import d_feature_selection as fs

def process_year(df_circumstances, df_locations, df_users, df_vehicles):
    """
    This function processes the data for a given year.
    It reads the data, calls processing functions, performs merges,
    and saves the final merged table for the specified year.
    """
    df_circumstances = na.rename_circumstances(df_circumstances)
    df_locations = na.rename_locations(df_locations)
    df_vehicles = na.rename_vehicles(df_vehicles)
    df_users = na.rename_users(df_users)
    
    # --- Call processing functions from feature_engineering.py ---
    df_users = me.process_users(df_users, df_circumstances[['id_accident', 'year']])
    df_vehicles = me.process_vehicles(df_vehicles)
    df_locations = me.process_locations(df_locations)

    # --- Perform merges using the processed dataframes ---
    vehicle_categories_involved = me.create_vehicle_dummies(df_vehicles)
    df_user_vehicle = me.process_user_vehicle_merge(df_users, df_vehicles)

    # --- Merge Tables ---
    merged_table = (
        df_circumstances
        .copy()
        .merge(df_locations, on='id_accident', how='left')
        .merge(df_user_vehicle, on='id_accident', how='left')
        .merge(vehicle_categories_involved, on='id_accident', how='left')
    )

    # Feature Engineering
    merged_table = fe.create_time_features(merged_table)
    merged_table = fe.create_age_feature(merged_table)
    merged_table = fe.create_security_equipment_one_hot(merged_table)

    final_table = fs.select_features(merged_table)

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