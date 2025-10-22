# Data_Merge.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data.

import pandas as pd
import glob
import re
# Import the new processing functions
import feature_engineering as fe


def process_year(year):
    """
    This function processes the data for a given year.
    It reads the data, calls processing functions, performs merges,
    and saves the final merged table for the specified year.
    """
    print(f"Processing data for year: {year}")

    # Import the data for the given year
    try:
        df_circumstances = pd.read_csv(f'data/renamed/characteristics-{year}.csv', sep=';')
        df_locations = pd.read_csv(f'data/renamed/locations-{year}.csv', sep=';')
        df_users = pd.read_csv(f'data/renamed/users-{year}.csv', sep=';')
        df_vehicles = pd.read_csv(f'data/renamed/vehicles-{year}.csv', sep=';')
    except FileNotFoundError as e:
        print(f"Skipping year {year} due to missing file: {e}")
        return

    # --- Call processing functions from feature_engineering.py ---
    processed_users = fe.process_users(df_users, df_circumstances)
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

    # Save the final table for the year
    output_filename = f'data/full/full_table_{year}.csv'
    final_table.to_csv(output_filename, sep=';', index=False)
    print(f"Successfully saved merged data for year {year} to {output_filename}")


if __name__ == "__main__":
    files = glob.glob('data/renamed/*-*.csv')
    years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files)))))

    if not years:
        print("No year-specific data files found in 'data/' directory.")
        print("Please ensure files are named like 'locations-2022.csv'.")
    else:
        print(f"Found data for years: {years}")
        for year in years:
            process_year(year)