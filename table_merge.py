# table_merge.py
# Description: This script handles all data integration. It reads the raw yearly files,
# processes them, and merges them into a single, comprehensive table per year.

import pandas as pd
import glob
import re
import os
import argparse


# --- Sub-Processing Functions ---

def process_users(df_users: pd.DataFrame) -> pd.DataFrame:
    """Processes user data to add a 'role' column."""
    print("  Processing users table...")
    df_users_copy = df_users.copy()
    role_map = {1: 'driver', 2: 'passenger', 3: 'pedestrian'}
    df_users_copy['role'] = df_users_copy['user_category'].map(role_map).fillna('other')
    return df_users_copy


def process_vehicles(df_vehicles: pd.DataFrame) -> pd.DataFrame:
    """Processes vehicle data by simplifying categories."""
    print("  Processing vehicles table...")
    # This is a placeholder for any vehicle-specific pre-merge processing.
    # The simplification logic now lives in the main feature_engineering.py,
    # but if any was needed FOR the merge, it would go here.
    return df_vehicles.copy()


def process_locations(df_locations: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the most relevant location entry for each accident based on data completeness.
    This is a critical step that belongs to the merging logic.
    """
    print("  Selecting best location entry per accident...")
    LOC_SCORE_COLS = [
        "road_category", "road_number", "traffic_regime", "number_of_traffic_lanes",
        "longitudinal_profile", "horizontal_alignment", "pavement_condition",
        "infrastructure", "accident_situation", "speed_limit"
    ]

    L = df_locations.copy()

    # Calculate a score based on how many relevant fields are filled
    completeness_score = L[LOC_SCORE_COLS].notna().sum(axis=1)
    L['__score'] = completeness_score

    # Sort by accident ID and score, then keep only the best entry per accident
    L = L.sort_values(["id_accident", "__score"]).drop_duplicates("id_accident", keep="last")
    L = L.drop(columns=["__score"])

    return L


def create_vehicle_dummies(df_vehicles: pd.DataFrame) -> pd.DataFrame:
    """Creates dummy variables for vehicle categories involved in each accident."""
    print("  Creating vehicle category dummies for merge...")
    # This step is often useful during merging to aggregate vehicle info per accident
    g = df_vehicles[['id_accident', 'vehicle_category']].copy()
    # Using a simplified category logic here if needed for the merge itself
    tmp = pd.get_dummies(g, columns=['vehicle_category'], prefix='vehicle_cat')
    return tmp.groupby('id_accident').sum().reset_index()


def process_user_vehicle_merge(df_users: pd.DataFrame, df_vehicles: pd.DataFrame) -> pd.DataFrame:
    """Merges user and vehicle data."""
    print("  Merging users and vehicles...")
    df_user_vehicle = df_users.merge(df_vehicles, on=['id_accident', 'id_vehicle'], how='left')
    return df_user_vehicle


# --- Main Orchestration Function ---

def run_table_merge(year: int, input_folder: str, output_folder: str):
    """
    Orchestrates the reading, processing, merging, and saving for a single year.
    """
    print(f"\nProcessing data for year: {year}")

    try:
        df_circumstances = pd.read_csv(f'{input_folder}/characteristics-{year}.csv', sep=';', low_memory=False)
        df_locations = pd.read_csv(f'{input_folder}/locations-{year}.csv', sep=';', low_memory=False)
        df_users = pd.read_csv(f'{input_folder}/users-{year}.csv', sep=';', low_memory=False)
        df_vehicles = pd.read_csv(f'{input_folder}/vehicles-{year}.csv', sep=';', low_memory=False)
    except FileNotFoundError as e:
        print(f"  Skipping year {year} due to missing file: {e}")
        return

    # --- 1. Process individual tables before merging ---
    processed_users = process_users(df_users)
    processed_vehicles = process_vehicles(df_vehicles)
    selected_locations = process_locations(df_locations)

    # --- 2. Create aggregated features for the main merge ---
    vehicle_dummies = create_vehicle_dummies(processed_vehicles)

    # --- 3. Merge user and vehicle data together ---
    df_user_vehicle = process_user_vehicle_merge(processed_users, processed_vehicles)

    # --- 4. Final Merge ---
    # Merge all processed parts into the final table
    final_table = (
        df_circumstances
        .merge(selected_locations, on='id_accident', how='left')
        .merge(df_user_vehicle, on='id_accident', how='left')
        .merge(vehicle_dummies, on='id_accident', how='left')
    )

    # --- 5. Save the result ---
    output_filename = f'{output_folder}/full_table_{year}.csv'
    final_table.to_csv(output_filename, sep=';', index=False)
    print(f"Successfully saved merged data for year {year} to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge raw accident data files into a single table per year.")
    parser.add_argument('--input_folder', type=str, default='data/renamed',
                        help="Folder containing the raw CSV files.")
    parser.add_argument('--output_folder', type=str, default='data/full',
                        help="Folder to save the merged 'full_table' files.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Automatically find all years from the filenames
    files = glob.glob(f'{args.input_folder}/*-*.csv')
    if not files:
        print(f"No data files found in '{args.input_folder}'.")
    else:
        years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files)))))
        print(f"Found data for years: {years}")
        for year in years:
            run_table_merge(int(year), args.input_folder, args.output_folder)