# preprocessing.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data using Multiprocessing.
import pandas as pd
import argparse
import glob
import re
import concurrent.futures
import os
from pathlib import Path
from typing import TypedDict, Tuple, Optional

# Import the new processing functions
from helpers import a_rename as na
from helpers import b_merge_tables as me
from helpers import c_feature_engineering as fe
from helpers import d_handle_missing_values as hmv
from helpers import e_feature_selection as fs


# Typing
class AccidentDataRenamed(TypedDict):
    circumstances: pd.DataFrame
    locations: pd.DataFrame
    vehicles: pd.DataFrame
    users: pd.DataFrame

class AccidentPreprocessingResult(TypedDict):
    B_renamed: AccidentDataRenamed
    C_merged: pd.DataFrame
    D_feature_engineering: pd.DataFrame
    E_missing_data_handling: pd.DataFrame
    F_feature_selection: pd.DataFrame


def process_year(df_circumstances: pd.DataFrame, df_locations: pd.DataFrame, df_users: pd.DataFrame,
                 df_vehicles: pd.DataFrame) -> AccidentPreprocessingResult:
    """
    This function processes the data for a given year.
    """
    df_circumstances_renamed = na.rename_circumstances(df_circumstances)
    df_locations_renamed = na.rename_locations(df_locations)
    df_vehicles_renamed = na.rename_vehicles(df_vehicles)
    df_users_renamed = na.rename_users(df_users)

    # --- Call processing functions ---
    df_circumstances_processed = df_circumstances_renamed
    df_users_processed = me.process_users(df_users_renamed)
    df_vehicles_processed = me.process_vehicles(df_vehicles_renamed)
    df_locations_processed = me.process_locations(df_locations_renamed)

    # --- Perform merges ---
    vehicle_categories_involved_processed = me.create_vehicle_dummies(df_vehicles_processed)
    df_user_vehicle_processed = me.process_user_vehicle_merge(df_users_processed, df_vehicles_processed)

    merged_table = (
        df_circumstances_processed
        .copy()
        .merge(df_locations_processed, on='id_accident', how='left')
        .merge(df_user_vehicle_processed, on='id_accident', how='left')
        .merge(vehicle_categories_involved_processed, on='id_accident', how='left')
    )

    # --- Clean up impossible speed limits ---
    initial_rows = len(merged_table)
    merged_table.drop(index=merged_table[merged_table['speed_limit'] > 130].index, inplace=True)

    # --- Feature Engineering Pipeline ---
    # Note: Removed standard prints to avoid console clutter in multiprocessing
    feature_engineering_table = fe.create_datetime_features(merged_table)
    feature_engineering_table = fe.create_age_features(feature_engineering_table)
    feature_engineering_table = fe.create_security_equipment_one_hot(feature_engineering_table)
    feature_engineering_table = fe.create_vehicle_features(feature_engineering_table)
    feature_engineering_table = fe.create_road_complexity_index(feature_engineering_table)
    feature_engineering_table = fe.create_surface_quality_indicator(feature_engineering_table)
    feature_engineering_table = fe.create_user_role(feature_engineering_table)
    feature_engineering_table = fe.create_ordinal_conditions(feature_engineering_table)
    feature_engineering_table = fe.create_ordinal_target(feature_engineering_table)

    # --- Imputation Step ---
    imputed_table = hmv.impute_missing_values(feature_engineering_table)

    # --- Feature Selection ---
    feature_selection_table = fs.select_features(imputed_table)

    # --- Add cluster column ---
    clustered_table = fe.create_cluster_feature(feature_selection_table)

    # --- Dtype Conversion ---
    columns_to_int16 = [
        'location', 'type_of_collision', 'reserved_lane_present', 'horizontal_alignment',
        'infrastructure', 'accident_situation', 'position', 'fixed_obstacle_struck',
        'mobile_obstacle_struck', 'initial_point_of_impact', 'main_maneuver_before_accident',
        'motor_type', 'fixed_obstacle_struck_other', 'mobile_obstacle_struck_other',
        'initial_point_of_impact_other', 'main_maneuver_before_accident_other',
        'motor_type_other', 'vehicle_category_involved_bicycle', 'vehicle_category_involved_bus_coach',
        'vehicle_category_involved_hgv_truck', 'vehicle_category_involved_light_motor_vehicle',
        'vehicle_category_involved_other', 'vehicle_category_involved_powered_2_3_wheeler',
        'used_belt', 'used_helmet', 'used_child_restraint', 'used_airbag',
        'impact_score', 'impact_score_other', 'impact_delta', 'surface_quality_indicator',
        'lighting_ordinal', 'weather_ordinal', 'injury_target', 'sex', 'day_of_week', 'speed_limit', 'age', 'cluster'
    ]

    existing_cols_to_convert = [col for col in columns_to_int16 if col in clustered_table.columns]

    if existing_cols_to_convert:
        clustered_table[existing_cols_to_convert] = clustered_table[existing_cols_to_convert].astype('int16')

    return {
        'B_renamed': {
            'circumstances': df_circumstances_renamed,
            'locations': df_locations_renamed,
            'vehicles': df_vehicles_renamed,
            'users': df_users_renamed
        },
        'C_merged': merged_table,
        'D_feature_engineering': feature_engineering_table,
        'E_missing_data_handling': imputed_table,
        'F_feature_selection': clustered_table
    }


def process_year_pipeline(year: str, input_path: str, output_path: Path) -> Tuple[str, Optional[pd.DataFrame], str]:
    """
    Worker function to be executed in parallel.
    Reads files, processes them, saves intermediate results, and returns the final dataframe.
    Returns: (year, final_dataframe, status_message)
    """
    try:
        print(f"[Start] Processing year: {year}")

        # 1. Read Files
        df_circumstances = pd.read_csv(f'{input_path}/characteristics-{year}.csv', sep=';', decimal=',',
                                       low_memory=False)
        df_locations = pd.read_csv(f'{input_path}/locations-{year}.csv', sep=';', decimal=',', low_memory=False)
        df_users = pd.read_csv(f'{input_path}/users-{year}.csv', sep=';', decimal=',', low_memory=False)
        df_vehicles = pd.read_csv(f'{input_path}/vehicles-{year}.csv', sep=';', decimal=',', low_memory=False)

        # 2. Process
        results: AccidentPreprocessingResult = process_year(df_circumstances, df_locations, df_users, df_vehicles)
        final_df_year = results['F_feature_selection']

        # 3. Save Intermediate Files
        folders = results.keys()
        for folder_name in folders:
            folder_path_full = output_path / folder_name
            # Note: exist_ok=True is thread/process safe
            folder_path_full.mkdir(exist_ok=True, parents=True)

            if folder_name == 'B_renamed':
                year_path = folder_path_full / year
                year_path.mkdir(exist_ok=True)

                for key, df in results['B_renamed'].items():
                    file_path: Path = year_path / f'{key}_renamed-{year}.csv'
                    df.to_csv(file_path, sep=';', index=False)
            else:
                filename_base = folder_name.split('_', 1)[1]
                file_path: Path = folder_path_full / f'{filename_base}-{year}.csv'
                results[folder_name].to_csv(file_path, sep=';', index=False)

        msg = f"[Done] Successfully processed year {year}"
        return year, final_df_year, msg

    except Exception as e:
        # Return None as dataframe if failed
        return year, None, f"[Error] Year {year} failed: {str(e)}"


def process_files(input_path: str, output_path: str = 'data'):
    files_path = glob.glob(f'{input_path}/*-*.csv')
    years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files_path)))))

    TEST_YEAR = '2023'
    train_val_dfs = []
    test_dfs = []

    if not years:
        print(f"No year-specific data files found in '{input_path}'.")
        return

    print(f"Found data for years: {years}")

    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # --- MULTIPROCESSING SECTION ---
    # We use ProcessPoolExecutor for CPU-bound tasks (Pandas processing)
    # Determine max_workers based on your CPU cores (leave one core free usually)
    max_workers = max(1, os.cpu_count() - 1)

    print(f"Starting multiprocessing with {max_workers} workers...")

    # Helper list to collect results before appending to ensure order
    results_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_year = {
            executor.submit(process_year_pipeline, year, input_path, output_path_obj): year
            for year in years
        }

        # Collect results
        for future in concurrent.futures.as_completed(future_to_year):
            year, df_result, msg = future.result()
            print(msg)
            results_list.append((year, df_result))

    # --- SORTING RESULTS ---
    # Crucial: Sort by year to ensure consistent order in the final CSV files
    # otherwise consecutive runs might produce different row orders based on CPU timing.
    results_list.sort(key=lambda x: x[0])

    # --- Distribute to Train/Test Lists ---
    for year, df_result in results_list:
        if df_result is not None:
            if year == TEST_YEAR:
                test_dfs.append(df_result)
            else:
                train_val_dfs.append(df_result)

    # --- Combining and Saving Final Datasets ---
    print("\nCombining and saving final train/test datasets...")

    if train_val_dfs:
        train_val_combined = pd.concat(train_val_dfs, ignore_index=True)

        train_val_path_csv = output_path_obj / 'train_val_data.csv'
        train_val_path_parquet = output_path_obj / 'train_val_data.parquet'
        train_val_path_csv_zip = output_path_obj / 'train_val_data.csv.zip'

        train_val_combined.to_csv(train_val_path_csv, sep=';', index=False)
        train_val_combined.to_parquet(train_val_path_parquet, index=False, engine='pyarrow')
        train_val_combined.to_csv(train_val_path_csv_zip, sep=';', index=False, compression='zip')

        print(f"Saved train/val data ({len(train_val_combined)} rows).")

    if test_dfs:
        test_combined = pd.concat(test_dfs, ignore_index=True)

        test_path_csv = output_path_obj / 'test_data.csv'
        test_path_parquet = output_path_obj / 'test_data.parquet'
        test_path_csv_zip = output_path_obj / 'test_data.csv.zip'

        test_combined.to_csv(test_path_csv, sep=';', index=False)
        test_combined.to_parquet(test_path_parquet, index=False, engine='pyarrow')
        test_combined.to_csv(test_path_csv_zip, sep=';', index=False, compression='zip')

        print(f"Saved test data ({len(test_combined)} rows).")


if __name__ == "__main__":
    # Multiprocessing requires the main execution block to be guarded
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Input path', type=str, nargs='?', default='../data/A_original')
    parser.add_argument('-o', '--output_path', help='Output path', type=str, default='../data/')

    args = parser.parse_args()
    process_files(args.input_path, args.output_path)