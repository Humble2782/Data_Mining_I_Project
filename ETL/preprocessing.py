# preprocessing.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data.
import pandas as pd
import argparse
import glob
import re
from pathlib import Path
from typing import TypedDict

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
    # Updated keys to match the actual return dictionary
    B_renamed: AccidentDataRenamed
    C_merged: pd.DataFrame
    D_feature_engineering: pd.DataFrame
    E_missing_data_handling: pd.DataFrame
    F_feature_selection: pd.DataFrame


def process_year(df_circumstances: pd.DataFrame, df_locations: pd.DataFrame, df_users: pd.DataFrame,
                 df_vehicles: pd.DataFrame) -> AccidentPreprocessingResult:
    """
    This function processes the data for a given year.
    It reads the data, calls processing functions, performs merges,
    and saves the final C_merged table for the specified year.
    """
    df_circumstances_renamed = na.rename_circumstances(df_circumstances)
    df_locations_renamed = na.rename_locations(df_locations)
    df_vehicles_renamed = na.rename_vehicles(df_vehicles)
    df_users_renamed = na.rename_users(df_users)

    # --- Call processing functions from D_feature_engineering.py ---
    df_circumstances_processed = df_circumstances_renamed
    df_users_processed = me.process_users(df_users_renamed, df_circumstances_renamed[['id_accident', 'year']])
    df_vehicles_processed = me.process_vehicles(df_vehicles_renamed)
    df_locations_processed = me.process_locations(df_locations_renamed)

    # --- Perform merges using the processed dataframes ---
    vehicle_categories_involved_processed = me.create_vehicle_dummies(df_vehicles_processed)
    df_user_vehicle_processed = me.process_user_vehicle_merge(df_users_processed, df_vehicles_processed)

    # --- Merge Tables ---
    merged_table = (
        df_circumstances_processed
        .copy()
        .merge(df_locations_processed, on='id_accident', how='left')
        .merge(df_user_vehicle_processed, on='id_accident', how='left')
        .merge(vehicle_categories_involved_processed, on='id_accident', how='left')
    )

    # --- Feature Engineering Pipeline ---
    print(f"Starting Feature Engineering for year {df_circumstances_renamed['year'].iloc[0]}...")
    feature_engineering_table = fe.create_datetime_features(merged_table)
    feature_engineering_table = fe.create_age_features(feature_engineering_table)
    feature_engineering_table = fe.create_security_equipment_one_hot(feature_engineering_table)
    feature_engineering_table = fe.create_vehicle_features(feature_engineering_table)
    feature_engineering_table = fe.create_road_complexity_index(feature_engineering_table)
    feature_engineering_table = fe.create_surface_quality_indicator(feature_engineering_table)
    feature_engineering_table = fe.create_user_role(feature_engineering_table)
    feature_engineering_table = fe.create_ordinal_conditions(feature_engineering_table)
    feature_engineering_table = fe.create_ordinal_target(feature_engineering_table)
    print("Feature Engineering complete.")

    # --- Imputation Step ---
    print("Starting Imputation...")
    imputed_table = hmv.impute_missing_values(feature_engineering_table)
    print("Imputation complete.")

    # --- Feature Selection ---
    feature_selection_table = fs.select_features(imputed_table)
    print("Feature Selection complete.")

    # --- Dtype Conversion ---
    print("Converting final features to int16 for memory efficiency...")

    # This list contains all columns you provided
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
        'lighting_ordinal', 'weather_ordinal', 'injury_target', 'sex', 'day_of_week', 'speed_limit', 'age'
    ]

    # Filter list to only include columns that survived feature selection
    existing_cols_to_convert = [col for col in columns_to_int16 if col in feature_selection_table.columns]

    # Perform the conversion
    if existing_cols_to_convert:
        feature_selection_table[existing_cols_to_convert] = feature_selection_table[existing_cols_to_convert].astype(
            'int16')

    print("Dtype conversion complete.")

    # --- Verification Step ---
    # Check for any NaNs that slipped through the targeted imputation
    remaining_nans = feature_selection_table.isnull().sum()
    remaining_nans = remaining_nans[remaining_nans > 0]  # Filter for columns that still have NaNs

    if not remaining_nans.empty:
        print("\n--- WARNING: Missing values found after imputation and feature selection ---")
        print("The following columns still contain NaNs:")
        print(remaining_nans)
        print("--------------------------------------------------------\n")

    # --- Updated return dictionary ---
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
        'F_feature_selection': feature_selection_table
    }


def process_files(input_path: str, output_path: str = 'data'):
    # Use the input_path argument instead of a hardcoded string
    files_path = glob.glob(f'{input_path}/*-*.csv')
    years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files_path)))))

    if not years:
        # Make the error message dynamic using the input_path
        print(f"No year-specific data files found in '{input_path}' directory.")
        print("Please ensure files are named like 'locations-2022.csv'.")
    else:
        print(f"Found data for years: {years}")

        # Create the output path if not existent
        try:
            output_path: Path = Path(output_path)
        except FileExistsError:
            print(f'The output folder "{output_path}" already exists. Skipping...')
        except PermissionError:
            print(f'Insufficient permissions to add the folder "{output_path}. Aborting')
            return
        except Exception as e:
            print(f'An unexpected error occured while trying to create the output folder "{output_path}". Aborting.')  #
            return

        for year in years:
            try:
                print(f"Processing data for year: {year}")
                df_circumstances = pd.read_csv(f'{input_path}/characteristics-{year}.csv', sep=';', decimal=',', low_memory=False)
                df_locations = pd.read_csv(f'{input_path}/locations-{year}.csv', sep=';', decimal=',', low_memory=False)
                df_users = pd.read_csv(f'{input_path}/users-{year}.csv', sep=';', decimal=',', low_memory=False)
                df_vehicles = pd.read_csv(f'{input_path}/vehicles-{year}.csv', sep=';', decimal=',', low_memory=False)

                results: AccidentPreprocessingResult = process_year(df_circumstances, df_locations, df_users,
                                                                    df_vehicles)
                folders = results.keys()

                for folder_name in folders:
                    folder_path = output_path / folder_name
                    folder_path.mkdir(exist_ok=True)

                    if folder_name == 'B_renamed':
                        year_path = folder_path / year
                        year_path.mkdir(exist_ok=True)

                        for key, df in results['B_renamed'].items():
                            file_path: Path = year_path / f'{key}_renamed-{year}.csv'
                            df.to_csv(file_path, sep=';', index=False)

                    else:
                        # Remove the 'C_', 'D_', 'E_', 'F_' prefix from the filename
                        # e.g., 'C_merged' -> 'merged'
                        filename_base = folder_name.split('_', 1)[1]
                        file_path: Path = folder_path / f'{filename_base}-{year}.csv'
                        results[folder_name].to_csv(file_path, sep=';', index=False)

                print(f"Successfully processed and saved data for year {year}")

            except FileNotFoundError as e:
                print(f"Skipping year {year} due to missing file: {e}")
            except PermissionError:
                print(f'Insufficient permissions to access {output_path} or one of its subpaths. Aborting.')
                return
            except Exception as e:
                print(f'An unexpected Exception occured: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_path',
        help='The file path of the source files. (Default: "../data/A_original")',
        type=str,
        nargs='?',  # Makes this argument optional
        default='../data/A_original'  # The value used if the argument is missing
    )

    parser.add_argument(
        '-o', '--output_path',
        help='The path of the output folder. (Default: "../data/")',
        type=str,
        default='../data/'  # The value used if -o is not specified
    )

    args = parser.parse_args()
    process_files(args.input_path, args.output_path)
