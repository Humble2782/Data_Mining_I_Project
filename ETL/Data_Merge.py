# Data_Merge.py
# Description: This script orchestrates the reading, processing, merging, and saving of accident data.
import pandas as pd
import argparse
import glob
import re
from pathlib import Path
from typing import TypedDict

# Import the new processing functions
import helpers.a_rename as na
import helpers.b_merge_tables as me
import helpers.c_feature_engineering as fe
import helpers.d_feature_selection as fs

# Typing
class AccidentDataRenamed(TypedDict):
    circumstances: pd.DataFrame
    locations: pd.DataFrame
    vehicles: pd.DataFrame
    users: pd.DataFrame

class AccidentPreprocessingResult(TypedDict):
    renamed: AccidentDataRenamed
    merged: pd.DataFrame
    feature_engineering: pd.DataFrame
    feature_selection: pd.DataFrame


def process_year(df_circumstances: pd.DataFrame, df_locations: pd.DataFrame, df_users: pd.DataFrame,
                 df_vehicles: pd.DataFrame) -> AccidentPreprocessingResult:
    """
    This function processes the data for a given year.
    It reads the data, calls processing functions, performs merges,
    and saves the final merged table for the specified year.
    """
    df_circumstances_renamed = na.rename_circumstances(df_circumstances)
    df_locations_renamed = na.rename_locations(df_locations)
    df_vehicles_renamed = na.rename_vehicles(df_vehicles)
    df_users_renamed = na.rename_users(df_users)

    # --- Call processing functions from feature_engineering.py ---
    df_circumstances_processed = df_circumstances_renamed
    df_users_processed = me.process_users(df_users_renamed)
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

    # Feature Engineering
    feature_engineering_table = fe.create_time_features(merged_table)
    feature_engineering_table = fe.create_age_feature(feature_engineering_table)
    feature_engineering_table = fe.create_security_equipment_one_hot(feature_engineering_table)

    feature_selection_table = fs.select_features(feature_engineering_table)

    return {
        'renamed': {
            'circumstances': df_circumstances_renamed,
            'locations': df_locations_renamed,
            'vehicles': df_vehicles_renamed,
            'users': df_users_renamed
        },
        'merged': merged_table,
        'feature_engineering': feature_engineering_table,
        'feature_selection': feature_selection_table
    }


def process_files(input_path: str, output_path: str = 'data'):
    input_path: Path = Path(input_path)

    files_path = glob.glob(f'{input_path}/*-*.csv')
    years = sorted(list(set(re.findall(r'-(\d{4})\.csv', ' '.join(files_path)))))

    if not years:
        print("No year-specific data files found in 'data/' directory.")
        print("Please ensure files are named like 'locations-2022.csv'.")
    else:
        print(f"Found data for years: {years}")

        # Create the output path if not existent
        try:
            output_path: Path = Path(output_path)
            output_path.mkdir(parents=True)
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
                df_circumstances = pd.read_csv(f'{input_path}/characteristics-{year}.csv', sep=';', decimal=',')
                df_locations = pd.read_csv(f'{input_path}/locations-{year}.csv', sep=';', decimal=',')
                df_users = pd.read_csv(f'{input_path}/users-{year}.csv', sep=';', decimal=',')
                df_vehicles = pd.read_csv(f'{input_path}/vehicles-{year}.csv', sep=';', decimal=',')

                results: AccidentPreprocessingResult = process_year(df_circumstances, df_locations, df_users,
                                                                    df_vehicles)
                folders = results.keys()

                for folder_name in folders:
                    folder_path = output_path / folder_name
                    folder_path.mkdir(exist_ok=True)

                    if folder_name == 'renamed':
                        year_path = folder_path / year
                        year_path.mkdir(exist_ok=True)

                        for key, df in results['renamed'].items():
                            file_path: Path = year_path / f'{key}_renamed-{year}.csv'
                            df.to_csv(file_path, sep=';', index=False)

                    else:
                        file_path: Path = folder_path / f'{folder_name}-{year}.csv'
                        results[folder_name].to_csv(file_path, sep=';', index=False)

                print(f"Successfully saved merged data for year {year} to {year_path}")

            except FileNotFoundError as e:
                print(f"Skipping year {year} due to missing file: {e}")
            except PermissionError:
                print(f'Insufficient permissions to access {output_path} or one of its subpaths. Aborting.')
                return
            #except Exception as e:
             #   print(f'An unexpected Exception occured: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help='The file path of the source files.', type=str)
    parser.add_argument('-o', '--output_path', help='The path of the output folder.', type=str)
    args = parser.parse_args()

    process_files(args.input_path, args.output_path)
