# d_feature_selection.py
# Description: Automatically finds D_engineered data files, selects the final features
# by dropping irrelevant columns, and saves the model-ready data.

import pandas as pd
import glob
import os


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are irrelevant for modeling, such as IDs or redundant source columns.
    """
    print("  Dropping irrelevant and redundant columns...")

    columns_to_drop = [
        # IDs and high-cardinality identifiers
        'id_accident', 'id_vehicle', 'id_user', 'number_vehicle', 'department_code',
        'commune_code', 'postal_address', 'road_number', 'road_number_index',
        'road_number_letter',

        # little information and a lot of missing values
        'width_central_reservation', 'number_occupants_in_public_transport',
        'id_vehicle_other', 'nearest_reference_marker_distance', 'nearest_reference_marker',

        # Original columns that have been engineered into new features
        'year_of_birth',
        'day', 'month', 'year'
        'vehicle_category',  # Replaced by 'vehicle_category_simplified' and 'impact_score'
        'safety_equipment_1', 'safety_equipment_2', 'safety_equipment_3',

        # Other columns deemed not useful for modeling
        'injured_pedestrian_alone', 'trip_purpose'
    ]

    # Identify which of the columns to drop actually exist in the DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_selected = df.drop(columns=existing_columns_to_drop)

    print(f"  Dropped {len(existing_columns_to_drop)} columns.")
    return df_selected


def run_feature_selection():
    """
    Main function to automatically find and process files.
    """
    INPUT_FOLDER = '../data/D_engineered'
    OUTPUT_FOLDER = '../data/E_modeling_base'

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Find all files created by the feature engineering step
    files_to_process = glob.glob(f'{INPUT_FOLDER}/engineered_table_*.csv')

    if not files_to_process:
        print(f"No 'engineered_table_*.csv' files found in '{INPUT_FOLDER}'.")
        print("Please run c_feature_engineering.py first.")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for filepath in files_to_process:
        print(f"\nStarting feature selection for {filepath}...")
        try:
            df_engineered = pd.read_csv(filepath, sep=';', low_memory=False)
        except Exception as e:
            print(f"  Could not read file. Error: {e}")
            continue

        # Apply the column dropping function
        df_final = drop_irrelevant_columns(df_engineered)

        # Create the new filename and save the final data
        filename = os.path.basename(filepath)
        output_filename = filename.replace('engineered_table_', 'modeling_base_')
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        df_final.to_csv(output_path, sep=';', index=False)
        print(f"Feature selection complete. Final data saved to {output_path} with shape {df_final.shape}.")


if __name__ == "__main__":
    run_feature_selection()