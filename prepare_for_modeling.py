# prepare_for_modeling_compact.py
import pandas as pd
import numpy as np
import glob


def preprocess_for_modeling_compact(input_filepath, category_threshold=0.01):
    """
    Loads a merged 'final_table' and performs preprocessing with a strategy
    to limit the number of columns created from categorical features.

    Args:
        input_filepath (str): Path to the input CSV file.
        category_threshold (float): Percentage of total rows a category must have
                                    to not be grouped into 'other'.
                                    (e.g., 0.01 means 1%).
    """
    print(f"Processing file: {input_filepath}...")

    try:
        df = pd.read_csv(input_filepath, sep=';')
    except FileNotFoundError:
        print(f"Error: File not found at {input_filepath}")
        return

    # --- 1. Drop Irrelevant Columns ---
    columns_to_drop = [
        'id_accident', 'year', 'day', 'department_code', 'commune_code',
        'postal_address', 'latitude', 'longitude', 'road_number',
        'road_number_index', 'road_number_letter', 'nearest_reference_marker',
        'nearest_reference_marker_distance', 'width_central_reservation',
        'carriageway_width', 'id_user', 'id_vehicle', 'number_vehicle_x',
        'number_vehicle_y', 'year_of_birth', 'trip_purpose', 'pedestrian_location',
        'pedestrian_action', 'injured_pedestrian_alone',
        'number_occupants_in_public_transport', 'id_vehicle_other', 'number_vehicle_other'
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_processed = df.drop(columns=existing_columns_to_drop)
    print(f"  Dropped {len(existing_columns_to_drop)} irrelevant columns.")

    # --- 2. Process Safety Equipment ---
    safety_cols = ['safety_equipment_1', 'safety_equipment_2', 'safety_equipment_3']
    for col in safety_cols:
        if col not in df_processed.columns: df_processed[col] = 0
    df_processed[safety_cols] = df_processed[safety_cols].fillna(0)
    df_processed['used_belt'] = ((df_processed[safety_cols] == 1).any(axis=1)).astype(int)
    df_processed['used_helmet'] = ((df_processed[safety_cols] == 2).any(axis=1)).astype(int)
    df_processed['used_child_seat'] = ((df_processed[safety_cols] == 3).any(axis=1)).astype(int)
    df_processed['used_reflective_vest'] = ((df_processed[safety_cols] == 4).any(axis=1)).astype(int)
    df_processed = df_processed.drop(columns=safety_cols)
    print("  Processed safety equipment features.")

    # --- 3. Handle Cyclical Features (Weekday) ---
    if 'weekday' in df_processed.columns:
        df_processed['weekday_sin'] = np.sin(2 * np.pi * df_processed['weekday'] / 7.0)
        df_processed['weekday_cos'] = np.cos(2 * np.pi * df_processed['weekday'] / 7.0)
        df_processed = df_processed.drop('weekday', axis=1)
        print("  Encoded 'weekday' as a cyclical feature.")

    # --- 4. Bin 'age' feature ---
    if 'age' in df_processed.columns:
        if df_processed['age'].isnull().any():
            df_processed['age'].fillna(df_processed['age'].median(), inplace=True)
        bins = [0, 18, 25, 40, 65, 100]
        labels = ['child_teen', 'young_adult', 'adult', 'middle_aged', 'senior']
        df_processed['age_group'] = pd.cut(df_processed['age'], bins=bins, labels=labels, right=False)
        df_processed = df_processed.drop('age', axis=1)
        print("  Binned 'age' into 'age_group'.")

    # --- 5. Smartly Encode Categorical Features ---
    categorical_cols_to_encode = [
        'atmospheric_conditions', 'lighting_conditions', 'road_category', 'traffic_regime',
        'longitudinal_profile', 'horizontal_alignment', 'pavement_condition',
        'infrastructure', 'accident_situation', 'user_category', 'gender', 'gravity',
        'fixed_obstacle_struck', 'mobile_obstacle_struck', 'initial_point_of_impact',
        'main_maneuver_before_accident', 'motor_type', 'age_group',
        'vehicle_category', 'vehicle_category_other'  # Add other known object/categorical columns
    ]

    # Filter list to only include columns present in the dataframe
    categorical_cols = [col for col in categorical_cols_to_encode if col in df_processed.columns]

    # Group rare categories
    for col in categorical_cols:
        # Calculate frequency of each category
        counts = df_processed[col].value_counts(normalize=True)
        # Identify rare categories (below threshold)
        rare_categories = counts[counts < category_threshold].index
        # Group them into 'other'
        if len(rare_categories) > 0:
            df_processed[col] = df_processed[col].replace(rare_categories, 'other')
            print(f"  Grouped {len(rare_categories)} rare categories in '{col}' into 'other'.")

    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, dummy_na=False)
    print(f"  One-hot encoded {len(categorical_cols)} categorical features with grouping.")

    # --- Save the processed data ---
    output_filename = input_filepath.replace('/full/full_table_', '/model_ready/model_ready_')
    df_encoded.to_csv(output_filename, index=False)

    print(f"\nPreprocessing complete! The final data has {df_encoded.shape[1]} columns.")
    print(f"Transformed data saved to '{output_filename}'.\n")

if __name__ == "__main__":
    # You can adjust the threshold here. 0.01 means categories that make up less than 1%
    # of the data will be grouped. Increase it to 0.02 (2%) to be more aggressive
    # and create even fewer columns.
    THRESHOLD = 0.01

    files_to_process = glob.glob('data/full/full_table_*.csv')

    if not files_to_process:
        print("No 'full_table_*.csv' files found in the 'data/full/' directory.")
    else:
        for f in files_to_process:
            preprocess_for_modeling_compact(f, category_threshold=THRESHOLD)