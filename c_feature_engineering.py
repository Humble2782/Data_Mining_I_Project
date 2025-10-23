# c_feature_engineering.py
# Description: Reads a merged table, creates new valuable features, and saves the result.
# This script ONLY adds columns, it does not remove them.

import pandas as pd
import numpy as np
import argparse
import glob
import os
from sklearn.preprocessing import MinMaxScaler

def create_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplifies vehicle categories and calculates an impact score.
    This logic was crucial and has been re-integrated.
    """
    print("  Creating simplified vehicle features and impact scores...")
    df_copy = df.copy()

    def simplify_vehicle_category(cat_id):
        if pd.isna(cat_id): return 'unknown'
        # Bicycles
        if cat_id in {1, 80}: return "bicycle"
        # Powered 2-3 wheelers
        if cat_id in {2, 30, 41, 31, 32, 33, 34, 42, 43}: return "powered_2_3_wheeler"
        # Light motor vehicles
        if cat_id in {7, 10}: return "light_motor_vehicle"
        # Heavy goods vehicles / trucks
        if cat_id in {13, 14, 15, 16, 17}: return "hgv_truck"
        # Bus or coach
        if cat_id in {37, 38}: return "bus_coach"
        return "other"

    IMPACT_WEIGHTS = {
        "hgv_truck": 6, "bus_coach": 5, "light_motor_vehicle": 4,
        "powered_2_3_wheeler": 3, "bicycle": 2, "other": 1, "unknown": 1,
    }

    if 'vehicle_category' in df_copy.columns:
        df_copy['vehicle_category_simplified'] = df_copy['vehicle_category'].apply(simplify_vehicle_category)
        df_copy['impact_score'] = df_copy['vehicle_category_simplified'].map(IMPACT_WEIGHTS).fillna(1)

    return df_copy

def create_road_complexity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 'road_complexity_index' by assigning weighted scores to
    different road characteristics, based on the verified official documentation.
    """
    print("  Creating road complexity index...")
    df_copy = df.copy()

    # 1. Define weights based on the provided official categories.
    # Higher values indicate higher complexity.

    # Intersection (`int`):
    intersection_weights = {
        1: 0,  # Outside intersection
        2: 4,  # X-intersection
        3: 3,  # T-intersection
        4: 3,  # Y-intersection
        5: 6,  # Intersection with >4 branches
        6: 7,  # Roundabout
        7: 8,  # Place / Square (often chaotic)
        8: 5,  # Railway crossing (specific, high risk)
        9: 2  # Other intersection
    }

    # Road Category (`catr`):
    road_category_weights = {
        1: 2,  # Motorway (less complex due to separation)
        2: 3,  # National Road
        3: 4,  # Departmental Road
        4: 5,  # Communal Way (often narrow, complex urban areas)
        5: 1,  # Off public network
        6: 3,  # Public parking lot (slow, but complex maneuvers)
        7: 4,  # Urban metropolis roads
        9: 1  # Other
    }

    # Traffic Regime (`circ`):
    traffic_regime_weights = {
        -1: 1,  # Not specified (treat as low complexity)
        1: 1,  # One-way
        2: 4,  # Two-way (potential for head-on collisions)
        3: 2,  # Separated carriageways
        4: 6  # With variable assignment lanes (high complexity)
    }

    # 2. Map the weights to the columns.
    # .fillna(1) assigns a low complexity score to any unlisted categories.
    df_copy['lanes_score'] = pd.to_numeric(df_copy['number_of_traffic_lanes'], errors='coerce').fillna(1).clip(upper=8)
    df_copy['intersection_score'] = df_copy['intersection'].map(intersection_weights).fillna(1)
    df_copy['road_cat_score'] = df_copy['road_category'].map(road_category_weights).fillna(1)
    df_copy['traffic_score'] = df_copy['traffic_regime'].map(traffic_regime_weights).fillna(1)

    # 3. Sum the individual scores to create the total index.
    df_copy['road_complexity_index'] = (
            df_copy['lanes_score'] +
            df_copy['intersection_score'] +
            df_copy['road_cat_score'] +
            df_copy['traffic_score']
    )

    # 4. Scale the index to a more interpretable range (e.g., 0 to 10).
    scaler = MinMaxScaler(feature_range=(0, 10))
    df_copy['road_complexity_index'] = scaler.fit_transform(df_copy[['road_complexity_index']])

    # 5. Remove the temporary score columns.
    df_copy.drop(columns=['lanes_score', 'intersection_score', 'road_cat_score', 'traffic_score'], inplace=True)

    return df_copy

def create_surface_quality_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary 'surface_quality_indicator' based on pavement
    condition and the road's longitudinal profile.

    The indicator is 1 (good) only if the surface is normal AND the profile is flat.
    Otherwise, it is 0 (bad).
    """
    print("  Creating surface quality indicator...")
    df_copy = df.copy()

    # Define the conditions for a "good" road surface based on official codes
    is_pavement_good = (df_copy['pavement_condition'] == 1)
    is_profile_good = (df_copy['longitudinal_profile'] == 1)

    # Combine the conditions: both must be true for the road to be considered good.
    # The .astype(int) converts the boolean result (True/False) to 1/0.
    df_copy['surface_quality_indicator'] = (is_pavement_good & is_profile_good).astype(int)

    return df_copy

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates age and bins it into descriptive groups."""
    print("  Creating age features...")
    df_copy = df.copy()
    if 'year_of_birth' in df_copy.columns and 'year' in df_copy.columns:
        df_copy['age'] = df_copy['year'] - df_copy['year_of_birth']
        bins = [0, 18, 25, 40, 65, 100]
        labels = ['child_teen', 'young_adult', 'adult', 'middle_aged', 'senior']
        df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    return df_copy


def create_safety_equipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary flags for used safety equipment."""
    print("  Creating safety equipment features...")
    df_copy = df.copy()
    safety_cols = ['safety_equipment_1', 'safety_equipment_2', 'safety_equipment_3']
    for col in safety_cols:
        if col not in df_copy.columns: df_copy[col] = 0
    df_copy[safety_cols] = df_copy[safety_cols].fillna(0)
    df_copy['used_belt'] = (df_copy[safety_cols] == 1).any(axis=1).astype(int)
    df_copy['used_helmet'] = (df_copy[safety_cols] == 2).any(axis=1).astype(int)
    return df_copy

def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates meaningful cyclical features from date columns (day, month, year).
    This function now correctly interprets the date components.
    """
    print("  Creating cyclical date features...")
    df_copy = df.copy()

    # --- 1. Create a datetime object from the individual columns ---
    # This is the crucial first step to correctly interpret the date.
    # We use 'coerce' to handle any potential errors in the date columns gracefully.
    date_cols = ['year', 'month', 'day']
    if all(col in df_copy.columns for col in date_cols):
        df_copy['datetime'] = pd.to_datetime(df_copy[date_cols], errors='coerce')

        # Drop rows where the date could not be parsed
        df_copy.dropna(subset=['datetime'], inplace=True)

        # --- 2. Create cyclical features for WEEKLY seasonality ---
        # Monday=0, Sunday=6
        df_copy['day_of_week'] = df_copy['datetime'].dt.dayofweek
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

        # --- 3. Create cyclical features for YEARLY seasonality (based on month) ---
        # Using the existing 'month' column
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)

        # --- 4. Create cyclical features for YEARLY seasonality (based on day of year) ---
        # This is more granular than just the month
        df_copy['day_of_year'] = df_copy['datetime'].dt.dayofyear
        df_copy['day_of_year_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365.25)
        df_copy['day_of_year_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365.25)

        # We can now drop the temporary columns
        df_copy.drop(columns=['datetime', 'day_of_week', 'day_of_year'], inplace=True)

        print("    -> Successfully created cyclical features for day of week and year.")

    else:
        print("    -> Could not create cyclical features because 'year', 'month', or 'day' column is missing.")

    return df_copy


def run_feature_engineering():
    """Main function to run the entire feature engineering pipeline automatically."""
    INPUT_FOLDER = 'data/C_full'
    OUTPUT_FOLDER = 'data/D_engineered'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files_to_process = glob.glob(f'{INPUT_FOLDER}/full_table_*.csv')

    if not files_to_process:
        print(f"No 'full_table_*.csv' files found in '{INPUT_FOLDER}'. Please run table_merge.py first.")
        return

    for filepath in files_to_process:
        print(f"\nStarting feature engineering for {filepath}...")
        try:
            df = pd.read_csv(filepath, sep=';', low_memory=False)
        except FileNotFoundError:
            print(f"Error: Input file not found at {filepath}")
            continue

        # Apply all feature creation functions
        df = create_vehicle_features(df)
        df = create_age_features(df)
        df = create_safety_equipment_features(df)
        df = create_cyclical_features(df)
        df = create_road_complexity_index(df)
        df = create_surface_quality_indicator(df)

        # Construct output path
        filename = os.path.basename(filepath)
        output_filename = filename.replace('full_table_', 'engineered_table_')
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        df.to_csv(output_path, sep=';', index=False)
        print(f"Feature engineering complete. Saved to {output_path}")


if __name__ == "__main__":
    run_feature_engineering()