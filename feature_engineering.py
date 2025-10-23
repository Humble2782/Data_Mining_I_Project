# feature_engineering.py
# Description: Reads a merged table, creates new valuable features, and saves the result.
# This script ONLY adds columns, it does not remove them.

import pandas as pd
import numpy as np
import argparse
import glob
import os


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
    """Encodes cyclical temporal features using sine and cosine."""
    print("  Creating cyclical features for weekday...")
    df_copy = df.copy()
    if 'day' in df_copy.columns:  # Assuming 'day' is the weekday
        df_copy['weekday_sin'] = np.sin(2 * np.pi * df_copy['day'] / 7.0)
        df_copy['weekday_cos'] = np.cos(2 * np.pi * df_copy['day'] / 7.0)
    return df_copy


def run_feature_engineering():
    """Main function to run the entire feature engineering pipeline automatically."""
    INPUT_FOLDER = 'data/full'
    OUTPUT_FOLDER = 'data/engineered'

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

        # Construct output path
        filename = os.path.basename(filepath)
        output_filename = filename.replace('full_table_', 'engineered_table_')
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        df.to_csv(output_path, sep=';', index=False)
        print(f"Feature engineering complete. Saved to {output_path}")


if __name__ == "__main__":
    run_feature_engineering()