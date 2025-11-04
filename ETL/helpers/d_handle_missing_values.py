import pandas as pd
import numpy as np


def impute_nans_own_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values for the 'own' vehicle and pedestrian-related columns.
    Uses .loc for assignments and runs dropna() at the end to avoid SettingWithCopyWarning.
    """

    # --- 1. Impute all columns first ---

    # Impute obstacles: -1 for pedestrians (N/A)
    cols_to_be_cleaned_obstacle = ['initial_point_of_impact', "fixed_obstacle_struck", "mobile_obstacle_struck"]
    for col in cols_to_be_cleaned_obstacle:
        if col in df.columns:
            df.loc[
                (df['role'] == 'pedestrian') & (df[col].isna()),
                col
            ] = -1
        else:
            print(f"Warning: Column '{col}' not found in impute_nans_own_vehicle. Skipping.")

    # Impute vehicle category: 'none' for pedestrians (N/A)
    if 'vehicle_category' in df.columns:
        df.loc[
            (df['vehicle_category'].isna()) & (df['role'] == 'pedestrian'),
            'vehicle_category'
        ] = 'none'

    # Impute 'own' vehicle stats: -1 for N/A roles, 0 for Unknown
    # 'impact_score' has been removed from this list
    cols_to_be_cleaned_own = ['main_maneuver_before_accident', 'motor_type']
    is_na_role = df['role'].isin(['pedestrian', 'passenger'])

    for col in cols_to_be_cleaned_own:
        if col in df.columns:
            is_na = df[col].isna()
            # First, fill N/A roles (pedestrian, passenger) with -1
            df.loc[is_na & is_na_role, col] = -1
            # Then, fill active roles (driver, other) with 0 (Unknown)
            df.loc[is_na & (~is_na_role), col] = 0
        else:
            print(f"Warning: Column '{col}' not found in impute_nans_own_vehicle. Skipping.")

    # --- 2. Drop all other NaN Values at the end ---
    # Define columns where NaNs are not allowed after imputation
    cols_to_drop_na_from = ['initial_point_of_impact', "fixed_obstacle_struck",
                            "mobile_obstacle_struck", 'vehicle_category']
    # Ensure only existing columns are used
    final_drop_cols = [col for col in cols_to_drop_na_from if col in df.columns]
    df = df.dropna(subset=final_drop_cols, how='any')

    return df


def impute_nans_other_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values for the 'other' vehicle columns.
    Uses .loc and moves dropna() to the end.
    Includes a check for column existence to prevent KeyErrors.
    """

    # --- 1. Impute all columns first ---
    # Create the 'other_vehicle' flag based on the merged 'id_vehicle_other'
    if 'id_vehicle_other' in df.columns:
        df['other_vehicle'] = df['id_vehicle_other'].notna().astype(int)
    else:
        # Fallback if 'id_vehicle_other' is missing
        print("Warning: 'id_vehicle_other' column not found. 'other_vehicle' imputation might be incorrect.")
        df['other_vehicle'] = 0  # Assume no other vehicle if column is missing

    # Impute initial impact other: -1 if no other vehicle
    if 'initial_point_of_impact_other' in df.columns:
        df.loc[
            (df['initial_point_of_impact_other'].isna()) & (df['other_vehicle'] == 0),
            'initial_point_of_impact_other'
        ] = -1

    # Impute vehicle category other: 'none' if no other vehicle
    if 'vehicle_category_other' in df.columns:
        df.loc[
            (df['vehicle_category_other'].isna()) & (df['other_vehicle'] == 0),
            'vehicle_category_other'
        ] = "none"

    # Impute 'other' vehicle stats: -1 for no other vehicle, 0 for Unknown
    # 'impact_score_other' has been removed from this list
    cols_to_be_cleaned_other = ['main_maneuver_before_accident_other', 'motor_type_other']

    for col in cols_to_be_cleaned_other:
        # Check if column exists before using it
        if col in df.columns:
            is_na = df[col].isna()
            # -1 if no other vehicle is involved (N/A)
            df.loc[is_na & (df['other_vehicle'] == 0), col] = -1
            # 0 if another vehicle is present, but the value is missing (Unknown)
            df.loc[is_na & (df['other_vehicle'] == 1), col] = 0
        else:
            # Inform user that column is missing, but don't crash
            print(f"Warning: Column '{col}' not found in impute_nans_other_vehicle. Skipping.")

    # --- 2. Drop all remaining NaNs at the end ---
    cols_to_drop_na_from = ['initial_point_of_impact_other', 'vehicle_category_other']
    final_drop_cols = [col for col in cols_to_drop_na_from if col in df.columns]
    if final_drop_cols:  # Only drop if list is not empty
        df = df.dropna(subset=final_drop_cols, how='any')

    # Drop the temporary flag
    df = df.drop(columns=['other_vehicle'], errors='ignore')
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to run all imputation steps.
    """
    print("  Handling missing values (Imputation & Filtering)...")
    # It's important to use .copy() when starting the chain here
    df_copy = df.copy()
    df_copy = impute_nans_own_vehicle(df_copy)
    df_copy = impute_nans_other_vehicle(df_copy)
    print("  ...Handling missing values complete.")
    return df_copy
