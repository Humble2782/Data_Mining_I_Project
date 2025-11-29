import pandas as pd

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

    # Impute N/A (-1) for all 'other' columns when other_vehicle == 0
    cols_to_impute_other = [
        'initial_point_of_impact_other',
        'vehicle_category_other',
        'main_maneuver_before_accident_other',
        'motor_type_other',
        'direction_of_travel_other',  # <-- Added from screenshot
        'fixed_obstacle_struck_other',  # <-- Added from screenshot
        'mobile_obstacle_struck_other'  # <-- Added from screenshot
    ]

    for col in cols_to_impute_other:
        if col in df.columns:
            is_na = df[col].isna()
            is_no_other_vehicle = (df['other_vehicle'] == 0)

            # Fill with -1 (N/A) if no other vehicle
            if col == 'vehicle_category_other':
                # Special case for category column: fill with 'none'
                df.loc[is_na & is_no_other_vehicle, col] = "none"
            else:
                # Fill numeric columns with -1
                df.loc[is_na & is_no_other_vehicle, col] = -1

                # --- Handle "Unknown" (0) for cases WITH another vehicle ---
                # These columns should be 0 (Unknown) if another vehicle exists but data is missing

                cols_to_impute_unknown = [
                    'main_maneuver_before_accident_other',
                    'motor_type_other',
                    'direction_of_travel_other',
                    'fixed_obstacle_struck_other',
                    'mobile_obstacle_struck_other'
                ]

                if col in cols_to_impute_unknown:
                    # Fill with 0 (Unknown) if NaN AND a 2nd vehicle exists
                    df.loc[is_na & (~is_no_other_vehicle), col] = 0

        else:
            print(f"Warning: Column '{col}' not found in impute_nans_other_vehicle. Skipping.")

    # --- 2. Drop all remaining NaNs at the end ---
    # These columns MUST have a value after imputation (either real, 'none', or -1)
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
    Includes a final "safety-net" imputation to remove all remaining NaNs.
    """
    print("  Handling missing values (Imputation & Filtering)...")
    # It's important to use .copy() when starting the chain here
    df_copy = df.copy()
    df_copy = impute_nans_own_vehicle(df_copy)
    df_copy = impute_nans_other_vehicle(df_copy)

    # CRITICAL: Drop rows where the target variable is missing
    # This MUST be done before the safety-net imputation.
    original_rows = len(df_copy)
    if 'injury_target' in df_copy.columns:
        df_copy = df_copy.dropna(subset=['injury_target'])
        rows_dropped = original_rows - len(df_copy)
        if rows_dropped > 0:
            print(f"  ...Dropped {rows_dropped} rows with missing target variable.")

    # --- Safety-net imputation (Targeted) ---
    # Impute all remaining NaNs to ensure no model errors

    # 1. Fill known categorical NaNs (e.g., 'age_group' from screenshot)
    cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        # Check if the column is a pandas Categorical type
        if pd.api.types.is_categorical_dtype(df_copy[col]):
            # If it is, we MUST add 'Unknown' as a valid category first
            if 'Unknown' not in df_copy[col].cat.categories:
                df_copy[col] = df_copy[col].cat.add_categories('Unknown')

        # Now we can safely fill the NaNs (works for 'object' and 'category')
        df_copy[col] = df_copy[col].fillna('Unknown')

    # 2. Fill specific known numerical NaNs
    # We use 0 as a neutral default, 'age_group' is already 'Unknown'
    if 'age' in df_copy.columns:
        df_copy.loc[df_copy['age'].isna(), 'age'] = 0

    print("  ...Handling missing values complete.")
    return df_copy