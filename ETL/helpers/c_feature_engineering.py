import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes


def create_datetime_features(df_merged: pd.DataFrame):
    """
    Creates a comprehensive set of time and date-based features.
    This includes buckets (time_of_day) and cyclical features (for hour,
    day of week, month, and day of year).
    """
    print("  Creating comprehensive datetime features...")
    df_copy = df_merged.copy()

    # --- 1. Handle Intra-day Time (from 'hour_minute') ---
    if 'hour_minute' in df_copy.columns:
        # Convert 'hour_minute' to datetime object to extract hour and minute
        datetime_series = pd.to_datetime(df_copy['hour_minute'], format='%H:%M')
        df_copy['hour'] = datetime_series.dt.hour
        df_copy['minute'] = datetime_series.dt.minute
        df_copy.drop(columns='hour_minute', inplace=True)  # Drop the original string column

        # Create cyclical hour features (for linear models, NNs)
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24.0)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24.0)

        # Create time_of_day buckets (for tree-based models)
        bins = [-1, 6, 10, 16, 20, 24]
        labels = ['Night', 'Morning_Rush', 'Midday', 'Evening_Rush', 'Night']

        # --- FIX ---
        # Added 'ordered=False' because the label 'Night' is used twice.
        # This tells pandas not to treat the bins as a strict ordered category.
        df_copy['time_of_day'] = pd.cut(df_copy['hour'], bins=bins, labels=labels, right=False, ordered=False)
    else:
        print("    ! 'hour_minute' column not found. Skipping hour-based features.")

    # --- 2. Handle Inter-day Date (from 'year', 'month', 'day') ---
    date_cols = ['year', 'month', 'day']
    if all(col in df_copy.columns for col in date_cols):
        df_copy['datetime'] = pd.to_datetime(df_copy[date_cols], errors='coerce')
        df_copy.dropna(subset=['datetime'], inplace=True)

        # Weekly seasonality
        df_copy['day_of_week'] = df_copy['datetime'].dt.dayofweek
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

        # Monthly seasonality
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)

        # Yearly seasonality
        df_copy['day_of_year'] = df_copy['datetime'].dt.dayofyear
        df_copy['day_of_year_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365.25)
        df_copy['day_of_year_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365.25)

        # We keep 'day_of_week' for tree-based models
        # and only drop the temporary 'datetime' and 'day_of_year' columns.
        df_copy.drop(columns=['datetime', 'day_of_year'], inplace=True, errors='ignore')
    else:
        print("    ! 'year', 'month', or 'day' column is missing. Skipping date-based features.")

    return df_copy


def create_age_features(df_merged: pd.DataFrame):
    """Calculates age and bins it into descriptive groups."""
    print("  Creating age features...")
    df_copy = df_merged.copy()
    if 'year_of_birth' in df_copy.columns and 'year' in df_copy.columns:
        df_copy['age'] = df_copy['year'] - df_copy['year_of_birth']
        bins = [0, 18, 25, 40, 65, 100]
        labels = ['child_teen', 'young_adult', 'adult', 'middle_aged', 'senior']
        df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    return df_copy


def create_security_equipment_one_hot(df_merged: pd.DataFrame):
    df_merged_copy = df_merged.copy()

    SECU_FLAG = {
        1: "used_belt", 2: "used_helmet", 3: "used_child_restraint", 4: "used_reflective_vest",
        5: "used_airbag", 6: "used_gloves", 7: "used_gloves_and_airbag", 9: "used_other"
    }
    SECU_COLS = ["safety_equipment_1", "safety_equipment_2", "safety_equipment_3"]

    S = df_merged_copy[SECU_COLS].apply(pd.to_numeric, errors="coerce")

    for code, name in SECU_FLAG.items():
        df_merged_copy[name] = S.isin([code]).any(axis=1).astype(int)

    # Merge Airbag and Airbag + Gloves
    df_merged_copy['used_airbag'] = df_merged_copy[['used_airbag', 'used_gloves_and_airbag']].any(axis=1).astype(int)
    df_merged_copy['used_gloves'] = df_merged_copy[['used_gloves', 'used_gloves_and_airbag']].any(axis=1).astype(int)

    # Remove columns not needed anymore
    df_merged_copy.drop(columns='used_gloves_and_airbag', inplace=True)

    return df_merged_copy


def create_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplifies vehicle categories and calculates an impact score
    for BOTH the primary and the 'other' vehicle.

    - We know 'vehicle_category' is strings.
    - We ADD logic for 'vehicle_category_other'.
    - We ADD a 'n/a' key to the weights for single-vehicle accidents.
    - We FIX 'impact_delta' to be directional (removed .abs()).
    """
    print("  Creating simplified vehicle features and impact scores...")
    df_copy = df.copy()

    IMPACT_WEIGHTS = {
        "hgv_truck": 6, "bus_coach": 5, "light_motor_vehicle": 4,
        "powered_2_3_wheeler": 3, "bicycle": 2, "other": 1, "unknown": 1,
        "n/a": 0
    }

    # --- 1. Process PRIMARY Vehicle ---
    if 'vehicle_category' in df_copy.columns:
        print("    ...Processing 'vehicle_category' as strings.")

        # 1. Assign and fill NaNs (from pedestrians, etc.)
        # A primary user MUST be in a vehicle (or "unknown")
        df_copy['vehicle_category_simplified'] = df_copy['vehicle_category'].fillna('unknown')

        # 2. Strip whitespace (critical fix)
        df_copy['vehicle_category_simplified'] = df_copy['vehicle_category_simplified'].astype(str).str.strip()

        # 3. Map to weights
        df_copy['impact_score'] = df_copy['vehicle_category_simplified'].map(IMPACT_WEIGHTS).fillna(1)

    # --- 2. Process OTHER Vehicle ---
    if 'vehicle_category_other' in df_copy.columns:
        print("    ...Processing 'vehicle_category_other' as strings.")

        # 1. Assign and fill NaNs
        # A NaN here means "no other vehicle", so we use "n/a"
        df_copy['vehicle_category_simplified_other'] = df_copy['vehicle_category_other'].fillna('none')

        # 2. Strip whitespace
        df_copy['vehicle_category_simplified_other'] = df_copy['vehicle_category_simplified_other'].astype(
            str).str.strip()

        # 3. Map to weights
        # "n/a" will map to 0.
        df_copy['impact_score_other'] = df_copy['vehicle_category_simplified_other'].map(IMPACT_WEIGHTS).fillna(1)

    # --- 3. Create Interaction Feature ---
    # Calculate the *directional* difference in impact scores
    # A negative value means "my vehicle" < "other vehicle" (higher risk)
    if 'impact_score' in df_copy.columns and 'impact_score_other' in df_copy.columns:
        print("    ...Calculating impact difference (delta).")
        df_copy['impact_delta'] = (df_copy['impact_score'] - df_copy['impact_score_other'])

    return df_copy

def create_road_complexity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 'road_complexity_index' by assigning weighted scores to
    different road characteristics.
    """
    print("  Creating road complexity index...")
    df_copy = df.copy()

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

    # Map the weights
    df_copy['lanes_score'] = pd.to_numeric(df_copy['number_of_traffic_lanes'], errors='coerce').fillna(1).clip(upper=8)
    df_copy['intersection_score'] = df_copy['intersection'].map(intersection_weights).fillna(1)
    df_copy['road_cat_score'] = df_copy['road_category'].map(road_category_weights).fillna(1)
    df_copy['traffic_score'] = df_copy['traffic_regime'].map(traffic_regime_weights).fillna(1)

    # Sum the individual scores
    df_copy['road_complexity_index'] = (
            df_copy['lanes_score'] +
            df_copy['intersection_score'] +
            df_copy['road_cat_score'] +
            df_copy['traffic_score']
    )

    # Scale the index (0 to 10)
    scaler = MinMaxScaler(feature_range=(0, 10))
    df_copy['road_complexity_index'] = scaler.fit_transform(df_copy[['road_complexity_index']])

    # Remove temporary score columns
    df_copy.drop(columns=['lanes_score', 'intersection_score', 'road_cat_score', 'traffic_score'], inplace=True, errors='ignore')

    return df_copy

def create_surface_quality_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary 'surface_quality_indicator'.
    1 (good) = surface is normal AND profile is flat.
    0 (bad) = otherwise.
    """
    print("  Creating surface quality indicator...")
    df_copy = df.copy()

    is_pavement_good = (df_copy['pavement_condition'] == 1)
    is_profile_good = (df_copy['longitudinal_profile'] == 1)

    df_copy['surface_quality_indicator'] = (is_pavement_good & is_profile_good).astype(int)

    return df_copy

def create_user_role(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the new categorical 'role' feature from 'user_category'.
    - 'driver': (From user category = 1)
    - 'passenger': (From user category = 2)
    - 'pedestrian': (From user category = 3)
    - 'other': (For any other or null value)
    """
    print("  Creating user 'role' feature...")
    df_copy = df_merged.copy()

    if 'user_category' in df_copy.columns:
        role_map = {
            1: 'driver',
            2: 'passenger',
            3: 'pedestrian'
        }
        # .fillna('other') maps all other values (like NaN, -1) to 'other'
        df_copy['role'] = df_copy['user_category'].map(role_map).fillna('other')
    else:
        print("    ! 'user_category' column not found. Skipping 'role' creation.")
        # Create a default column if source is missing so imputation doesn't fail
        df_copy['role'] = 'other'

    return df_copy

def create_ordinal_conditions(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Remaps categorical weather and lighting conditions to a
    meaningful ordinal scale representing risk/visibility.
    """
    print("  Creating ordinal weather and lighting features...")
    df_copy = df_merged.copy()

    # --- Lighting Condition ---
    # 0 (Good) -> 1: Full day
    # 1 (Medium) -> 5: Night with public lighting on
    # 2 (Poor) -> 2: Twilight or dawn
    # 3 (Very Poor) -> 3: Night without public lighting, 4: Night with public lighting not on
    # Default: -1, others -> 0 (Good)
    if 'lighting_condition' in df_copy.columns:
        lighting_map = {
            1: 0,  # Full day
            5: 1,  # Night w/ light on
            2: 2,  # Twilight/dawn
            3: 3,  # Night w/o light
            4: 3  # Night w/ light off
        }
        # .fillna(0) maps -1 (Not specified) and others to 0 (Good)
        df_copy['lighting_ordinal'] = df_copy['lighting_condition'].map(lighting_map).fillna(0).astype(int)

    # --- Weather Condition ---
    # 0 (Good) -> 1: Normal
    # 1 (Okay) -> 8: Overcast
    # 2 (Slight Risk) -> 2: Light rain, 7: Dazzling weather
    # 3 (Medium Risk) -> 6: Strong wind/storm, 3: Heavy rain
    # 4 (High Risk) -> 5: Fog/smoke, 4: Snow/hail
    # Default: -1, 9, others -> 0 (Good)
    if 'weather_condition' in df_copy.columns:
        weather_map = {
            1: 0,  # Normal
            8: 1,  # Overcast
            2: 2,  # Light rain
            7: 2,  # Dazzling
            6: 3,  # Wind/storm
            3: 3,  # Heavy rain
            5: 4,  # Fog/smoke
            4: 4  # Snow/hail
        }
        # .fillna(0) maps -1, 9 (Other), and others to 0 (Good)
        df_copy['weather_ordinal'] = df_copy['weather_condition'].map(weather_map).fillna(0).astype(int)

    return df_copy

def create_ordinal_target(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the new ordinal target variable 'injury_target' based on
    the 3-class system requested:
    - 0: Uninjured (Original: 1)
    - 1: Lightly Injured (Original: 4)
    - 2: Severe (Hospitalized or Killed) (Original: 2, 3)
    """
    print("  Creating new ordinal target 'injury_target' (0, 1, 2)...")
    df_copy = df_merged.copy()

    # Define the mapping for the 3-class ordinal target
    ordinal_map = {
        1: 0,  # Uninjured
        4: 1,  # Lightly Injured
        3: 2,  # Hospitalized
        2: 2  # Killed
    }

    df_copy['injury_target'] = df_copy['injury_severity'].map(ordinal_map)

    # Note: The original 'injury_severity' column will be
    # dropped in the d_feature_selection step.

    return df_copy

def create_cluster_feature(df_merged: pd.DataFrame) -> pd.DataFrame:
    df_copy = df_merged.copy()
    df_copy.drop(['injury_target'], axis=1, inplace=True)
    num_cols = ['latitude', 'longitude', 'speed_limit', 'age', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]
    ord_cols = ['lighting_ordinal', 'weather_ordinal', 'road_complexity_index', 'impact_score', 'impact_score_other', 'impact_delta']
    cat_cols = []
    
    for col in df_copy.columns:
        if col not in num_cols and col not in ord_cols:
            cat_cols.append(col)
            df_copy[col] = df_copy[col].astype(str)
    cat_indices = [df_copy.columns.get_loc(col) for col in cat_cols]
    df_np = df_copy.to_numpy()
    
    kproto = KPrototypes(
        n_clusters=3,
        init='Huang',
        n_init=5,
        verbose=1,
        random_state=42
    )

    clusters = kproto.fit_predict(df_np, categorical=cat_indices)
    df_copy['cluster'] = clusters

    return df_copy