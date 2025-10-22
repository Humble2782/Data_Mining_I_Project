# feature_engineering.py
# Description: This file contains all functions for data transformation and feature engineering.

import pandas as pd


# --- User Data Transformations ---
def process_users(df_users, df_circumstances):
    """Processes user data to create new features like role, age, etc."""
    print("  Processing user data...")
    df_users_copy = df_users.copy()

    # Map user category to a readable role
    role_map = {1: 'driver', 2: 'passenger', 3: 'pedestrian'}
    df_users_copy['role'] = df_users_copy['user_category'].map(role_map).fillna('other')
    df_users_copy['is_pedestrian'] = (df_users_copy['role'] == 'pedestrian')

    # Create the 'age' feature
    df_users_copy = df_users_copy.merge(df_circumstances[['id_accident', 'year']], on='id_accident', how='left')
    df_users_copy['age'] = (df_users_copy['year'] - df_users_copy['year_of_birth']).astype('Int64')

    # Clean up temporary or redundant columns
    df_users_copy = df_users_copy.drop(columns=['year', 'number_vehicle'])

    return df_users_copy


# --- Vehicle Data Transformations ---
def process_vehicles(df_vehicles):
    """Processes vehicle data, simplifies categories, and calculates impact scores."""
    print("  Processing vehicle data...")
    df_vehicles_copy = df_vehicles.copy()

    def simplify_catv_6(x):
        if pd.isna(x): return pd.NA
        if x in {1, 80}: return "bicycle"
        if x in {2, 30, 41, 31, 32, 33, 34, 42, 43}: return "powered_2_3_wheeler"
        if x in {7, 10}: return "light_motor_vehicle"
        if x in {13, 14, 15, 16, 17}: return "hgv_truck"
        if x in {37, 38}: return "bus_coach"
        return "other"

    NA_CODES = {
        "vehicle_category": {-1}, "main_maneuver_before_accident": {-1},
        "initial_point_of_impact": {-1}, "fixed_obstacle_struck": {-1},
        "mobile_obstacle_struck": {-1}, "motor_type": {-1}, "direction_of_travel": {-1},
        "number_occupants_in_public_transport": {-1},
    }

    def apply_na_codes(df, na_codes=NA_CODES):
        df = df.copy()
        for col, codes in na_codes.items():
            if col in df.columns:
                df[col] = df[col].where(~df[col].isin(codes))
        return df

    CAT6_WEIGHT = {
        "rail_vehicle": 7, "hgv_truck": 6, "bus_coach": 5,
        "light_motor_vehicle": 4, "powered_2_3_wheeler": 3,
        "bicycle": 2, "other": 1, "unknown": 1,
    }

    V = apply_na_codes(df_vehicles_copy)
    V["vehicle_category"] = V["vehicle_category"].apply(simplify_catv_6)
    V["impact_score"] = V["vehicle_category"].map(CAT6_WEIGHT).fillna(1)
    V.sort_values(by=['id_accident', 'impact_score'], ascending=[True, False], inplace=True)

    one_entry = V.groupby('id_accident').filter(lambda x: len(x) == 1)
    for x in one_entry.columns:
        if x != 'id_accident':
            one_entry[f'{x}_other'] = pd.NA

    V_merged = (
        V.merge(V, on='id_accident', how='left', suffixes=['', '_other'])
        .query('id_vehicle != id_vehicle_other')
    )
    V_merged = V_merged.sort_values(by=['id_accident', 'id_vehicle', 'impact_score_other']).drop_duplicates(
        subset=['id_accident', 'id_vehicle'], keep='last')

    return pd.concat([one_entry, V_merged]).sort_index()


def create_vehicle_dummies(df_vehicles):
    """Creates dummy variables for vehicle categories involved in each accident."""
    print("  Creating vehicle category dummies...")
    g = df_vehicles[['id_accident', 'vehicle_category']].copy()
    tmp = pd.get_dummies(g, columns=['vehicle_category'], prefix='vehicle_category')
    return tmp.groupby('id_accident').sum()


# --- Location Data Transformations ---
def process_locations(df_locations):
    """Selects the most relevant location entry for each accident."""
    print("  Processing location data...")
    LOC_SCORE_COLS = [
        "road_category", "road_number", "road_number_index", "road_number_letter",
        "traffic_regime", "number_of_traffic_lanes", "reserved_lane_present",
        "longitudinal_profile", "nearest_reference_marker", "nearest_reference_marker_distance",
        "horizontal_alignment", "width_central_reservation", "carriageway_width",
        "pavement_condition", "infrastructure", "accident_situation", "speed_limit"
    ]
    LOC_WEIGHTS = {"road_category": 2.0, "speed_limit": 2.0}

    def _completeness_score(df: pd.DataFrame) -> pd.Series:
        present = ((df[LOC_SCORE_COLS].notna()) & (df[LOC_SCORE_COLS] != -1))
        for c, w in LOC_WEIGHTS.items():
            if c in present.columns:
                present[c] = present[c] * w
        return present.sum(axis=1)

    def _major_road_rank(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        return s.fillna(99)

    L = df_locations.copy()
    L["__score"] = _completeness_score(L)
    L["__has_vma"] = L["speed_limit"].notna().astype(int)
    L["__has_pr"] = L["nearest_reference_marker"].notna().astype(int) | L[
        "nearest_reference_marker_distance"].notna().astype(int)
    L["__road_rank"] = -_major_road_rank(L["road_category"])

    L = (L
         .sort_values(["id_accident", "__score", "__has_vma", "__has_pr", "__road_rank"])
         .drop_duplicates("id_accident", keep="last")
         .drop(columns=["__score", "__has_vma", "__has_pr", "__road_rank"])
         )
    return L


# --- Merged Data Transformations ---
def process_user_vehicle_merge(df_users, df_vehicles):
    """Merges user and vehicle data and handles special cases like pedestrians."""
    print("  Merging user and vehicle data...")
    df_user_vehicle = df_users.merge(df_vehicles, on=['id_accident', 'id_vehicle'], how='left')

    vehicle_features = ['id_vehicle', 'number_vehicle', 'direction_of_travel', 'vehicle_category',
                        'fixed_obstacle_struck', 'mobile_obstacle_struck', 'initial_point_of_impact',
                        'main_maneuver_before_accident', 'motor_type',
                        'number_occupants_in_public_transport', 'impact_score']
    vehicle_other_features = [f'{x}_other' for x in vehicle_features]

    mask = df_user_vehicle['is_pedestrian']
    # For pedestrians, the 'other' vehicle is the one that struck them
    df_user_vehicle.loc[mask, vehicle_other_features] = df_user_vehicle.loc[mask, vehicle_features].to_numpy()
    # Clear the original vehicle columns for pedestrians
    df_user_vehicle.loc[mask, vehicle_features] = pd.NA

    return df_user_vehicle