import pandas as pd
import EDA_helper as helper
from pathlib import Path

# --- Configuration ---
# Define the paths to the data you want to compare
PATH_BEFORE = '../../data/C_merged'  # Data *before* feature engineering
PATH_AFTER = '../../data/F_feature_selection'  # Data *after* all processing
YEAR_TO_CHECK = '2022'  # Which year file to load


# ---------------------

def load_data(path: str, year: str) -> pd.DataFrame:
    """Helper to load a CSV file."""
    file_path_glob = list(Path(path).glob(f'*{year}.csv'))
    if not file_path_glob:
        print(f"Error: No '*{year}.csv' file found in '{path}'.")
        print("Please run the preprocessing pipeline first or check the path.")
        return None

    try:
        df = pd.read_csv(file_path_glob[0], sep=';', decimal=',', low_memory=False)
        print(f"\nSuccessfully loaded {file_path_glob[0]}")
        return df
    except Exception as e:
        print(f"Error loading {file_path_glob[0]}: {e}")
        return None


def run_inspection():
    """
    Runs the 'before' and 'after' analysis to debug feature engineering.
    """

    # --- 1. "BEFORE" ANALYSIS ---
    # This checks the raw input to your function
    print("--- (1/2) 'BEFORE' ANALYSIS (Data from C_merged) ---")
    df_before = load_data(PATH_BEFORE, YEAR_TO_CHECK)

    if df_before is not None:
        # This is the key inspection. What is in 'vehicle_category'
        # *before* your create_vehicle_features function runs?
        helper.inspect_column_values(df_before, 'vehicle_category')

    # --- 2. "AFTER" ANALYSIS ---
    # This confirms the problem you are seeing
    print("\n--- (2/2) 'AFTER' ANALYSIS (Data from F_feature_selection) ---")
    df_after = load_data(PATH_AFTER, YEAR_TO_CHECK)

    if df_after is not None:
        # This should show 'other' or 'unknown' as 100%
        helper.inspect_column_values(df_after, 'vehicle_category_simplified')

        # This should show '1' as 100%
        helper.inspect_column_values(df_after, 'impact_score')

        # We can add the 'role' check here, where it *should* exist
        helper.inspect_column_values(df_after, 'role')


if __name__ == "__main__":
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 100)
    run_inspection()

