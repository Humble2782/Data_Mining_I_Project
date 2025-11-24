import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.testing as pdt
from pathlib import Path

def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes a DataFrame and returns a summary of missing values for each column.

    The summary includes the absolute count of missing values and the
    percentage of missing values, sorted from highest to lowest percentage.

    Args:
        df: The pandas DataFrame to analyze.

    Returns:
        A pandas DataFrame indexed by column name, with 'missing_count'
        and 'missing_percentage' columns.
    """
    print("Analyzing missing values...")

    # Calculate the absolute count of missing values
    missing_count = df.isnull().sum()

    # Calculate the total number of rows
    total_rows = len(df)

    # Calculate the percentage of missing values
    missing_percentage = (missing_count / total_rows) * 100

    # Create a new DataFrame to hold the summary
    missing_summary_df = pd.DataFrame({
        'missing_count': missing_count,
        'missing_percentage': missing_percentage
    })

    # Filter out columns that have no missing values to keep the summary clean
    missing_summary_df = missing_summary_df[missing_summary_df['missing_count'] > 0]

    # Sort the DataFrame by the percentage of missing values (descending)
    missing_summary_df = missing_summary_df.sort_values(by='missing_percentage', ascending=False)

    if missing_summary_df.empty:
        print("...No missing values found.")
    else:
        print(f"...Found missing values in {len(missing_summary_df)} columns.")

    return missing_summary_df

def compare_file_schemas(input_path: str, file_patterns: list, sep=';', decimal=',') -> pd.DataFrame:
    """
    Compares the column schemas of multiple CSV files and returns a comparison DataFrame.

    This is useful for diagnosing integration problems, e.g., why 2019 data fails
    when 2022 data works.

    Args:
        input_path: The directory path containing the original data.
        file_patterns: A list of filenames to compare (e.g., ['users-2019.csv', 'users-2022.csv']).
        sep: The separator used in the CSV files.
        decimal: The decimal separator used in the CSV files.

    Returns:
        A pandas DataFrame where rows are all unique column names found,
        columns are the file patterns, and values are True/False if that
        column exists in that file.
    """
    print(f"Comparing schemas for files: {file_patterns} in path {input_path}")

    all_cols_sets = {}
    all_cols_global = set()

    for file in file_patterns:
        file_path = f"{input_path}/{file}"
        try:
            # Read just the header to be fast
            df_header = pd.read_csv(
                file_path,
                sep=sep,
                decimal=decimal,
                nrows=0,
                low_memory=False
            )

            col_set = set(df_header.columns)
            all_cols_sets[file] = col_set
            all_cols_global.update(col_set)
            print(f"  ...Successfully read header for {file}")

        except FileNotFoundError:
            print(f"  ...ERROR: File not found: {file_path}")
            all_cols_sets[file] = set() # Add empty set
        except Exception as e:
            print(f"  ...ERROR: Could not read {file_path}. Error: {e}")
            all_cols_sets[file] = set()

    # Create the comparison DataFrame
    comparison_df = pd.DataFrame(
        index=sorted(list(all_cols_global)),
        columns=file_patterns,
        dtype=bool
    )

    # Populate the DataFrame
    for file in file_patterns:
        for col in comparison_df.index:
            comparison_df.loc[col, file] = col in all_cols_sets[file]

    # Add a column to highlight differences
    comparison_df['is_different'] = comparison_df.apply(lambda row: len(row.unique()) > 1, axis=1)

    return comparison_df.sort_values(by='is_different', ascending=False)


def get_descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a comprehensive descriptive summary for ALL columns in a DataFrame.

    This helps identify issues like:
    - Constant columns (unique_count == 1)
    - Highly skewed columns (frequency_percentage > 95)
    - Data type issues (dtype)

    Args:
        df: The pandas DataFrame to analyze.

    Returns:
        A pandas DataFrame where each row corresponds to a column from the
        input DataFrame, providing a full statistical summary.
    """
    print("Generating comprehensive descriptive summary...")
    summary_list = []
    total_rows = len(df)

    for col_name in df.columns:
        col = df[col_name]

        # Universal Stats
        dtype = col.dtype
        missing_count = col.isnull().sum()
        missing_percentage = (missing_count / total_rows) * 100
        unique_count = col.nunique()

        # Value counts stats (handle all-NaN columns)
        value_counts_norm = col.value_counts(normalize=True, dropna=False)

        if value_counts_norm.empty:
            most_frequent_value = None
            frequency_percentage = 100.0 if missing_count == total_rows else 0.0
        else:
            # Handle potential NaN index (if NaN is most frequent)
            most_frequent_value = value_counts_norm.index[0]
            if pd.isna(most_frequent_value):
                most_frequent_value = "NaN"  # Use string representation
            frequency_percentage = value_counts_norm.iloc[0] * 100

        col_summary = {
            'dtype': dtype,
            'missing_percentage': missing_percentage,
            'unique_count': unique_count,
            'most_frequent_value': most_frequent_value,
            'frequency_percentage': frequency_percentage
        }

        # Add numeric-specific stats
        if pd.api.types.is_numeric_dtype(col):
            desc = col.describe()
            col_summary.update({
                'mean': desc.get('mean'),
                'std': desc.get('std'),
                'min': desc.get('min'),
                '25%': desc.get('25%'),
                '50%': desc.get('50%'),
                '75%': desc.get('75%'),
                'max': desc.get('max')
            })
        else:
            # Fill non-numeric with NaNs for consistent columns
            col_summary.update({
                'mean': None, 'std': None, 'min': None, '25%': None,
                '50%': None, '75%': None, 'max': None
            })

        summary_list.append(col_summary)

    # Create the final DataFrame
    summary_df = pd.DataFrame(summary_list, index=df.columns)

    # Define a logical column order
    column_order = [
        'dtype', 'missing_percentage', 'unique_count',
        'most_frequent_value', 'frequency_percentage',
        'mean', 'std', 'min', '25%', '50%', '75%', 'max'
    ]

    print("...Summary generation complete.")
    return summary_df[column_order]


def inspect_column_values(df: pd.DataFrame, column_name: str, top_n: int = 20):
    """
    Provides a deep-dive analysis of a single column, focusing on
    its data type and most frequent values.

    Args:
        df: The DataFrame to analyze.
        column_name: The name of the column to inspect.
        top_n: How many of the most frequent values to display.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the DataFrame.")
        return

    print(f"\n--- Inspecting Column: '{column_name}' ---")

    col = df[column_name]

    # 1. Data Type and Memory
    print("\n[Data Type and Info]")
    print(f"  dtype: {col.dtype}")
    print(f"  Non-Null Count: {col.count()} / {len(df)}")

    # 2. Value Counts
    print(f"\n[Top {top_n} Most Frequent Values (incl. NaNs)]")

    # Get value counts, including NaNs
    value_counts = col.value_counts(dropna=False).nlargest(top_n)

    # Create a small DataFrame for nice printing
    summary_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage': (value_counts / len(df)) * 100
    })

    print(summary_df)

    if len(value_counts) < col.nunique():
        print(f"  ...and {col.nunique() - len(value_counts)} other unique values.")
    print("---------------------------------")

def compare_dataframes(df_left: pd.DataFrame,
                       df_right: pd.DataFrame,
                       check_column_order: bool = False):
    """
    Compares two DataFrames using pandas.testing.assert_frame_equal.

    This provides a detailed error message if they are not identical.

    Args:
        df_left: The first DataFrame (e.g., "expected").
        df_right: The second DataFrame (e.g., "actual").
        check_column_order: If False (default), columns will be sorted
                            by name before comparison, so order doesn't matter.
    """
    print("--- Starting DataFrame Comparison ---")

    # If column order doesn't matter, sort them alphabetically.
    # This is equivalent to check_like=True in the assertion.
    if not check_column_order:
        df_left = df_left.sort_index(axis=1)
        df_right = df_right.sort_index(axis=1)

    try:
        pdt.assert_frame_equal(
            df_left,
            df_right,
            check_dtype=True,  # Ensure 'int64' is not 'float64'
            check_exact=True  # Ensure floats are identical
        )
        print("\n✅ SUCCESS: DataFrames are identical.")
        print("---------------------------------------")
        return True

    except AssertionError as e:
        print("\n❌ FAILURE: DataFrames are NOT identical.")
        print("---------------------------------------")
        print("\nDetailed Error:")
        print(e)
        print("---------------------------------------")
        return False


def print_dataframe_shapes(folder_path):
    """
    Iterates through a folder, reads .csv and .parquet files,
    and prints their shape (Rows, Columns).

    Improvements:
    - Uses 'sep=None' and 'engine=python' for CSVs to auto-detect delimiters (e.g. ; or ,).
    """
    path = Path(folder_path)

    if not path.exists():
        print(f"Error: Path '{folder_path}' does not exist.")
        return

    print(f"--- Inspecting DataFrames in: {path.resolve()} ---")
    print(f"{'File Name':<30} | {'Rows':<10} | {'Columns':<10}")
    print("-" * 60)

    files_found = False
    target_extensions = {'.csv', '.parquet'}

    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in target_extensions:
            files_found = True
            try:
                # Determine file type and load accordingly
                if file_path.suffix.lower() == '.csv':
                    # sep=None prompts pandas to sniff the delimiter automatically
                    df = pd.read_csv(file_path, sep=None, engine='python')
                elif file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)

                # Get shape
                rows, cols = df.shape
                print(f"{file_path.name:<30} | {rows:<10} | {cols:<10}")

            except Exception as e:
                print(f"{file_path.name:<30} | Error reading file: {e}")

    print("-" * 60)
    if not files_found:
        print("No .csv or .parquet files found.")