import pandas as pd
import numpy as np
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)


def compare_xlsx(file1, file2, key_column='predicted_class', tolerance=1e-5):
    """
    Compare two XLSX files and print out the differences with highlighted key column differences.

    Parameters:
    - file1: Path to the first XLSX file.
    - file2: Path to the second XLSX file.
    - key_column: Column name to always compare and report differences.
    - tolerance: Tolerance level for floating-point comparisons in non-key columns.
    """
    # Load the files into pandas DataFrames
    try:
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
    except Exception as e:
        print(f"{Fore.RED}Error reading files: {e}{Style.RESET_ALL}")
        return

    # Check if columns match
    if not df1.columns.equals(df2.columns):
        print(f"{Fore.YELLOW}Columns do not match.{Style.RESET_ALL}")
        print("Columns in first file:", df1.columns.tolist())
        print("Columns in second file:", df2.columns.tolist())
        # Optionally, you can decide to stop the comparison here
        # return
    else:
        print(f"{Fore.GREEN}Columns match.{Style.RESET_ALL}")

    # Ensure both DataFrames have the same number of rows
    if len(df1) != len(df2):
        print(
            f"{Fore.YELLOW}Number of rows differ. First file has {len(df1)} rows; second file has {len(df2)} rows.{Style.RESET_ALL}")
        # Proceed to compare up to the minimum number of rows
        min_rows = min(len(df1), len(df2))
        df1 = df1.iloc[:min_rows].reset_index(drop=True)
        df2 = df2.iloc[:min_rows].reset_index(drop=True)
    else:
        min_rows = len(df1)

    # Initialize a list to collect differences
    differences = []

    # Iterate through each column to find differences
    for column in df1.columns:
        if column == key_column:
            # Always compare the key_column and report any differences
            diff_series = df1[column] != df2[column]
        else:
            # For numeric columns, use the specified tolerance
            if pd.api.types.is_numeric_dtype(df1[column]):
                diff_series = (df1[column] - df2[column]).abs() > tolerance
            else:
                # For non-numeric columns, check for exact differences
                diff_series = df1[column] != df2[column]

        # Handle NaN comparisons: if both are NaN, consider them equal
        both_nan = df1[column].isna() & df2[column].isna()
        diff_series = diff_series & ~both_nan

        # Get indices where differences occur
        differing_indices = diff_series[diff_series].index

        for idx in differing_indices:
            val1 = df1.at[idx, column]
            val2 = df2.at[idx, column]
            difference_type = 'Key Column' if column == key_column else 'Other Column'
            differences.append({
                'Row': idx + 1,  # Assuming 1-based indexing for rows
                'Column': column,
                'First File': val1,
                'Second File': val2,
                'Difference Type': difference_type
            })

    # Display the differences
    if not differences:
        print(f"{Fore.GREEN}The files are identical within the specified criteria.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}The files differ in {len(differences)} places:{Style.RESET_ALL}")
        # Convert differences to a DataFrame for better readability
        diff_df = pd.DataFrame(differences)
        # Optionally, sort the differences by difference type, then by row and column
        diff_df['Difference Type Order'] = diff_df['Difference Type'].apply(lambda x: 0 if x == 'Key Column' else 1)
        diff_df = diff_df.sort_values(by=['Difference Type Order', 'Row', 'Column'])
        diff_df.drop('Difference Type Order', axis=1, inplace=True)
        # Reset index for clean display
        diff_df.reset_index(drop=True, inplace=True)

        # Function to apply coloring based on Difference Type
        def highlight_row(row):
            if row['Difference Type'] == 'Key Column':
                return [
                    Fore.RED + str(row['Row']) + Style.RESET_ALL,
                    Fore.RED + str(row['Column']) + Style.RESET_ALL,
                    Fore.RED + str(row['First File']) + Style.RESET_ALL,
                    Fore.RED + str(row['Second File']) + Style.RESET_ALL,
                    Fore.RED + str(row['Difference Type']) + Style.RESET_ALL
                ]
            else:
                return [
                    str(row['Row']),
                    str(row['Column']),
                    str(row['First File']),
                    str(row['Second File']),
                    str(row['Difference Type'])
                ]

        # Apply highlighting
        highlighted_diff = diff_df.apply(highlight_row, axis=1)

        # Convert to string and print
        for row in highlighted_diff:
            print(' | '.join(row))

        # Optionally, save the differences to an Excel or CSV file with the Difference Type
        # Uncomment the following lines if you wish to save the output
        # diff_df.to_excel('differences.xlsx', index=False)
        # print(f"{Fore.BLUE}Differences have been saved to 'differences.xlsx'.{Style.RESET_ALL}")


# Example usage:
# Replace 'prediction.xlsx' and 'prediction_new_test.xlsx' with your actual file paths
# and ensure 'predicted_class' is the exact column name you want to monitor.
compare_xlsx('prediction_v1.xlsx', 'WueVision_predicted_test_dataset.xlsx', key_column='predicted_class', tolerance=1e-5)
