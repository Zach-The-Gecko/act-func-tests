import pandas as pd
import sys


def column_letter_to_index(letter):
    """Convert Excel-style column letter to 0-based index.

    Examples:
        A -> 0
        B -> 1
        Z -> 25
        AA -> 26
        AZ -> 51
    """
    index = 0
    for char in letter.upper():
        index = index * 26 + (ord(char) - ord('A') + 1)
    return index - 1


def strip_csv(input_file, output_file, columns):
    """Keep only specified columns from a CSV file.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        columns: List of column letters to keep (e.g., ["A", "B", "AZ"])
    """
    # Read the CSV file without headers first to get all columns
    df = pd.read_csv(input_file, header=None)

    # Convert column letters to indices
    column_indices = [column_letter_to_index(col) for col in columns]

    # Validate indices
    max_index = max(column_indices)
    if max_index >= len(df.columns):
        raise ValueError(f"Column index {max_index} (letter: {columns[column_indices.index(max_index)]}) "
                         f"exceeds available columns (max index: {len(df.columns) - 1})")

    # Select only the specified columns
    df_filtered = df.iloc[:, column_indices]

    # Save to output file
    df_filtered.to_csv(output_file, index=False, header=False)
    print(
        f"Saved filtered CSV with {len(column_indices)} columns to: {output_file}")


if __name__ == "__main__":
    # Hardcoded columns to keep
    columns = ["A", "B", "AZ"]

    # Get input and output file paths from command line or use defaults
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(
            sys.argv) >= 3 else input_file.replace(".csv", "_filtered.csv")
    else:
        print("Usage: python strip_csv.py <input_file.csv> [output_file.csv]")
        print(f"Will keep columns: {columns}")
        sys.exit(1)

    strip_csv(input_file, output_file, columns)
