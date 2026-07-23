"""
dataset.py

Handles loading the raw CICIDS2017 CSV file(s) into a Pandas DataFrame.

Why this file exists:
    Dataset loading logic should live in exactly one place. Every other
    module (EDA, preprocessing, training) will import `load_raw_dataset`
    from here rather than writing its own pd.read_csv() calls. This
    guarantees consistent behavior (e.g., column name cleaning) everywhere
    the data is touched.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up column names in the raw CICIDS2017 CSV.

    Why this is needed:
        The original CICIDS2017 CSVs have column names with inconsistent
        leading/trailing whitespace (e.g., " Destination Port" with a
        leading space). This causes silent bugs later — for example,
        df["Label"] would raise a KeyError if the real column is
        " Label" with a leading space, which is EXTREMELY hard to spot
        just by looking at printed output.

    Args:
        df: The raw DataFrame with potentially messy column names.

    Returns:
        The same DataFrame with stripped, cleaned column names.
    """
    df.columns = df.columns.str.strip()
    return df


def load_raw_dataset(filename: str) -> pd.DataFrame:
    """
    Loads a single raw CICIDS2017 CSV file from data/raw/ into a DataFrame.

    Args:
        filename: Name of the CSV file inside data/raw/
                  (e.g., "Wednesday-workingHours.pcap_ISCX.csv").

    Returns:
        A cleaned Pandas DataFrame containing the raw traffic data.

    Raises:
        FileNotFoundError: If the specified file doesn't exist in data/raw/.
    """
    file_path: Path = RAW_DATA_DIR / filename

    if not file_path.exists():
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(
            f"Could not find '{filename}' in {RAW_DATA_DIR}. "
            f"Make sure you downloaded it and placed it in data/raw/."
        )

    logger.info(f"Loading dataset from: {file_path}")

    # low_memory=False: CICIDS2017 CSVs mix types in some columns (e.g.
    # numeric columns that occasionally contain 'Infinity' as a string).
    # Without this, Pandas may guess inconsistent dtypes per chunk and
    # emit a DtypeWarning.
    df = pd.read_csv(file_path, low_memory=False)

    df = clean_column_names(df)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Logs a quick sanity-check summary of the dataset: shape, columns,
    dtypes, missing values, and label distribution (if a Label column
    exists).

    Why this matters:
        Before we do ANY preprocessing or modeling, we need to know what
        we're actually working with. This is the first line of defense
        against silently training a model on broken data.

    Args:
        df: The DataFrame to summarize.
    """
    logger.info(f"Number of rows: {df.shape[0]}")
    logger.info(f"Number of columns: {df.shape[1]}")

    # Count of missing (NaN) values per column, but only show columns
    # that actually HAVE missing values (keeps the log readable).
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        logger.info("No missing values found in any column.")
    else:
        logger.warning(f"Missing values found in {len(missing_counts)} column(s):")
        for column_name, count in missing_counts.items():
            logger.warning(f"  {column_name}: {count} missing")

    # CICIDS2017 has a 'Label' column identifying BENIGN vs attack type.
    if "Label" in df.columns:
        label_counts = df["Label"].value_counts()
        total_rows = df.shape[0]
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / total_rows) * 100
            logger.info(f"  {label}: {count} ({percentage:.2f}%)")
    else:
        logger.warning("No 'Label' column found in dataset.")


if __name__ == "__main__":
    # Quick manual test: run `python -m src.dataset` directly.
    DATASET_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

    dataframe = load_raw_dataset(DATASET_FILENAME)
    summarize_dataset(dataframe)

    # Show the first few rows and column