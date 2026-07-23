"""
eda.py

Exploratory Data Analysis for the CICIDS2017 dataset.

Why this file exists:
    Before we preprocess or train anything, we need a clear diagnostic
    picture of the data: are there infinite values? How imbalanced are
    the classes? Are any features broken or redundant? This module
    answers those questions and saves visualizations to reports/figures/
    for later use in documentation/presentation.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend: saves plots to file
                        # instead of trying to pop up a GUI window, which
                        # can fail/hang when run from a plain script.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.dataset import load_raw_dataset, clean_column_names
from src.logger import get_logger

logger = get_logger(__name__)

FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"


def check_infinite_values(df: pd.DataFrame) -> pd.Series:
    """
    Identifies columns containing infinite (np.inf or -np.inf) values.

    Why this is separate from checking NaNs:
        df.isnull() does NOT detect np.inf. Pandas treats infinity as a
        valid float value, not a missing one. If we only checked for NaN
        (as we did in Milestone 3), infinite values would silently pass
        through untouched into preprocessing and training, where they
        can cause NaN losses or numerical instability in the model.

    Args:
        df: The DataFrame to check. Only numeric columns are considered.

    Returns:
        A Series indexed by column name, containing the count of
        infinite values in each column that has at least one.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # np.isinf() returns a boolean DataFrame the same shape as numeric_df.
    inf_counts = np.isinf(numeric_df).sum()
    inf_counts = inf_counts[inf_counts > 0]

    if inf_counts.empty:
        logger.info("No infinite values found in any numeric column.")
    else:
        logger.warning(f"Infinite values found in {len(inf_counts)} column(s):")
        for column_name, count in inf_counts.items():
            logger.warning(f"  {column_name}: {count} infinite values")

    return inf_counts


def plot_label_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plots and saves a bar chart of label (class) distribution on a
    log-scaled y-axis.

    Why log scale:
        BENIGN (~440K) and Heartbleed (11) differ by 4 orders of
        magnitude. On a normal linear scale, small classes would render
        as invisible flat lines at the bottom of the chart. A log scale
        makes every class visibly comparable on the same plot.

    Args:
        df: The DataFrame containing a 'Label' column.
        save_path: File path (including filename) to save the PNG to.
    """
    label_counts = df["Label"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    label_counts.plot(kind="bar", ax=ax, color="steelblue")

    ax.set_yscale("log")
    ax.set_title("Label Distribution (log scale)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count (log scale)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # Free memory; important when generating many plots.

    logger.info(f"Label distribution plot saved to: {save_path}")


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes basic descriptive statistics (min, max, mean, std) for all
    numeric columns, used to spot broken/suspicious features.

    Why this matters:
        A column where min == max (zero variance) contributes nothing
        to the model and can be safely dropped. A column with an
        absurdly large max (e.g., 1e15) might indicate a data quality
        issue worth investigating before training.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame of descriptive statistics, one row per numeric
        column.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().transpose()

    # Flag zero-variance columns (min == max means every value is identical).
    zero_variance_cols = stats[stats["min"] == stats["max"]].index.tolist()

    if zero_variance_cols:
        logger.warning(
            f"Zero-variance columns found ({len(zero_variance_cols)}): "
            f"{zero_variance_cols}"
        )
    else:
        logger.info("No zero-variance columns found.")

    return stats


def plot_correlation_heatmap(
    df: pd.DataFrame, save_path: Path, correlation_threshold: float = 0.95
) -> list[tuple[str, str, float]]:
    """
    Computes a correlation matrix for numeric features, saves a heatmap,
    and identifies pairs of highly correlated features.

    Why we identify pairs explicitly (not just plot the heatmap):
        A 78x78 heatmap is visually dense and hard to read precisely.
        Explicitly listing pairs above the threshold gives us an
        actionable list to review during Milestone 5 (Preprocessing),
        where we'll decide whether to drop redundant features.

    Args:
        df: The DataFrame to analyze.
        save_path: File path to save the heatmap PNG to.
        correlation_threshold: Absolute correlation value above which a
            pair is considered "highly correlated" (default 0.95).

    Returns:
        A list of (feature_a, feature_b, correlation_value) tuples for
        every pair exceeding the threshold.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Replace inf with NaN temporarily just for this correlation
    # calculation, so infinite values (which we already flagged
    # separately) don't distort the correlation math.
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr_matrix.columns, fontsize=6)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"Correlation heatmap saved to: {save_path}")

    # Find highly correlated pairs (excluding the diagonal, which is
    # always 1.0 since every feature perfectly correlates with itself).
    high_corr_pairs: list[tuple[str, str, float]] = []
    columns = corr_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):  # j > i avoids duplicate pairs and the diagonal
            corr_value = corr_matrix.iloc[i, j]
            if pd.notna(corr_value) and abs(corr_value) >= correlation_threshold:
                high_corr_pairs.append((columns[i], columns[j], corr_value))

    if high_corr_pairs:
        logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pair(s):")
        for feature_a, feature_b, corr_value in high_corr_pairs:
            logger.warning(f"  {feature_a} <-> {feature_b}: {corr_value:.3f}")
    else:
        logger.info(f"No feature pairs exceeded correlation threshold of {correlation_threshold}.")

    return high_corr_pairs


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    DATASET_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

    df = load_raw_dataset(DATASET_FILENAME)

    logger.info("=== Checking for infinite values ===")
    check_infinite_values(df)

    logger.info("=== Plotting label distribution ===")
    plot_label_distribution(df, FIGURES_DIR / "label_distribution.png")

    logger.info("=== Computing feature statistics ===")
    stats = get_feature_statistics(df)
    print("\n--- Feature statistics (first 10 rows) ---")
    print(stats.head(10))

    logger.info("=== Computing correlation heatmap ===")
    high_corr_pairs = plot_correlation_heatmap(df, FIGURES_DIR / "correlation_heatmap.png")

    logger.info("EDA complete.")



def check_negative_values(df: pd.DataFrame) -> pd.Series:
    """
    Identifies columns containing infinite (np.inf or -np.inf) values.

    Why this is separate from checking NaNs:
        df.isnull() does NOT detect np.inf. Pandas treats infinity as a
        valid float value, not a missing one. If we only checked for NaN
        (as we did in Milestone 3), infinite values would silently pass
        through untouched into preprocessing and training, where they
        can cause NaN losses or numerical instability in the model.

    Args:
        df: The DataFrame to check. Only numeric columns are considered.

    Returns:
        A Series indexed by column name, containing the count of
        infinite values in each column that has at least one.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # np.isinf() returns a boolean DataFrame the same shape as numeric_df.
    inf_counts = np.isinf(numeric_df<0).sum()
    inf_counts = inf_counts[inf_counts > 0]

    if inf_counts.empty:
        logger.info("No negative values found in any numeric column.")
    else:
        logger.warning(f"Negative values found in {len(inf_counts)} column(s):")
        for column_name, count in inf_counts.items():
            logger.warning(f"  {column_name}: {count} negative values")

    return inf_counts
