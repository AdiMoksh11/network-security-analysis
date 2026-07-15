"""
config.py

Centralized configuration for the Network Traffic Attack Detection project.

Why this file exists:
    Instead of hardcoding file paths, random seeds, or hyperparameters
    inside every individual script, we define them ONCE here. Every other
    module imports from this file. This means if we ever move a folder or
    change a setting, we update it in exactly one place.

    We use pathlib.Path instead of raw strings for file paths because:
    - It works identically on Windows, Mac, and Linux (no manual '\\' vs '/' handling).
    - It provides convenient methods like .exists(), .mkdir(), etc.
"""

from pathlib import Path

# -----------------------------------------------------------------------
# BASE DIRECTORY
# -----------------------------------------------------------------------
# Path(__file__) = this file's location (src/config.py)
# .resolve()     = converts it to an absolute path
# .parent        = the folder containing this file        -> src/
# .parent again  = the folder containing that folder       -> project root
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------
# DATA DIRECTORIES
# -----------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# -----------------------------------------------------------------------
# MODEL & LOGGING DIRECTORIES
# -----------------------------------------------------------------------
MODELS_DIR: Path = BASE_DIR / "models"
LOGS_DIR: Path = BASE_DIR / "logs"

# -----------------------------------------------------------------------
# REPRODUCIBILITY
# -----------------------------------------------------------------------
# A fixed random seed ensures that any random operations (train/test split,
# model weight initialization, shuffling, etc.) produce the SAME result
# every time we run the code. This is critical for debugging and for
# comparing experiments fairly.
RANDOM_SEED: int = 42

# -----------------------------------------------------------------------
# ENSURE DIRECTORIES EXIST
# -----------------------------------------------------------------------
def ensure_directories_exist() -> None:
    """
    Creates all required project directories if they don't already exist.

    Why a function instead of just running this at import time?
        Explicit is better than implicit. We call this function
        deliberately (e.g., at the start of main scripts) rather than
        having side effects happen silently just from importing config.
    """
    directories: list[Path] = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        # parents=True -> creates intermediate folders if needed
        # exist_ok=True -> don't raise an error if it already exists
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # This block only runs if you execute `python src/config.py` directly,
    # NOT when this file is imported by other modules. Useful for quick
    # manual testing.
    ensure_directories_exist()
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"LOGS_DIR: {LOGS_DIR}")
    print("All directories verified/created successfully.")