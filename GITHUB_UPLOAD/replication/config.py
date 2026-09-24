"""Configuration for the replication package.

All paths are resolved relative to this file, so the package can live
anywhere (including paths with non-ASCII characters) and be run from any
working directory.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")

# Raw inputs. We look for each file as plain .csv first, then gzipped .csv.gz,
# in the project root and then in ./data. The repository ships the gzipped
# copies (14 MB and 7 MB) because GitHub rejects files above 100 MB; pandas
# reads .gz transparently, so nothing else in the package changes.
# Override the search root with the CLIMATE_DATA_DIR environment variable.
RAW_DIR = os.environ.get("CLIMATE_DATA_DIR", PROJECT)


def _find(stem):
    for folder in (RAW_DIR, os.path.join(RAW_DIR, "data"), os.path.join(PROJECT, "data")):
        for ext in (".csv", ".csv.gz"):
            p = os.path.join(folder, stem + ext)
            if os.path.exists(p):
                return p
    return os.path.join(RAW_DIR, stem + ".csv")   # reported as missing below


MONTHLY_CSV = _find("climate_food_monthly_estimation")
ANNUAL_CSV = _find("climate_macro_annual_with_exposure")
PANEL_PKL = os.path.join(RESULTS, "analysis_panel.pkl")

# ---------------------------------------------------------------- estimation
P_LAG = 12                 # lags of shock and outcome in every local projection
P_LAG_MAX = 24             # deepest lag stored; pre-trend tests shift the control window
HORIZONS = list(range(-12, 25))
CROPS = ("wheat", "rice")  # maize is excluded from the study by design

# Chronic-hyperinflation economies excluded from the estimation sample.
HYPER = ["VEN", "ZWE", "LBN", "ARG", "SDN", "SSD", "SYR", "YEM", "IRN", "TUR"]
YOY_CAP = 40.0             # drop country-months with |headline yoy| >= 40%
WINSOR = 0.005             # two-sided winsorisation of monthly log changes

# --------------------------------------------------------------- run control
# FAST=1 shrinks the two Monte-Carlo style exercises so the whole package
# finishes in a few minutes. FAST=0 reproduces the numbers in the paper.
FAST = os.environ.get("REPFAST", "0") == "1"
N_PERM = 200 if FAST else 1000       # randomisation-inference draws
RI_TIME_BUDGET = 120 if FAST else 900   # seconds
DML_FOLDS = 5
SEED = 7


def ascii_stdout():
    """Make printing safe on Windows consoles using cp1252.

    Every table in this package is ASCII-only, but we still force UTF-8 when
    the interpreter allows it and fall back silently when it does not.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)


def check_inputs():
    missing = [p for p in (MONTHLY_CSV, ANNUAL_CSV) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Could not find the raw data file(s):\n  "
            + "\n  ".join(missing)
            + "\n\nPut the two CSVs next to the 'replication' folder, or set the\n"
              "CLIMATE_DATA_DIR environment variable to the folder holding them."
        )
