"""
utils.py
--------
Shared helper functions for the Museum Game Interaction Dataset analysis.

Used by all analysis scripts (01_data_screening_eda.py, 02_hypothesis_testing.py,
03_diagnostic_analysis.py) to keep data loading, ordinal encoding, and
plotting configuration consistent across the pipeline.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allows script to run without a display (e.g. in CI / headless servers)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Museum_Game_Interaction_Dataset.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
TABLE_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ordinal category orders (Section 4.6, Data Preprocessing)
# ---------------------------------------------------------------------------
AGE_GROUP_ORDER = ["3-5", "6-8", "9-12"]
COGNITIVE_LEVEL_ORDER = ["Early", "Developing", "Advanced"]
ENGAGEMENT_LEVEL_ORDER = ["Low", "Medium", "High"]
CROWD_LEVEL_ORDER = ["Low", "Medium", "High"]
PERFORMANCE_LEVEL_ORDER = ["Struggling", "Disengaged", "Optimal"]

CONTINUOUS_VARS = [
    "Time_Spent",
    "Total_Actions",
    "Correct_Responses",
    "Incorrect_Responses",
    "Hint_Usage",
    "Eye_Tracking_Focus_Duration",
    "Touch_Interactions",
    "Reaction_Time",
]

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.autolayout"] = True


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Museum Game Interaction Dataset and apply ordered categorical dtypes.

    Also derives Accuracy_Rate = Correct_Responses / Total_Actions, used
    throughout the EDA and as a covariate in the diagnostic analysis (Section 5.3).
    """
    df = pd.read_csv(path)

    df["Age_Group"] = pd.Categorical(df["Age_Group"], categories=AGE_GROUP_ORDER, ordered=True)
    df["Cognitive_Level"] = pd.Categorical(df["Cognitive_Level"], categories=COGNITIVE_LEVEL_ORDER, ordered=True)
    df["Engagement_Level"] = pd.Categorical(df["Engagement_Level"], categories=ENGAGEMENT_LEVEL_ORDER, ordered=True)
    df["Crowd_Level"] = pd.Categorical(df["Crowd_Level"], categories=CROWD_LEVEL_ORDER, ordered=True)
    df["Performance_Level"] = pd.Categorical(df["Performance_Level"], categories=PERFORMANCE_LEVEL_ORDER, ordered=False)

    df["Accuracy_Rate"] = df["Correct_Responses"] / df["Total_Actions"]

    return df


def save_table(df: pd.DataFrame, filename: str, index: bool = True) -> None:
    """Save a results table (e.g. hypothesis test output) as a CSV in outputs/tables/."""
    out_path = os.path.join(TABLE_DIR, filename)
    df.to_csv(out_path, index=index)
    print(f"  -> saved table: {out_path}")


def save_figure(fig, filename: str) -> None:
    """Save a matplotlib figure as a PNG in outputs/figures/."""
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved figure: {out_path}")
