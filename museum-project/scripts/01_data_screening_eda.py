"""
01_data_screening_eda.py
-------------------------
Data screening and exploratory data analysis for the Museum Game Interaction
Dataset, corresponding to Dissertation Sections 4.6 (Data Preprocessing),
4.7 (Exploratory Data Analysis) and 5.1 (Data Screening and Descriptive
Statistics).

Outputs:
    outputs/tables/table_5_1_descriptive_statistics.csv
    outputs/figures/figure_5_1_continuous_distributions.png
    outputs/figures/figure_5_2_categorical_distributions.png
    outputs/figures/figure_5_3_correlation_matrix.png

Run:
    python scripts/01_data_screening_eda.py
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data,
    save_table,
    save_figure,
    CONTINUOUS_VARS,
    AGE_GROUP_ORDER,
    COGNITIVE_LEVEL_ORDER,
    ENGAGEMENT_LEVEL_ORDER,
    CROWD_LEVEL_ORDER,
    PERFORMANCE_LEVEL_ORDER,
)


def screen_data(df: pd.DataFrame) -> None:
    """Print a data-quality report: shape, missingness, duplicate IDs, and
    the internal-consistency check that Correct + Incorrect == Total_Actions."""
    print("=" * 70)
    print("DATA SCREENING")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    missing = df.isna().sum().sum()
    print(f"Total missing values: {missing}")

    dup_participants = df["Participant_ID"].duplicated().sum()
    dup_sessions = df["Game_Session_ID"].duplicated().sum()
    print(f"Duplicate Participant_IDs: {dup_participants}")
    print(f"Duplicate Game_Session_IDs: {dup_sessions}")

    consistency_check = (df["Correct_Responses"] + df["Incorrect_Responses"] == df["Total_Actions"]).all()
    print(f"Correct_Responses + Incorrect_Responses == Total_Actions for all rows: {consistency_check}")

    print("\nCategorical group sizes:")
    for col, order in [
        ("Age_Group", AGE_GROUP_ORDER),
        ("Cognitive_Level", COGNITIVE_LEVEL_ORDER),
        ("Engagement_Level", ENGAGEMENT_LEVEL_ORDER),
        ("Crowd_Level", CROWD_LEVEL_ORDER),
        ("Performance_Level", PERFORMANCE_LEVEL_ORDER),
    ]:
        counts = df[col].value_counts().reindex(order)
        print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    completion_counts = df["Game_Completion_Status"].value_counts().sort_index()
    print(f"  Game_Completion_Status: 0={completion_counts.get(0, 0)}, 1={completion_counts.get(1, 0)}")


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Build Table 5.1: descriptive statistics for the continuous behavioural
    variables plus the derived Accuracy_Rate."""
    rows = []
    for col in CONTINUOUS_VARS + ["Accuracy_Rate"]:
        s = df[col]
        rows.append({
            "Variable": col,
            "Mean": round(s.mean(), 3),
            "SD": round(s.std(), 3),
            "Median": round(s.median(), 3),
            "Min": round(s.min(), 3),
            "Max": round(s.max(), 3),
            "Skew": round(skew(s.dropna()), 2),
        })
    table = pd.DataFrame(rows).set_index("Variable")
    print("\nTable 5.1: Descriptive statistics for continuous behavioural variables")
    print(table)
    return table


def plot_continuous_distributions(df: pd.DataFrame):
    """Figure 5.1: univariate distributions of the eight primary continuous
    behavioural variables plus the derived accuracy rate."""
    cols = CONTINUOUS_VARS + ["Accuracy_Rate"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Figure 5.1: Univariate distributions of continuous behavioural variables", y=1.02)
    return fig


def plot_categorical_distributions(df: pd.DataFrame):
    """Figure 5.2: distribution of the main categorical variables."""
    cat_specs = [
        ("Age_Group", AGE_GROUP_ORDER),
        ("Cognitive_Level", COGNITIVE_LEVEL_ORDER),
        ("Engagement_Level", ENGAGEMENT_LEVEL_ORDER),
        ("Crowd_Level", CROWD_LEVEL_ORDER),
        ("Performance_Level", PERFORMANCE_LEVEL_ORDER),
        ("Museum_Exhibit_Type", sorted(df["Museum_Exhibit_Type"].unique())),
        ("Facial_Expression_Sentiment", sorted(df["Facial_Expression_Sentiment"].unique())),
        ("Game_Completion_Status", [0, 1]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for ax, (col, order) in zip(axes, cat_specs):
        counts = df[col].value_counts().reindex(order)
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#55A868")
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Figure 5.2: Distribution of categorical variables (N = 1,000)", y=1.03)
    return fig


def plot_correlation_matrix(df: pd.DataFrame):
    """Figure 5.3: Pearson correlation matrix of continuous behavioural features."""
    corr = df[CONTINUOUS_VARS + ["Accuracy_Rate", "Game_Completion_Status"]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Figure 5.3: Pearson correlation matrix of continuous behavioural features")
    return fig


def main():
    df = load_data()

    screen_data(df)

    table_5_1 = descriptive_statistics(df)
    save_table(table_5_1, "table_5_1_descriptive_statistics.csv")

    save_figure(plot_continuous_distributions(df), "figure_5_1_continuous_distributions.png")
    save_figure(plot_categorical_distributions(df), "figure_5_2_categorical_distributions.png")
    save_figure(plot_correlation_matrix(df), "figure_5_3_correlation_matrix.png")

    print("\nData screening and EDA complete.")


if __name__ == "__main__":
    main()
