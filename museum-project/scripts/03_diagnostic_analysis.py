"""
03_diagnostic_analysis.py
---------------------------
Diagnostic investigation into the structural basis of Performance_Level,
corresponding to Dissertation Section 5.3 (Diagnostic Finding) and
Appendix B.4 (Kruskal-Wallis test: Correct_Responses by Engagement_Level).

Since four of the five pre-registered hypotheses returned null results,
this script cross-tabulates Engagement_Level and accuracy against
Performance_Level to show that Performance_Level is structurally derived
from these two variables rather than an independent behavioural outcome.

Outputs:
    outputs/tables/table_appendixB4_kw_correct_by_engagement.csv
    outputs/figures/figure_5_9_performance_level_structure.png

Run:
    python scripts/03_diagnostic_analysis.py
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, save_table, save_figure, ENGAGEMENT_LEVEL_ORDER, PERFORMANCE_LEVEL_ORDER


def kruskal_correct_by_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Appendix B.4: Kruskal-Wallis test of Correct_Responses across
    Engagement_Level groups."""
    groups = [df.loc[df["Engagement_Level"] == g, "Correct_Responses"].values for g in ENGAGEMENT_LEVEL_ORDER]
    h_stat, p = stats.kruskal(*groups)
    n = len(df)
    k = len(groups)
    eps2 = (h_stat - k + 1) / (n - k)

    result = pd.DataFrame(
        {"Statistic": ["H", "df", "p", "epsilon_sq"],
         "Value": [round(h_stat, 3), k - 1, round(p, 4), round(eps2, 4)]}
    ).set_index("Statistic")

    print("Appendix B.4: Kruskal-Wallis test - Correct_Responses by Engagement_Level")
    print(result)
    return result


def plot_performance_level_structure(df: pd.DataFrame):
    """Figure 5.9: show that Performance_Level tracks Engagement_Level and
    Accuracy_Rate almost deterministically, explaining the null findings
    for H1-H5 (Section 5.3)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ct = pd.crosstab(df["Engagement_Level"], df["Performance_Level"], normalize="index").reindex(ENGAGEMENT_LEVEL_ORDER)
    ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="magma")
    axes[0].set_title("Performance level composition by engagement level")
    axes[0].set_ylabel("Proportion")

    sns.boxplot(data=df, x="Performance_Level", y="Accuracy_Rate", order=PERFORMANCE_LEVEL_ORDER, ax=axes[1])
    axes[1].set_title("Accuracy rate by performance level")

    fig.suptitle(
        "Figure 5.9: Performance_Level is almost entirely a function of\n"
        "Engagement_Level and session accuracy rate"
    )
    return fig


def main():
    df = load_data()

    kw_table = kruskal_correct_by_engagement(df)
    save_table(kw_table, "table_appendixB4_kw_correct_by_engagement.csv")

    fig = plot_performance_level_structure(df)
    save_figure(fig, "figure_5_9_performance_level_structure.png")

    # Cross-tab printed to console for quick inspection, mirrors Section 5.3 narrative
    print("\nCross-tabulation: Engagement_Level x Performance_Level (row %)")
    print((pd.crosstab(df["Engagement_Level"], df["Performance_Level"], normalize="index") * 100).round(1))

    print("\nDiagnostic analysis complete.")


if __name__ == "__main__":
    main()
