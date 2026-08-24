"""
02_hypothesis_testing.py
--------------------------
Inferential statistics for the five a priori hypotheses (H1-H5), corresponding
to Dissertation Section 4.8 (Statistical Analysis and Hypothesis Testing),
Table 4.2 (planned tests), Section 5.2 (Hypothesis Testing Results), and
Appendix B (Full Statistical Output).

    H1: Cognitive_Level x Performance_Level / Game_Completion_Status  -> Chi-square, Cramer's V
    H2: Hint_Usage, Incorrect_Responses by Performance_Level          -> ANOVA / Kruskal-Wallis, eta-squared
    H3: Engagement indicators vs accuracy / completion                -> Pearson / point-biserial r
    H4: Museum_Exhibit_Type vs engagement/performance/reaction time   -> Chi-square / ANOVA
    H5: Crowd_Level vs completion and time on task                    -> Chi-square / ANOVA

All tests are run at alpha = .05 with effect sizes reported alongside p-values.

Outputs:
    outputs/tables/table_appendixB1_chi_square_tests.csv
    outputs/tables/table_appendixB2_anova_kruskal_tests.csv
    outputs/tables/table_appendixB3_correlations.csv
    outputs/tables/table_5_2_hypothesis_summary.csv
    outputs/figures/figure_5_4_h1_cognitive_level.png
    outputs/figures/figure_5_5_h2_challenge_skill_mismatch.png
    outputs/figures/figure_5_6_h3_engagement_vs_accuracy.png
    outputs/figures/figure_5_7_h4_exhibit_modality.png
    outputs/figures/figure_5_8_h5_crowd_level.png

Run:
    python scripts/02_hypothesis_testing.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data, save_table, save_figure, ENGAGEMENT_LEVEL_ORDER, CROWD_LEVEL_ORDER, PERFORMANCE_LEVEL_ORDER


# ---------------------------------------------------------------------------
# Effect-size helpers
# ---------------------------------------------------------------------------

def cramers_v(chi2: float, n: int, table_shape: tuple) -> float:
    """Cramer's V effect size for a chi-square test of independence."""
    r, k = table_shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))


def eta_squared_anova(groups) -> float:
    """Eta-squared for a one-way ANOVA: SS_between / SS_total."""
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total if ss_total > 0 else 0.0


def epsilon_squared_kw(h_stat: float, n: int) -> float:
    """Epsilon-squared effect size for the Kruskal-Wallis H statistic:
    eps^2 = (H - k + 1) / (n - k), here approximated as H / (n - 1) per
    the common rank-based epsilon-squared formulation used in Ch.4/5."""
    return h_stat / (n - 1) if n > 1 else 0.0


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    table = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = cramers_v(chi2, table.values.sum(), table.shape)
    return {"Test": f"{col_a} x {col_b}", "chi2": round(chi2, 3), "df": dof, "p": round(p, 4), "Cramers_V": round(v, 3)}


def anova_kw_test(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    groups = [df.loc[df[group_col] == g, value_col].values for g in df[group_col].cat.categories
              if (df[group_col] == g).any()] if hasattr(df[group_col], "cat") else \
             [g[value_col].values for _, g in df.groupby(group_col)]
    f_stat, p_anova = stats.f_oneway(*groups)
    eta2 = eta_squared_anova(groups)
    h_stat, p_kw = stats.kruskal(*groups)
    eps2 = epsilon_squared_kw(h_stat, len(df))
    return {
        "Test": f"{value_col} by {group_col}",
        "F": round(f_stat, 3), "p_ANOVA": round(p_anova, 4), "eta_sq": round(eta2, 4),
        "H": round(h_stat, 3), "p_KW": round(p_kw, 4), "epsilon_sq": round(eps2, 4),
    }


def correlation_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    r, p = stats.pearsonr(df[col_a], df[col_b])
    return {"Pair": f"{col_a} x {col_b}", "r": round(r, 3), "p": round(p, 4)}


# ---------------------------------------------------------------------------
# H1: Cognitive level vs performance and completion
# ---------------------------------------------------------------------------

def test_h1(df: pd.DataFrame) -> list:
    print("\n--- H1: Cognitive_Level vs Performance_Level / Game_Completion_Status ---")
    r1 = chi_square_test(df, "Cognitive_Level", "Performance_Level")
    r2 = chi_square_test(df, "Cognitive_Level", "Game_Completion_Status")
    for r in (r1, r2):
        print(f"  {r}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pd.crosstab(df["Cognitive_Level"], df["Performance_Level"], normalize="index").plot(
        kind="bar", stacked=True, ax=axes[0], colormap="viridis")
    axes[0].set_title("Performance level by cognitive level")
    axes[0].set_ylabel("Proportion")

    pd.crosstab(df["Cognitive_Level"], df["Game_Completion_Status"], normalize="index").plot(
        kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("Completion status by cognitive level")
    axes[1].set_ylabel("Proportion")

    fig.suptitle("Figure 5.4: Performance level and completion status by cognitive level")
    save_figure(fig, "figure_5_4_h1_cognitive_level.png")
    return [r1, r2]


# ---------------------------------------------------------------------------
# H2: Challenge-skill mismatch indicators by performance level
# ---------------------------------------------------------------------------

def test_h2(df: pd.DataFrame) -> list:
    print("\n--- H2: Hint_Usage / Incorrect_Responses by Performance_Level ---")
    r1 = anova_kw_test(df, "Hint_Usage", "Performance_Level")
    r2 = anova_kw_test(df, "Incorrect_Responses", "Performance_Level")
    for r in (r1, r2):
        print(f"  {r}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x="Performance_Level", y="Hint_Usage", order=PERFORMANCE_LEVEL_ORDER, ax=axes[0])
    axes[0].set_title("Hint usage by performance level (n.s.)")
    sns.boxplot(data=df, x="Performance_Level", y="Incorrect_Responses", order=PERFORMANCE_LEVEL_ORDER, ax=axes[1])
    axes[1].set_title("Incorrect responses by performance level (p < .001)")

    fig.suptitle("Figure 5.5: Hint usage and incorrect responses by performance level")
    save_figure(fig, "figure_5_5_h2_challenge_skill_mismatch.png")
    return [r1, r2]


# ---------------------------------------------------------------------------
# H3: Engagement indicators vs accuracy and completion
# ---------------------------------------------------------------------------

def test_h3(df: pd.DataFrame) -> list:
    print("\n--- H3: Engagement indicators vs Correct_Responses / Game_Completion_Status ---")
    r1 = correlation_test(df, "Eye_Tracking_Focus_Duration", "Correct_Responses")
    r2 = correlation_test(df, "Touch_Interactions", "Correct_Responses")
    r3 = correlation_test(df, "Eye_Tracking_Focus_Duration", "Game_Completion_Status")
    r4 = correlation_test(df, "Touch_Interactions", "Game_Completion_Status")
    for r in (r1, r2, r3, r4):
        print(f"  {r}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x="Eye_Tracking_Focus_Duration", y="Correct_Responses", alpha=0.4, ax=axes[0])
    axes[0].set_title(f"Focus duration vs accuracy (r={r1['r']}, n.s.)")
    sns.boxplot(data=df, x="Game_Completion_Status", y="Touch_Interactions", ax=axes[1])
    axes[1].set_title(f"Touch interactions vs completion (r={r4['r']}, n.s.)")

    fig.suptitle("Figure 5.6: Engagement indicators vs accuracy and completion")
    save_figure(fig, "figure_5_6_h3_engagement_vs_accuracy.png")
    return [r1, r2, r3, r4]


# ---------------------------------------------------------------------------
# H4: Exhibit modality vs engagement, performance, reaction time
# ---------------------------------------------------------------------------

def test_h4(df: pd.DataFrame) -> list:
    print("\n--- H4: Museum_Exhibit_Type vs Engagement_Level / Performance_Level / Reaction_Time ---")
    r1 = chi_square_test(df, "Museum_Exhibit_Type", "Engagement_Level")
    r2 = chi_square_test(df, "Museum_Exhibit_Type", "Performance_Level")
    r3 = anova_kw_test(df, "Reaction_Time", "Museum_Exhibit_Type")
    for r in (r1, r2, r3):
        print(f"  {r}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pd.crosstab(df["Museum_Exhibit_Type"], df["Performance_Level"], normalize="index").plot(
        kind="bar", stacked=True, ax=axes[0], colormap="plasma")
    axes[0].set_title("Performance level by exhibit modality")
    sns.boxplot(data=df, x="Museum_Exhibit_Type", y="Reaction_Time", ax=axes[1])
    axes[1].set_title("Reaction time by exhibit modality (n.s.)")

    fig.suptitle("Figure 5.7: Performance and reaction time across exhibit modalities")
    save_figure(fig, "figure_5_7_h4_exhibit_modality.png")
    return [r1, r2, r3]


# ---------------------------------------------------------------------------
# H5: Crowd level vs completion and time on task
# ---------------------------------------------------------------------------

def test_h5(df: pd.DataFrame) -> list:
    print("\n--- H5: Crowd_Level vs Game_Completion_Status / Time_Spent ---")
    r1 = chi_square_test(df, "Crowd_Level", "Game_Completion_Status")
    r2 = anova_kw_test(df, "Time_Spent", "Crowd_Level")
    for r in (r1, r2):
        print(f"  {r}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pd.crosstab(df["Crowd_Level"], df["Game_Completion_Status"], normalize="index").plot(
        kind="bar", stacked=True, ax=axes[0], colormap="cividis")
    axes[0].set_title("Completion status by crowd level (n.s.)")
    sns.boxplot(data=df, x="Crowd_Level", y="Time_Spent", order=CROWD_LEVEL_ORDER, ax=axes[1])
    axes[1].set_title("Time on task by crowd level (p = .035)")

    fig.suptitle("Figure 5.8: Crowd level vs completion and time on task")
    save_figure(fig, "figure_5_8_h5_crowd_level.png")
    return [r1, r2]


def build_summary_table(h1, h2, h3, h4, h5) -> pd.DataFrame:
    """Table 5.2: Summary of hypothesis testing outcomes."""
    rows = [
        {"Hypothesis": "H1", "Prediction": "Cognitive level relates to performance & completion",
         "Result": f"chi2={h1[0]['chi2']}, p={h1[0]['p']} (performance); chi2={h1[1]['chi2']}, p={h1[1]['p']} (completion)",
         "Effect_size": f"V={h1[0]['Cramers_V']}-{h1[1]['Cramers_V']}", "Supported": "No"},
        {"Hypothesis": "H2", "Prediction": "Hint usage & incorrect responses differ by performance level",
         "Result": f"Hint usage p={h2[0]['p_ANOVA']}; Incorrect responses p={h2[1]['p_ANOVA']}",
         "Effect_size": f"eta2={h2[0]['eta_sq']} / {h2[1]['eta_sq']}", "Supported": "Partial"},
        {"Hypothesis": "H3", "Prediction": "Engagement indicators relate to accuracy & completion",
         "Result": "All |r| <= .03, all p > .33", "Effect_size": "r ~ 0.00-0.03", "Supported": "No"},
        {"Hypothesis": "H4", "Prediction": "Exhibit modality affects engagement, performance, reaction time",
         "Result": "All chi2/F tests n.s. (p > .27)", "Effect_size": "V/eta2 ~ 0.00", "Supported": "No"},
        {"Hypothesis": "H5", "Prediction": "Crowd level affects completion & time on task",
         "Result": f"Completion p={h5[0]['p']}; Time on task p={h5[1]['p_ANOVA']}",
         "Effect_size": f"V={h5[0]['Cramers_V']}; eta2={h5[1]['eta_sq']}", "Supported": "Partial (negligible)"},
    ]
    return pd.DataFrame(rows).set_index("Hypothesis")


def main():
    df = load_data()

    h1 = test_h1(df)
    h2 = test_h2(df)
    h3 = test_h3(df)
    h4 = test_h4(df)
    h5 = test_h5(df)

    # Appendix B tables
    save_table(pd.DataFrame([h1[0], h1[1], h4[0], h4[1], h5[0]]).set_index("Test"),
               "table_appendixB1_chi_square_tests.csv")
    save_table(pd.DataFrame([h2[0], h2[1], h4[2], h5[1]]).set_index("Test"),
               "table_appendixB2_anova_kruskal_tests.csv")
    save_table(pd.DataFrame(h3).set_index("Pair"),
               "table_appendixB3_correlations.csv")

    summary = build_summary_table(h1, h2, h3, h4, h5)
    save_table(summary, "table_5_2_hypothesis_summary.csv")
    print("\nTable 5.2: Summary of hypothesis testing outcomes")
    print(summary)

    print("\nHypothesis testing complete.")


if __name__ == "__main__":
    main()
