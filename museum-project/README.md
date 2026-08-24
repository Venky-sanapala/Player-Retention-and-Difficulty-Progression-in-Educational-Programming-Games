# Player Retention and Difficulty Progression in Educational Programming Games
### An Empirical Analysis of the Museum Game Interaction Dataset

MSc Data Science — Research Project (AM41PR), Aston University, 2024–2025
Author: Venkatesh Sanapala

This repository contains the full data-analysis pipeline for the dissertation
*"Player Retention and Difficulty Progression in Educational Programming
Games: An Empirical Analysis of the Museum Game Interaction Dataset"*. It
reproduces every descriptive statistic, figure, and inferential test reported
in the dissertation (Chapters 4 and 5, and Appendices A–B) from the raw
dataset.

## Overview

The dataset (`N = 1,000`) records single-session interactions of children
aged 3–12 with a museum-based educational programming exhibit, across three
interaction modalities (VR-based, AR-based, Physical Display). Because the
data is cross-sectional and single-session, "retention" and "difficulty
progression" are operationalised as theory-driven behavioural proxies
(Section 4.5) rather than measured longitudinally.

Five a priori hypotheses (H1–H5) are tested:

| # | Hypothesis | Test |
|---|---|---|
| H1 | Cognitive level relates to performance level & completion status | Chi-square, Cramér's V |
| H2 | Hint usage & incorrect responses differ by performance-level group | ANOVA / Kruskal-Wallis, η² |
| H3 | Engagement indicators correlate with accuracy & completion | Pearson / point-biserial r |
| H4 | Exhibit modality affects engagement, performance, reaction time | Chi-square / ANOVA |
| H5 | Crowd level affects completion & time on task | Chi-square / ANOVA |

A diagnostic follow-up analysis (Section 5.3) then shows that
`Performance_Level` is structurally derived from `Engagement_Level` and
session accuracy, which explains why most of H1–H5 return null results.

## Repository structure

```
.
├── data/
│   └── Museum_Game_Interaction_Dataset.csv   # raw dataset (N = 1,000, 18 variables)
├── scripts/
│   ├── utils.py                       # shared data loading / plotting helpers
│   ├── 01_data_screening_eda.py       # Sections 4.6-4.7, 5.1 — screening, Table 5.1, Figures 5.1-5.3
│   ├── 02_hypothesis_testing.py       # Section 5.2, Table 4.2, Appendix B — H1-H5, Figures 5.4-5.8
│   └── 03_diagnostic_analysis.py      # Section 5.3, Appendix B.4 — Figure 5.9
├── outputs/
│   ├── figures/                       # generated PNGs (Figures 5.1-5.9)
│   └── tables/                        # generated CSVs (Tables 5.1-5.2, Appendix B.1-B.4)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <this-repo-url>
cd museum-game-interaction-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the analysis

Run the three scripts in order from the `scripts/` directory (each writes its
outputs into `outputs/tables/` and `outputs/figures/`):

```bash
cd scripts
python 01_data_screening_eda.py
python 02_hypothesis_testing.py
python 03_diagnostic_analysis.py
```

## Key findings

Four of the five hypotheses returned mostly null results. The only
statistically significant effects were:

- **H2**: Incorrect responses differed significantly across performance-level
  groups (p < .001, ε² = .15) — a small-to-moderate effect. Hint usage did
  not differ significantly.
- **H5**: Time on task differed significantly by crowd level (p = .035),
  though the effect size was negligible (η² = .007).

All other tests (H1, H3, H4) were non-significant. The diagnostic analysis in
Section 5.3 shows that `Performance_Level` is mechanically derived from
`Engagement_Level` and accuracy rate rather than being an independent
behavioural outcome — which accounts for the otherwise-surprising pattern of
null findings across the other predictors (cognitive level, exhibit
modality, crowd level).

## Dataset variables

See Appendix A of the dissertation (`Table A.1`) for the complete 18-variable
feature inventory, or inspect `data/Museum_Game_Interaction_Dataset.csv`
directly — the column headers are self-describing (e.g. `Age_Group`,
`Cognitive_Level`, `Game_Completion_Status`, `Performance_Level`).

## Tools

Python 3, pandas, numpy, scipy.stats, matplotlib, seaborn.

## Ethics

The dataset contains no directly identifying information; participants are
coded with anonymous IDs (`P1`–`P1000`).
