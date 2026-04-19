# CV Tuning of Logistic Regression's C - Empirical Study

This repository contains the code and data for an empirical study investigating whether cross-validated tuning of logistic regression's regularization parameter C reliably improves classification accuracy, and whether dataset structural properties predict when tuning helps.

## Research Question
Does CV-based tuning of logistic regression's C outperform scikit-learn's default C=1.0, and under what dataset conditions?

## Study Design
- **Phase 1:** Synthetic datasets generated with scikit-learn's `make_classification`, varying three structural properties (feature-to-sample ratio, class imbalance, feature redundancy) across 10 intensity levels each, with 20 random seeds per condition
- **Phase 2:** 10 real-world datasets (4 baseline from scikit-learn + 6 targeted from OpenML for extreme structural property values), each run across 20 random seeds
- **Statistical analysis:** one-sample t-tests per dataset, sign test, Pearson and Spearman correlation analysis

## Files
| File | Description |
|------|-------------|
| `phase1_synthetic_data.py` | Phase 1 pipeline |
| `phase2_real_datasets.py` | Phase 2 pipeline |
| `descriptive_statistics.py` | Statistical analysis (t-tests, correlations, sign test) |
| `visualizations.py` | All figures and tables |
| `*.csv` | Intermediate results from each phase |

## Requirements
```
Python 3.14.0
scikit-learn 1.8.0
numpy 2.4.4
pandas 3.0.2
scipy 1.17.1
```
