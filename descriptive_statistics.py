import pandas as pd
import numpy as np
from scipy import stats

#loading the data
df = pd.read_csv("phase2_real_datasets.csv")

n_seeds = 20
n_datasets = len(df)

#gain significance (t-test + confidence intervals)
#for each dataset: is the mean gain across 20 seeds statistically different from zero?
#done using one-sample t-test

#std error 
df["se_gain"] = df["std_gain"] / np.sqrt(n_seeds)
#if std=0 bc all 20 seends have the same gain, to avoid division by 0
df["se_gain"] = df["se_gain"].replace(0, np.nan)

#how many std errors the mean gain is away from zero
df["t_stat"] = df["mean_gain"] / df["se_gain"]

#to 2-tailed p-val with 19 degrees of freedom bc we have 20 sqeeds
df["p_value_gain"]= stats.t.sf(np.abs(df["t_stat"]), df=n_seeds-1 ) * 2

#critical t-value for a 95% confidence interval with 19 degrees of freedom
#we use 0.975 bc 2-tailed test
t_crit =stats.t.ppf(0.975, df=n_seeds - 1)

#if 95% CI has 0, gain not statistically significant
df["ci_low"]= df["mean_gain"]- t_crit * df["se_gain"]
df["ci_high"] =df["mean_gain"] + t_crit * df["se_gain"]

#if p-val < 0.05, gain is statistically distinguishable from zero across 20 seeds
df["significant_gain"] = df["p_value_gain"] < 0.05

df[["dataset", "type", "mean_gain", "ci_low", "ci_high", "p_value_gain", "significant_gain"]].to_csv("phase3_gain_significance.csv", index=False)

#analysign the correlation (Pearson + CI + Spearman)

#fisher_ci computes a 95% confidence interval around a Pearson r value
#(can't compute CIs directly on r bc its [-1,1])
def fisher_ci(r, n):

    #if r= 1 or -1, arctanh is undefined so CI is the pt itself alone
    if abs(r) == 1:
        return r, r

    #transforming to an unbounded scale to compute the CIS
    z=np.arctanh(r)
    # std error of z
    se=1/np.sqrt(n-3)

    #computing interval bounds
    z_crit= 1.96 #critical value for a 95% two-tailed interval
    lo= z - z_crit*se
    hi= z + z_crit*se

    #transforming back to original scale to get 95% CI interval around o.g. r value
    return np.tanh(lo), np.tanh(hi)

#structural properties were trying to correlate with gain
#keys: column names in the dataframe, vals: labels
properties = {
    "feature_sample_ratio":     "Feature/Sample Ratio",
    "class_imbalance_ratio":    "Class Imbalance Ratio",
    "mean_feature_correlation": "Mean Feature Correlation",
    "n_samples":                "N Samples",
    "n_features":               "N Features"
}

#Note: correlations are computed across 10 datasets (n=10), so results are exploratory.


#empty list to collect results for saving to CSV later
corr_results = []

#looping through each structural property
for col, label in properties.items():

    #pearson r: measures linear relationship between this property and gain
    #p: two-tailed p-value for pearson
    r,p = stats.pearsonr(df[col], df["mean_gain"])

    #spearman rho: measures monotonic relationship (more robust to outliers)
    #p_s: two-tailed p-value for spearman
    #we compute both because one extreme dataset could distort pearson alone
    rho,p_s= stats.spearmanr( df[col], df["mean_gain"] )
    #compute 95% confidence interval around pearson r using fisher transformation
    ci_low,ci_high= fisher_ci(r, n_datasets)

    #classifying the strength of the correlation fomm |r|
    #(indep. of p value) with n=10, almost nothing will be significant, so effect size matters more
    if abs(r) >= 0.7:
        strength="strong"
    elif abs(r) >= 0.5:
        strength="moderate"
    elif abs(r) >= 0.3:
        strength="weak"
    else:
        strength="very weak"

    #results for csv
    corr_results.append({
        "property": label,
        "pearson_r": round(r, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "pearson_p": round(p, 3),
        "spearman_rho": round(rho, 3),
        "spearman_p": round(p_s, 3),
        "strength": strength
    })

#sign test (binomial)

# # of datasets with  positive mean gain across 20 seeds
n_positive=(df["mean_gain"] > 0).sum()
n_total=len(df)
prop_positive=n_positive/n_total

#binomial test w ith null hypthesis: tuning has no effect,
binom_result=stats.binomtest(n_positive, n_total, p=0.5)

pd.DataFrame([{
    "n_positive": int(n_positive),
    "n_total": n_total,
    "prop_positive": round(prop_positive, 4),
    "binom_p_value": round(binom_result.pvalue, 4),
    "significant": binom_result.pvalue < 0.05
}]).to_csv("phase3_sign_test.csv", index=False)


display_cols = ["dataset", "type","feature_sample_ratio", "class_imbalance_ratio", "mean_feature_correlation","mean_gain", "std_gain", "se_gain","ci_low", "ci_high", "p_value_gain", "significant_gain"]

#saving results

pd.DataFrame(corr_results).to_csv("phase3_correlation_analysis.csv",index=False)
df[display_cols].to_csv("phase3_descriptive_analysis.csv",index=False)

print("\nSaved to phase3_correlation_analysis.csv and phase3_descriptive_analysis.csv")