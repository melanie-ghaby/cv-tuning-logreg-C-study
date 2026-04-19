import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#setting colors and dimensions that will be reused accrss figures for aesthetic consistency
background_col= "#FFFFFF"
panel_background_col="#D6E0F0"
main_col="#2E5FA3"
main_col2= "#3D8B6E"
main_col3= "#A63D2F"
text_col= "#0D1B2A"
subtext_col= "#4A6491"
grid_col= "#B8CADE"
white_col= "#F0F4FA"
title_col= "#0D1B2A"

plt.rcParams.update({
    "figure.facecolor":background_col,
    "axes.facecolor":panel_background_col,
    "axes.edgecolor":grid_col,
    "axes.labelcolor":text_col,
    "xtick.color":text_col,
    "ytick.color":text_col,
    "text.color":text_col,
    "grid.color":grid_col,
    "grid.linewidth":0.6,
    "font.family":"DejaVu Sans",
    "font.size":9,
    "axes.titlesize":11,
    "axes.titleweight":"bold",
    "axes.titlecolor":text_col,
    "axes.labelsize":9,
    "legend.facecolor":panel_background_col,
    "legend.edgecolor":grid_col,
    "legend.labelcolor":text_col,
})

#loading the data
phase1= pd.read_csv("phase1_synthetic_data.csv")
phase2=pd.read_csv("phase2_real_datasets.csv")
phase3_sig =pd.read_csv("phase3_gain_significance.csv")
phase3_corr= pd.read_csv("phase3_correlation_analysis.csv")


n_seeds=20
#helper functions for error bars computations in the figures
def se(std):
    return std/np.sqrt(n_seeds)
def ci95(std):
    return 1.96* se(std)

#extracting the phase 1 data

#splitting the rows into 3 groups, one for each structural property
ratio_df= phase1[phase1["property"]=="feature_sample_ratio"].reset_index(drop=True)
imbal_df =phase1[phase1["property"]=="class_imbalance_ratio"].reset_index(drop=True)
corr_df=phase1[phase1["property"]=="mean_feature_correlation"].reset_index(drop=True)

#extract x-axis values for each property panel from the property_value strings
ratios= ratio_df["property_value"].str.extract(r"=\s*([\d.]+)").astype(float).squeeze().tolist()
imbal_x=(imbal_df["property_value"].str.extract(r"split\s*\d+-(\d+)").astype(float).squeeze()/100).tolist()
corr_df= corr_df.iloc[:8].reset_index(drop=True)
red_x= corr_df["property_value"].str.extract(r"(\d+)\s*red").astype(float).squeeze().tolist()

#extracting the mean gain and std dev from each structural property
mean_gain_r= ratio_df["mean_gain"].tolist()
std_gain_r= ratio_df["std_gain"].tolist()
mean_gain_i= imbal_df["mean_gain"].tolist()
std_gain_i= imbal_df["std_gain"].tolist()
mean_gain_c= corr_df["mean_gain"].tolist()
std_gain_c= corr_df["std_gain"].tolist()

#extracting the phase 2 data and phase 3 for combined vis. (real-world datasets)
merged= pd.merge(phase2, phase3_sig, on="dataset")
#replace underscores with newlines for readable x-axis labels
dataset_labels= [d.replace("_","\n") for d in merged["dataset"].tolist()]
mean_gain_d= merged["mean_gain_x"].tolist()
ci_low= merged["ci_low"].tolist()
ci_high= merged["ci_high"].tolist()
significant= merged["significant_gain"].tolist()
mean_default_d= merged["mean_default"].tolist()
mean_tuned_d= merged["mean_tuned"].tolist()

#figure 1: syntehtic data, 3 side by side panels, one for each structural property
fig1,axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
fig1.patch.set_facecolor(background_col)

configs = [
    (axes[0], ratios,  mean_gain_r, std_gain_r,
     "Feature-to-Sample Ratio (p/n)",
     "Feature-to-Sample Ratio"),
    (axes[1], imbal_x, mean_gain_i, std_gain_i,
     "Class Imbalance Ratio\n(minority / majority class count)",
     "Minority Class Proportion"),
    (axes[2], red_x,   mean_gain_c, std_gain_c,
     "Feature Redundancy as Correlation Proxy\n(number of redundant features)",
     "Number of Redundant Features")
]

#drawing each panel
for ax,x,mg,sg,title, xlabel in configs:
    ci = [ci95(s) for s in sg]
    ax.axhline(0, color=main_col3, linewidth=1.2, linestyle="--", alpha=0.8, zorder=1)
    ax.grid(axis="y", zorder=0)
    colors = [main_col2 if v>0 else main_col3 for v in mg]
    ax.bar(range(len(x)), mg, color=colors, alpha=0.8,width=0.55,zorder=2)
    ax.errorbar(range(len(x)), mg, yerr=ci,fmt="none", color=text_col, capsize=3,capthick=1.2,elinewidth=1.2, zorder=3)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([f"{v:.2f}" if isinstance(v,float) else str(v) for v in x], rotation=45,ha="right", fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Mean Accuracy Gain (tuned − default)", fontsize=9)
    ax.set_title(title, color=title_col, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

#shared title among the 3 panels
fig1.suptitle("Figure 1.  Mean Accuracy Gain by Dataset Structural Property — Synthetic Study",color=title_col, fontsize=12, fontweight="bold", y=1.03)
fig1.text(0.5,-0.04, "Error bars = 95% CI around the mean gain (1.96 × SE).""Green = tuning improved accuracy,  Red = tuning degraded accuracy.", ha="center",fontsize=8, color=subtext_col ,style="italic")
fig1.tight_layout()
fig1.savefig("fig_synthetic.png",dpi=180,bbox_inches="tight",facecolor=background_col)
plt.close()
print("fig_synthetic.png saved")

#figure 2 for real datasets.  mean accuracy gain per real dataset with asymmetric 95% CI error bars
fig2,ax=plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor(background_col)

#x positions and  t-testCI error bar lengths
x =np.arange(len(dataset_labels))
ci_err_low = [ mg-lo for mg, lo in zip(mean_gain_d,ci_low) ]
ci_err_high =[ hi -mg for mg, hi in zip(mean_gain_d,ci_high) ]

bar_colors = []
for mg,sig in zip(mean_gain_d, significant):
    if sig and mg > 0:
        bar_colors.append(main_col2)
    elif mg < 0:
        bar_colors.append(main_col3)
    else:
        bar_colors.append(main_col)

#drawing bars, error lines, ref line
ax.bar(x, mean_gain_d, color=bar_colors, alpha=0.85, width=0.6, zorder=2)
ax.errorbar(x, mean_gain_d, yerr=[ci_err_low, ci_err_high],fmt="none", color=text_col, capsize=4, capthick=1.4,elinewidth=1.4, zorder=3)
ax.axhline(0, color=text_col, linewidth=0.8, linestyle="--", alpha=0.4)
ax.grid(axis="y", zorder=0)
ax.axvline(3.5, color=subtext_col, linewidth=1, linestyle=":", alpha=0.6)

#labels and seperators to make the graph clearer
ymax= max(ci_high) + 0.003
ax.text(1.5, ymax, "← Baseline", ha="center", va="bottom",color=subtext_col, fontsize=8.5, style="italic")
ax.text(7.5, ymax, "Targeted →",ha="center",va="bottom",color=subtext_col, fontsize=8.5, style="italic")
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels, fontsize=8.5)
ax.set_ylabel("Mean Accuracy Gain (tuned − default)", fontsize=9)
ax.set_title("Figure 2.  Mean Accuracy Gain Across Real Datasets", color=title_col, fontweight="bold", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)

legend_elements = [
    mpatches.Patch(color=main_col2, label="Statistically significant gain (p<0.05)"),
    mpatches.Patch(color=main_col,  label="Non-significant positive gain"),
    mpatches.Patch(color=main_col3, label="Non-significant loss")]

ax.legend(handles=legend_elements, fontsize=8.5, loc="upper left")
fig2.text(0.5, -0.04,"Error bars = 95% CI from one-sample t-test (H₀: gain = 0, df=19)." "Significance threshold: p<0.05.", ha="center", fontsize=8, color=subtext_col, style="italic")
fig2.tight_layout()
fig2.savefig("fig_real_datasets.png", dpi=180, bbox_inches="tight", facecolor=background_col)
plt.close()
print("fig_real_datasets.png saved")

#figure 3: scatterplot for the real datasets with line meaning no improvement
fig3,ax =plt.subplots(figsize=(7, 7))
fig3.patch.set_facecolor(background_col)

#putting the diagonal reference point of no tuning gain or loss
diag= np.linspace(0.46, 1.05, 100)
ax.plot(diag, diag, "--", color=subtext_col, linewidth=1.2, alpha=0.7, zorder=1)

#one color/dataset, diamond for signifiant gain
point_colors = ["#1B3A6B","#2E5FA3","#4A90D9","#6B5EA8","#A63D2F","#D4763B","#3D8B6E","#2ECC71","#E3B341","#E05C97"]
dataset_names_clean = [d.replace("\n", " ") for d in dataset_labels]
for xv,yv, name,sig,color in zip(mean_default_d, mean_tuned_d,dataset_names_clean, significant, point_colors):
    marker= "D" if sig else "o"
    ax.scatter(xv, yv, color=color,s=110, zorder=3, alpha=0.95,marker=marker, edgecolors="white", linewidths=0.8, label=name)

#setting the labels, titles, grid
ax.set_xlabel("Default Accuracy (C=1.0, scikit-learn default)", fontsize=9)
ax.set_ylabel("Tuned Accuracy (CV-selected C)", fontsize=9)
ax.set_title("Figure 3.  Default vs. Tuned Classification Accuracy", color=title_col, fontweight="bold", fontsize=11)
ax.grid(zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0.46, 1.06)
ax.set_ylim(0.46, 1.06)

#placing the legend outside of the plot for clarity
handles,labels = ax.get_legend_handles_labels()
diag_line= plt.Line2D([0], [0], linestyle="--",color=subtext_col,label="Diagonal: tuned = default")
diamond = plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="grey", markersize=7, label="◆ Significant gain (p<0.05)")
circle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",markersize=7, label="● Non-significant")
ax.legend(handles=handles + [diag_line, diamond, circle],fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, framealpha=0.9, labelspacing=1.2, handletextpad=0.8)

fig3.text(0.5, -0.03,"Each point = one dataset, mean accuracy averaged over 20 random seeds." "Points above diagonal = tuning helped.  Points below = tuning degraded accuracy.", ha="center", fontsize=8, color=subtext_col, style="italic")
fig3.tight_layout()
fig3.savefig("fig_scatter.png", dpi=180, bbox_inches="tight", facecolor=background_col)
plt.close()
print("fig_scatter.png saved")

#table 2: correlation table
table_data = []

#one row for each structural property
for _,row in phase3_corr.iterrows():
    table_data.append([
           row["property"],
        f"{row['pearson_r']:+.3f}",
        f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
        f"{row['pearson_p']:.3f}",
        f"{row['spearman_rho']:+.3f}",
        f"{row['spearman_p']:.3f}",
        str(row["strength"]).capitalize()])

#titles of each column
col_labels = ["Property", "Pearson r", "95% CI", "p-value", "Spearman ρ", "p-value (ρ)", "Strength"]

fig4,ax = plt.subplots(figsize=(11,3.2))
fig4.patch.set_facecolor(background_col)
ax.set_facecolor(background_col)
ax.axis("off")

#table creating with font and dimensions
tbl=ax.table(cellText=table_data, colLabels=col_labels,cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.9)

#first column is wider bc it contains text, need to widen it so all text visible
for row in range( len(table_data)+1 ):
    tbl[row, 0].set_width(0.28)
    for col in range(1, 7):
        tbl[row, col].set_width(0.12)

#styling each cell, alternating for clarity
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(main_col)
        cell.set_text_props(fontweight="bold", color=white_col)
    else:
        cell.set_facecolor(white_col if row % 2 == 0 else panel_background_col)
        cell.set_text_props(color=text_col)
    cell.set_edgecolor(grid_col)

#titles nad captions
ax.set_title("Table 2.  Pearson & Spearman Correlations: Structural Dataset Properties vs. Mean Accuracy Gain",color=title_col, fontweight="bold", fontsize=10, pad=14)
fig4.text(0.5, 0.01,"95% CI via Fisher z-transformation.  All correlations non-significant, therefore the results are exploratory given n=10.",ha="center", fontsize=7.5, color=subtext_col, style="italic")
fig4.tight_layout()
fig4.savefig("fig_correlations.png", dpi=180, bbox_inches="tight", facecolor=background_col)
plt.close()
print("fig_correlations.png saved")

#table 1: tageted datasets table

#hardcoding the rows and columns names
table_data_t = [
    ["Colon cancer","High feature/sample ratio","p/n = 32.26","+0.77%"],
    ["Diabetes pima","Low feature/sample ratio","p/n = 0.01","−0.06%"],
    ["Credit fraud","High class imbalance","minority = 0.18%", "+0.00%"],
    ["Banknote auth","Low class imbalance","minority = 44.8%", "+0.73%"],
    ["Abalone","High feature correlation", "r̄ = 0.68", "+0.12%"],
    ["Madelon", "Low feature correlation","r̄ = 0.02","+3.08%"] ]
col_labels_t = ["Dataset", "Target Property", "Property Value", "Mean Gain"]

fig5, ax5 = plt.subplots(figsize=(10, 3.0))
fig5.patch.set_facecolor(background_col)
ax5.set_facecolor(background_col)
ax5.axis("off") #no axes

#table creation
tbl5 = ax5.table(cellText=table_data_t, colLabels=col_labels_t,cellLoc="center", loc="center")
tbl5.auto_set_font_size(False)
tbl5.set_fontsize(9)
tbl5.scale(1, 2.0)

#alternating cell colors to make it more visible from far away
for (row,col),cell in tbl5.get_celld().items():
    if row == 0:
        cell.set_facecolor(main_col)
        cell.set_text_props(fontweight="bold", color=white_col)
    else:
        cell.set_facecolor(white_col if row % 2 == 0 else panel_background_col)
        cell.set_text_props(color=text_col)
    cell.set_edgecolor(grid_col)

ax5.set_title("Table 1.  Targeted Datasets: Structural Properties and Mean Accuracy Gain",color=title_col, fontweight="bold", fontsize=11, pad=14)
fig5.text(0.5, 0.01,"Each dataset selected for an extreme value of one structural property.""Significance from one-sample t-test (H₀: gain = 0, df=19, α=0.05).",ha="center", fontsize=7.5, color=subtext_col, style="italic")
fig5.tight_layout()
fig5.savefig("fig_targeted_table.png", dpi=180, bbox_inches="tight", facecolor=background_col)
plt.close()
print("fig_targeted_table.png saved")