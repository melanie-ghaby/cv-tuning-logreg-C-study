import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


#3 different structural properties of datasets are being evaluated
#for each property, 10 levels of differing intensity are evaluated
#and 20 random seeds are tested for every level
#the same 20 seeds are used across all properties so that
#any differences in gain between properties are not due to
#which random splits happened to be used

seeds=list(range(20))
c_vals=[0.001, 0.01, 0.1, 1, 10, 100]

#takes as input a dataset (X, y) and a seed number
#returns: default accuracy,tuned accuracy,tuning gain,best c value
def run_one_seed(X,y,seed):
    X_train,X_test,y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=seed)

    #standardizing the data (s.t. every feature has mean 0 and std dev 1)
    #fit only on training data (so no leak), but applied to both
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #baseline with no tuning
    #training logistic regression with sklearn's default C=1.0
    #evaluating on the test and storing the accuracy
    default_model=LogisticRegression(C=1.0, random_state=seed, max_iter=2000)
    default_model.fit(X_train_scaled, y_train)
    default_acc= default_model.score(X_test_scaled, y_test)

    #trying each of the 6 candidate C values
    #for each one,we run 5-fold cross-validation only on the training set
    #we keep track of the C that produced the best average validation accuracy
    best_C,best_cv = None,-1
    for C in c_vals:
        model=LogisticRegression(C=C, random_state=seed, max_iter=2000)
        cv_scores= cross_val_score(model,X_train_scaled, y_train, cv=5, scoring='accuracy')
        if cv_scores.mean() > best_cv:
            best_cv=cv_scores.mean()
            best_C=C

    #training one final model using all the training data for best C, not just 4/5 like during cross-validation
    #evaluate on the test set to get the tuned accuracy
    tuned_model=LogisticRegression(C=best_C, random_state=seed, max_iter=2000)
    tuned_model.fit(X_train_scaled, y_train)
    tuned_acc=tuned_model.score(X_test_scaled, y_test)

    return default_acc, tuned_acc, tuned_acc - default_acc, best_C


def run_averaged(X, y, label, property_name, property_value):
    #to collect data accross all 20 seeds
    default_accs, tuned_accs, gains, best_Cs = [], [], [], []

    #testing for each seed
    for seed in seeds:
        d,t,g,c = run_one_seed(X, y, seed)
        default_accs.append(d)
        tuned_accs.append(t)
        gains.append(g)
        best_Cs.append(c)

    #counting how many times each C value was chosen across the 20 seeds
    C_counts= {}
    for c in best_Cs:
        if c in C_counts:
            C_counts[c]= C_counts[c]+1
        else:
            C_counts[c]= 1

    #finding the post popular C
    modal_C=None
    highest_count=0
    for c,count in C_counts.items():
        if count > highest_count:
            highest_count=count
            modal_C=c

    return {
        "property": property_name,
        "level": label,
        "property_value": property_value,
        "mean_default": round(np.mean(default_accs), 4),  #avg of the 20 default accuracies
        "mean_tuned": round(np.mean(tuned_accs), 4),  #avg of the 20 tuned accuracies
        "mean_gain": round(np.mean(gains), 4),  #avg gain across 20 seeds
        "std_gain": round(np.std(gains), 4),  #std deviation across that gain
        "gain_pct": f"{np.mean(gains) * 100:+.2f}%",
        "std_pct": f"+/-{np.std(gains)*100:.2f}%",
        "modal_best_C": modal_C
    }
results = []

######## dataset types 1: varying feature/sample ratios

#producing 10 different synthetic datasets
#fixed properties: n_features=20, n_informative=8, n_redundant=1, class_sep=1.0
#variying properties: n_samples from 1000 (very low ratio) down to 60 (very high ratio)


#each tuple contains: # samples to generate, a label for the level, the ratio
configs_ratio = [
    (1000, "01_ratio=0.02", "20/1000 = 0.020"),
    (667, "02_ratio=0.03", "20/667  = 0.030"),
    (500, "03_ratio=0.04", "20/500  = 0.040"),
    (333, "04_ratio=0.06", "20/333  = 0.060"),
    (250, "05_ratio=0.08", "20/250  = 0.080"),
    (200, "06_ratio=0.10", "20/200  = 0.100"),
    (150, "07_ratio=0.13", "20/150  = 0.133"),
    (120, "08_ratio=0.17", "20/120  = 0.167"),
    (100, "09_ratio=0.20", "20/100  = 0.200"),
    (80, "10_ratio=0.25", "20/80   = 0.250")
]
for n_samples, level, prop_val in configs_ratio:
    #generating the synthetic dataset
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=8,
        n_redundant=1, n_repeated=0, n_classes=2,
        class_sep=1.0, weights=None, random_state=42
    )
    # running the experiment with 20 seeds
    r = run_averaged(X, y, level, "feature_sample_ratio", prop_val)
    results.append(r)

#### property 2: class imbalance
#fixed properties: n_samples=500, n_features=10, n_informative=8, class_sep=1.0
#varied properties: minority class proportion from 45% down to 1%


#weights=[0.55, 0.45] means 55% of samples belong to class 0 and 45% to class 1
#Everything  except class proportions is identical across all 10 levels.
configs_imbal = [
    ([0.55, 0.45], "01_split=55/45",  "split 55-45"),
    ([0.60, 0.40], "02_split=60/40",  "split 60-40"),
    ([0.65, 0.35], "03_split=65/35",  "split 65-35"),
    ([0.70, 0.30], "04_split=70/30",  "split 70-30"),
    ([0.75, 0.25], "05_split=75/25",  "split 75-25"),
    ([0.80, 0.20], "06_split=80/20",  "split 80-20"),
    ([0.85, 0.15], "07_split=85/15",  "split 85-15"),
    ([0.90, 0.10], "08_split=90/10",  "split 90-10"),
    ([0.95, 0.05], "09_split=95/5",   "split 95-5"),
    ([0.99, 0.01], "10_split=99/1",   "split 99-1")
]

for weights, level, prop_val in configs_imbal:
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=1, n_repeated=0, n_classes=2,
        class_sep=1.0, weights=weights, random_state=42
    )
    r = run_averaged(X, y, level, "class_imbalance_ratio", prop_val)
    results.append(r)

#### property 3: feature correlation
# fixed properties: n_samples=500, n_features=10, class_sep=1.0
# varied properties: number of informative vs redundant features (more redundant = higher correlation)


# 1st number is n_informative, 2n is n_redundant.
# redudant features are linear combinations of the informative ones
configs_corr = [
    (9, 0, "01_9inf_0red", "9 inf / 0 red"),
    (8, 1, "02_8inf_1red", "8 inf / 1 red"),
    (7, 2, "03_7inf_2red", "7 inf / 2 red"),
    (6, 3, "04_6inf_3red", "6 inf / 3 red"),
    (5, 4, "05_5inf_4red", "5 inf / 4 red"),
    (4, 5, "06_4inf_5red", "4 inf / 5 red"),
    (3, 6, "07_3inf_6red", "3 inf / 6 red"),
    (2, 7, "08_2inf_7red", "2 inf / 7 red"),
    (2, 6, "09_2inf_6red_s", "2 inf / 6 red (n=9)"),  # n_features=9 bc skt-learn doesnt allow  <2 informative
    (2, 5, "10_2inf_5red_s", "2 inf / 5 red (n=8)"),  # n_features=8
]
feature_counts = [10, 10, 10, 10, 10, 10, 10, 10, 9, 8]
for (n_inf, n_red, level, prop_val), n_feat in zip(configs_corr, feature_counts):
    X, y = make_classification(
        n_samples=500, n_features=n_feat, n_informative=n_inf,
        n_redundant=n_red, n_repeated=0, n_classes=2,
        class_sep=1.0, weights=None, random_state=42
    )
    r = run_averaged(X, y, level, "mean_feature_correlation", prop_val)
    results.append(r)

#summary table
df = pd.DataFrame(results)[[
    "property", "level", "property_value",
    "mean_default", "mean_tuned", "mean_gain", "std_gain",
    "gain_pct", "std_pct", "modal_best_C"
]]
df.to_csv("phase1_synthetic_data.csv", index=False)
print("\nsaved to phase1_synthetic_data.csv ")