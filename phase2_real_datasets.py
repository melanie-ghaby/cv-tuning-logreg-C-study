import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression


#phase 2: 10 real datasets
# 4 baseline datasets (moderate properties): breast_cancer, wine, iris, digits
#6 targeted datasets for extremes of the 3 structural propeties
#colon_cancer has high feature/sample ratio (62 samples, 2000 features)
#diabetes_pima has low feature/sample ration (768 samples, 8 features)
#credit_fraud has a high class imbalance (0.17% for the minority class)
#banknote_auth has a low class imbalance
#abalone has a high feature correlation (sea snail physical measurement dataset)
#features are shell length, diameter, height and weight measurements
#all about the same underlying thing (animal size) so they are
#naturally highly correlated (bigger shell is heavier, wider and taller too)
#madelone has a low feature correlation (artificially engineered dataset)

#20 random seeds are tested for every level
#the same 20 seeds are used across all properties so that
#any differences in gain between properties are not due to
#which random splits happened to be used

SEEDS=list(range(20))
C_VALUES=[0.001, 0.01, 0.1, 1, 10, 100]

#takes dataset (x,y) as input
#computes 5 numbers that describe its structural properties
#populates the properties table of the output
#which tells n_samples,n_features, ratio of datasets
def get_dataset_properties(X, y):
    #finidng the number of samples and features
    #and the feature/sample ratio
    n_samples,n_features= X.shape
    classes,counts = np.unique(y, return_counts=True)
    feature_sample_ratio= n_features/n_samples

    #finding the class imbalance
    imbalance_ratio=counts.min()/counts.max()

    #remove cst features first (to not divide by 0)
    non_constant=np.std(X,axis=0) > 0
    X_filtered=X[:, non_constant]
    #computing full correlation matrix
    #upper triangle only to not double count
    #average of the absolute values
    #result is one number summarizing how correlated
    #the features are with each other on average
    if X_filtered.shape[1]>=2:
        corr_matrix=np.corrcoef(X_filtered.T)
        upper=corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        mean_correlation=np.mean(np.abs(upper))
    else:
        mean_correlation=np.nan

    #this adds one row in properties table (one per dataset)
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_sample_ratio": round(float(feature_sample_ratio),4),
        "class_imbalance_ratio": round(float(imbalance_ratio),4),
        "mean_feature_correlation": round(float(mean_correlation),4),
    }

def run_one_seed(X, y, seed):
    #stratify=y to keep class proporitions in train and test set
    #matters for credit_fraud esp. (minority class = 0.17%)
    #without stratification, random splits could contain zero fraud in test set
    try:
        X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y )
    except ValueError: #no stratification in case of failure
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    #fitting scaler on train only, but applying to both train and test
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    #default for train is C=1.0
    default_model=LogisticRegression(C=1.0, random_state=seed, max_iter=2000)
    default_model.fit(X_train_scaled, y_train)
    default_acc =default_model.score(X_test_scaled, y_test)

    #trying all 6 c values with 5-fold cross validation
    best_C,best_cv = None,-1
    for C in C_VALUES:
        model=LogisticRegression(C=C,random_state=seed, max_iter=2000)
        try:
            cv_scores= cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', error_score='raise')
            if cv_scores.mean() > best_cv:
                best_cv=cv_scores.mean()
                best_C=C
        #in case cross-validation fails, skip C value instead of causing a crash
        except Exception:
            continue

    #if every C value fails, go back to default C=1.0
    if best_C is None:
        best_C = 1.0

    #and we finally retrain all the training set with best C
    #evaluate on test set
    tuned_model=LogisticRegression(C=best_C, random_state=seed, max_iter=2000)
    tuned_model.fit(X_train_scaled, y_train)
    tuned_acc= tuned_model.score(X_test_scaled, y_test)

    #returns default accuracy, tuned accuracy, gain, and best C value
    return default_acc,tuned_acc, tuned_acc - default_acc , best_C

#calls run_one_seed 20 times (20 different seeds)
#collections results and average thems
#also takes as input dataset_type (baseline vs targeted)
#and property_name
#to label in the results table
def run_averaged(X, y, name, dataset_type, property_name):
    print(f"  Running {name} ({len(SEEDS)} seeds)...", end=" ", flush=True)
    default_accs,tuned_accs, gains, best_Cs = [], [], [], []

    #trying to catch in case a seed fails
    #(for example if we get an empty class in some fold, moves to the next seed instead of crashing)
    #results are average over the seeds that did not cause issues (n_seeds in the return)
    for seed in SEEDS:
        try:
            d,t,g,c = run_one_seed(X, y, seed)
            default_accs.append(d)
            tuned_accs.append(t)
            gains.append(g)
            best_Cs.append(c)
        except Exception as e:
            print(f"\n Warning: seed {seed} failed ({e}), skipping.")

    #if every single seed fails and best_Cs is empty, we return none for modal_C
    modal_C= max(set(best_Cs), key=best_Cs.count) if best_Cs else None

    #n_seeds:how many seeds actually completed successfully
    return {
        "dataset":      name,
        "type":         dataset_type,
        "property":     property_name,
        "n_seeds":      len(gains),
        "mean_default": round(np.mean(default_accs), 4),
        "mean_tuned":   round(np.mean(tuned_accs), 4),
        "mean_gain":    round(np.mean(gains), 4),
        "std_gain":     round(np.std(gains), 4),
        "gain_pct":     f"{np.mean(gains)*100:+.2f}%",
        "std_pct": f"+/-{np.std(gains)*100:.2f}%",
        "modal_best_C": modal_C}

#downloads a dataset from OpenML and converts it into a clean numpy array
def load_and_clean(openml_id, name, max_samples=None):
    print(f"  Loading {name} ", end=" ", flush=True)
    try:
        #load as a pandas DataFrame so we can inspect column types
        try:
            dataset= fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
            X_df= dataset.data #features
            y=dataset.target #labels
            #go through every column (feature) in the dataframe
            for col in X_df.columns:
                #check if the column contains strings or non-numeric categories
                #to encode as integers instead and then convert to an array
                if X_df[col].dtype == object or str(X_df[col].dtype) == 'category':
                    X_df[col]= LabelEncoder().fit_transform(X_df[col].astype(str))
            X=X_df.values.astype(float)

            #if as_frame=True fails, start directly as ann array
        except Exception:
            dataset= fetch_openml(data_id=openml_id, as_frame=False, parser="auto")
            #loadingfeatures and labels directly as arrays
            X=dataset.data
            y=dataset.target

            #convert sparse matrix to full matrix
            if hasattr(X, "toarray"):
                X=X.toarray()
            #ensure X is a numpy float array
            X=np.array(X, dtype=float)

        #droppping empty columns (all values are NaN)
        #also dropping rows and their corresponding labels when the rows have
        #at least one NaN value (ie incomplete samples)
        col_mask= ~np.all(np.isnan(X), axis=0)
        X= X[:, col_mask]
        row_mask= ~np.any(np.isnan(X), axis=1)
        X= X[row_mask]
        y= np.array(y)[row_mask]

        #encodingthe labels as clean integers starting from 0
        le= LabelEncoder()
        y= le.fit_transform(y.astype(str))

        #if a cap  was specified and the dataset is larger, need to subsample
        if max_samples is not None and len(y) > max_samples:
            rng= np.random.default_rng(42) #to ensure the same rows are selected every run
            idx=rng.choice(len(y), size=max_samples, replace=False)
            X,y= X[idx], y[idx]
        print(f"loaded. Shape: {X.shape}")
        return X, y

    except Exception as e:
        print(f"FAILED: {e}")
        return None, None



print("\n--- Loading baseline datasets ---")
datasets={}

##our baseline is 4 skt-learn datasets with moderate prpoperties
d=load_breast_cancer()
datasets["breast_cancer"]= (d.data,d.target, "baseline", "baseline")
d=load_wine()
datasets["wine"]=(d.data,d.target, "baseline", "baseline")
d=load_iris()
datasets["iris"] =(d.data,d.target, "baseline", "baseline")
d=load_digits()
datasets["digits"]= (d.data, d.target, "baseline", "baseline")

print("\n Loading targeted datasets")

#6 targeted datasets selected for extreme property values
X,y = load_and_clean(1432, "colon_cancer")
if X is not None: datasets["colon_cancer"] = (X,y, "targeted", "high_feature_ratio")
X,y = load_and_clean(37, "diabetes_pima")
if X is not None: datasets["diabetes_pima"] = (X,y, "targeted", "low_feature_ratio")
X,y = load_and_clean(1597, "credit_fraud", max_samples=50000) #subsampling
if X is not None: datasets["credit_fraud"] = (X,y, "targeted", "high_imbalance")
X,y = load_and_clean(1462, "banknote_auth")
if X is not None: datasets["banknote_auth"] = (X, y, "targeted", "low_imbalance")

#abalone is handled seperately, because the labels are put into binar
#to convert the 28 ring counts into 2 categories, we split at the median.
# everything above it is old (=1) and = or below it is young (=0)
try:
    dataset= fetch_openml(data_id=183, as_frame=True, parser="auto")
    X_df= dataset.data
    y_raw=dataset.target #unbanireized
    #encoding the sex column as integer
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype) == 'category':
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
    X = X_df.values.astype(float)
    #converting ring count strings ot their equivalent float
    rings= y_raw.astype(float).values
    #median ring count across all
    median_rings= np.median(rings)

    #binarizing
    y= (rings > median_rings).astype(int)
    datasets["abalone"] = (X, y, "targeted", "high_correlation")

#if anything above fails, print the error instead of crashing
except Exception as e:
    print(f"FAILED: {e}")

X,y=load_and_clean(1485, "madelon")
if X is not None: datasets["madelon"] = (X,y,"targeted", "low_correlation")

print("\n--- Running pipeline ---")

results=[] #one dictionary/dataset with accuracy and gain numbers
properties=[] #one dictionary/dataset with structural properties


#we loop through all 10 datasets one by one
#datasets is a dictionary with
#key = dataset name(breast cancer for example)
#val = (X,y,dataset_type,property_name)
for name, (X, y, dataset_type, property_name) in datasets.items():
    #computing the properties
    props=get_dataset_properties(X, y)

    #adding identifying information to the properties dictionary
    #so we know which dataset and type this row belongs to in the final table
    props["dataset"]=name
    props["type"]=dataset_type
    props["property"]=property_name
    properties.append(props)

    # full experiment: 20 seeds, default vs tuned, compute gain
    #returns one dictionary summarizing the averaged results
    result= run_averaged(X,y, name,dataset_type,property_name)
    results.append(result)


#######bulding the tables

#convert the list of 10 properties dictionaries into a pandas table
#each dictionary becomes one row, keys become column names
props_df= pd.DataFrame(properties)[["type", "property", "dataset", "n_samples", "n_features", "feature_sample_ratio", "class_imbalance_ratio", "mean_feature_correlation"]]

results_df= pd.DataFrame(results)[["type", "property", "dataset", "mean_default", "mean_tuned", "mean_gain", "std_gain", "gain_pct", "std_pct", "modal_best_C" ]]

#combining both tables into 1
#by matching rows that share the same dataset names,types and properties
#this results in one big table with both structural properties and results side by side
full_df=pd.merge(props_df, results_df, on=["dataset", "type", "property"])

#save the full table to a CSV file so it can be opened in Excel
#index=False means don't include the row numbers in the file
full_df.to_csv("phase2_real_datasets.csv", index=False)
print("\nSaved to phase2_real_datasets.csv")