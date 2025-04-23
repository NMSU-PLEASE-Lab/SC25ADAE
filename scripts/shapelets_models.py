"""
Shapelet-Based Time Series Classification Pipeline
--------------------------------------------------
This script reads multivariate time series datasets and corresponding labels,
extracts discriminative shapelets using the LearningShapelets method from tslearn,
transforms the time series using learned shapelets, and trains three machine learning models:
Random Forest, SVM, and Decision Tree using 5-fold cross-validation.

Expected File Pairs:
    - *_time_series_data.csv
    - *_labels.csv

Arguments:
    --input_dir       : Directory containing the dataset CSV files
    --l               : Shapelet length proportion (float)
    --r               : Number of shapelets per class (int)

Outputs:
    - *_shape_rf.pkl  : Random Forest model
    - *_shape_svm.pkl : SVM model
    - *_shape_dt.pkl  : Decision Tree model

Prints:
    - Feature transformation output
    - Accuracy and F1-score for each model
    - Model size and training time
"""

import os
import csv
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict, LearningShapelets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# === Command Line Arguments ===
parser = argparse.ArgumentParser(description="Shapelet-based time series classification")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing dataset CSV files')
parser.add_argument('--app_name', type=str, required=True, help='Application name for saving results')
parser.add_argument('--l', type=float, default=0.02, help='Shapelet length proportion')
parser.add_argument('--r', type=int, default=6, help='Number of shapelets per class')
args = parser.parse_args()

input_dir = args.input_dir
shapelet_l = args.l
shapelet_r = args.r
app_name = args.app_name

summary_file = "../results/" + app_name + "_performance.csv"

# === Define App-Specific Label Mappings ===
label_mappings_by_app = {
    "miniAMR": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1},
    "LAMMPS": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1},
    "miniFE": {"Low Quality": 2, "Med Low Quality": 3, "Med High Quality": 1, "High Quality": 0},
    "CoMD": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 1, "High Quality": 3},
    "PENNANT": {"Low Quality": 3, "Med Low Quality": 1, "Med High Quality": 2, "High Quality": 0},
    "default": {"Low Quality": 0,"Med Low Quality": 2,"Med High Quality": 3,"High Quality": 1}
}

label_mapping = label_mappings_by_app.get(app_name, label_mappings_by_app["default"])
# === Discover CSV Pairs ===
all_files = os.listdir(input_dir)
csv_pairs = [
    (
        f.replace('_time_series_data.csv', ''),
        os.path.join(input_dir, f),
        os.path.join(input_dir, f.replace('_time_series_data.csv', '_labels.csv'))
    )
    for f in all_files if f.endswith('_time_series_data.csv')
]

# Store F1 scores per app
results_by_app = {}
# === Define Cross-Validation and Scorer ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

# === Helper: Evaluate and Save Model ===
def evaluate_and_save_model(model, name, features, labels, prefix):
    f1_scores = cross_val_score(model, features, labels, cv=cv, scoring=f1_scorer)

    model.fit(features, labels)
    filename = f"{prefix}_{name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, filename)

    if prefix not in results_by_app:
        results_by_app[prefix] = {"Heartbeat Metric": prefix, "FRF": None, "FDT": None, "FSVM": None, "SRF": None, "SDT": None, "SSVM": None, "CNN-LSTM": None}

    df = pd.read_csv(summary_file) 
    #df.loc[df["Heartbeat Metric"] == prefix, name] = round(f1_scores.mean(), 3)
    #key = name[:3].upper()
    if name == "SRF":
        df.loc[df["Heartbeat Metric"] == prefix, "SRF"] = round(f1_scores.mean(), 3)
        #results_by_app[prefix]["SRF"] = round(f1_scores.mean(), 3)
        print(df)
        df.to_csv(summary_file,index=False)
        #results_by_app[prefix]["RF"] = round(f1_scores.mean(), 3)
    elif name == "SDT":
        df.loc[df["Heartbeat Metric"] == prefix, "SDT"] = round(f1_scores.mean(), 3)
        results_by_app[prefix]["SDT"] = round(f1_scores.mean(), 3)
        df.to_csv(summary_file,index=False)
        #results_by_app[prefix]["DT"] = round(f1_scores.mean(), 3)
    elif name == "SSVM":
        df.loc[df["Heartbeat Metric"] == prefix, "SSVM"] = round(f1_scores.mean(), 3)
        results_by_app[prefix]["SSVM"] = round(f1_scores.mean(), 3)
        #results_by_app[prefix]["SVM"] = round(f1_scores.mean(), 3)
        df.to_csv(summary_file,index=False)

#results_by_app = {}

# === Helper Function ===
#def train_evaluate_model(model, name, X, y, prefix):
#    print(f"\nTraining {name} with 5-fold cross-validation...")
#    start_time = time.time()
#
#    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
#    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
#
#    training_time = time.time() - start_time
#    model.fit(X, y)
#
#    model_path = f"{prefix}_shape_{name.lower()}.pkl"
#    joblib.dump(model, model_path)
#    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
#
#    print(f"Training time: {training_time:.2f} seconds")
#    print(f"Model size: {model_size_mb:.2f} MB")
#    print(f"Cross-validation Accuracy: {acc_scores.mean():.2f} ± {acc_scores.std():.2f}")
#    print(f"Cross-validation Weighted F1-score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# === Process Each File Pair ===
for prefix, file_path, label_path in csv_pairs:
    if prefix == "LAMMPS_hbDur1" or prefix == "LAMMPS_hbC1":
        break
    print(f"\n=== Processing {prefix} ===")

    # Load and Prepare Time Series Data
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        max_length = max(len(row) for row in reader)

    colnames = [str(i) for i in range(max_length)]
    df = pd.read_csv(file_path, header=None, names=colnames).fillna(0)

    # Load and Map Labels
    y = pd.read_csv(label_path, header=None)
    y.columns = ['label']
    y['label'] = y['label'].map(label_mapping)

    # Normalize and Split Data
    X_train_shapelets, _, y_train_shapelets, _ = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
    X_train_shapelets = MinMaxScaler().fit_transform(X_train_shapelets)
    df_scaled = MinMaxScaler().fit_transform(df)

    # Learn Shapelets
    n_ts, ts_sz = X_train_shapelets.shape
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=4, l=shapelet_l, r=shapelet_r)

    shapelet_model = LearningShapelets(
        n_shapelets_per_size=shapelet_sizes,
        optimizer="sgd",
        weight_regularizer=0.01,
        max_iter=15,
        verbose=1,
        random_state=42,
    )
    shapelet_model.fit(X_train_shapelets, y_train_shapelets)

    # Transform Full Data
    data_transformed = shapelet_model.transform(df_scaled)

    # Train Models
    evaluate_and_save_model(RandomForestClassifier(n_estimators=100, random_state=42), "SRF", data_transformed, y, prefix)
    #train_evaluate_model(RandomForestClassifier(n_estimators=100, random_state=42), "rf", data_transformed, y, prefix)
    evaluate_and_save_model(SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42), "SSVM", data_transformed, y, prefix)
    evaluate_and_save_model(DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42), "SDT", data_transformed, y, prefix)

if not results_by_app:
    print("No results to update.")
    
#summary_file = "../results/" + app_name + "_performance.csv"

#if os.path.exists(summary_file):
#    existing_df = pd.read_csv(summary_file)
#else:
#    existing_df = pd.DataFrame(columns=["Heartbeat Metric", "FRF", "FDT", "FSV", "SRF", "SDT", "SSVM", "CNN-LSTM"])
#
#new_df = pd.DataFrame(results_by_app.values())
#updated_df = pd.concat([existing_df, new_df], ignore_index=True)
#updated_df = updated_df.drop_duplicates(subset=["Heartbeat Metric"])
#
#updated_df.to_csv(summary_file, index=False)
#print(f"Updated summary saved to: {summary_file}")
