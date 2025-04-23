# Add this block at the top with other imports
import warnings
warnings.filterwarnings("ignore")  # Suppress TSFRESH and model warnings

import os
import csv
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# === Command Line Arguments ===
parser = argparse.ArgumentParser(description="Extract features and train models on time series data")
parser.add_argument('--app_name', type=str, required=True, help='Name of the application')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input CSV files')
parser.add_argument('--features_file', type=str, required=True, help='Path to selected features JSON file')
args = parser.parse_args()

app_name = args.app_name
input_dir = args.input_dir
features_file = args.features_file

# === Identify CSV Pairs in Directory ===
all_files = os.listdir(input_dir)
csv_pairs = [
    (
        f.replace('_time_series_data.csv', ''),
        os.path.join(input_dir, f),
        os.path.join(input_dir, f.replace('_time_series_data.csv', '_labels.csv'))
    )
    for f in all_files if f.endswith('_time_series_data.csv')
]

# === Load Feature Configuration ===
with open(features_file, 'r') as file:
    selected_features = json.load(file)
print(f"Loaded {len(selected_features)} selected features")

# === Define Cross-Validation and Scorer ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

# Store F1 scores per app
results_by_app = {}

# === Helper: Evaluate and Save Model ===
def evaluate_and_save_model(model, name, features, labels, prefix):
    f1_scores = cross_val_score(model, features, labels, cv=cv, scoring=f1_scorer)

    model.fit(features, labels)
    filename = f"{prefix}_{name.lower().replace(' ', '_')}_model.pkl"
    path = "../results"
    filepath = os.path.join(path, filename)
    joblib.dump(model, filepath)

    if prefix not in results_by_app:
        results_by_app[prefix] = {"Heartbeat Metric": prefix, "FRF": None, "FDT": None, "FSVM": None, "SRF": None, "SDT": None, "SSVM": None, "CNN-LSTM": None}

    #key = name[:3].upper()
    if name == "FRF":
        results_by_app[prefix]["FRF"] = round(f1_scores.mean(), 3)
        #results_by_app[prefix]["RF"] = round(f1_scores.mean(), 3)
    elif name == "FDT":
        results_by_app[prefix]["FDT"] = round(f1_scores.mean(), 3)
        #results_by_app[prefix]["DT"] = round(f1_scores.mean(), 3)
    elif name == "FSVM":
        results_by_app[prefix]["FSVM"] = round(f1_scores.mean(), 3)
        #results_by_app[prefix]["SVM"] = round(f1_scores.mean(), 3)

# === Define App-Specific Label Mappings ===
label_mappings_by_app = {
    "miniAMR": {
        "Low Quality": 0,
        "Med Low Quality": 2,
        "Med High Quality": 3,
        "High Quality": 1
    },
    "LAMMPS": {
        "Low Quality": 0,
        "Med Low Quality": 2,
        "Med High Quality": 3,
        "High Quality": 1
    },
    "miniFE": {
        "Low Quality": 2,
        "Med Low Quality": 3,
        "Med High Quality": 1,
        "High Quality": 0
    },
    "CoMD": {
        "Low Quality": 0,
        "Med Low Quality": 2,
        "Med High Quality": 1,
        "High Quality": 3
    },
    "PENNANT": {
        "Low Quality": 3,
        "Med Low Quality": 1,
        "Med High Quality": 2,
        "High Quality": 0
    },
    "default": {
        "Low Quality": 0,
        "Med Low Quality": 2,
        "Med High Quality": 3,
        "High Quality": 1
    }
}

# === Iterate Through CSV Pairs ===
for prefix, data_file, label_file in csv_pairs:
    print(f"\nProcessing application: {prefix}")

    # Load Labels
    y = pd.read_csv(label_file, header=None)
    y.columns = ['label']
    label_mapping = label_mappings_by_app.get(prefix, label_mappings_by_app.get("default"))
    y['label'] = y['label'].map(label_mapping)
    numerical_labels = y['label'].to_numpy()

    unique_classes, counts = np.unique(numerical_labels, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} instances")

    # Determine the maximum number of columns in the CSV
    max_length = max(len(row) for row in csv.reader(open(data_file)))

    # Load time series into DataFrame
    df = pd.read_csv(data_file, header=None, names=[str(i) for i in range(max_length)])
    df["id"] = df.index

    # Melt the data into long format for TSFRESH compatibility
    df = df.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)

    # Fill missing values
    if df['value'].isnull().any():
        df['value'] = df['value'].fillna(0)

    # Feature Extraction
    start_time = time.time()
    X = extract_features(df, column_id="id", column_sort="time", 
                         default_fc_parameters=selected_features, 
                         impute_function=impute)
    print(f"Number of extracted features: {X.shape[1]}")
    print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")
    feature_file = prefix + "_features.csv"
    path = "../results"
    filepath = os.path.join(path, feature_file)
    X.to_csv(filepath, index=False)

    # Normalize Features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(X)

    # Train and Save Models
    evaluate_and_save_model(DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42), "FDT", features, numerical_labels, prefix)
    evaluate_and_save_model(RandomForestClassifier(n_estimators=100, random_state=42), "FRF", features, numerical_labels, prefix)
    evaluate_and_save_model(SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42), "FSVM", features, numerical_labels, prefix)

# Save final combined results table
#if results_by_app:
#    output_dir = os.path.join(input_dir, f"{app_name}_summary")
#    os.makedirs(output_dir, exist_ok=True)
#
#    combined_table_path = os.path.join(output_dir, f"{app_name}_combined_model_f1_scores.csv")
#    if os.path.exists(combined_table_path):
#        existing_df = pd.read_csv(combined_table_path)
#    else:
#        existing_df = pd.DataFrame(columns=["Heartbeat Metric", "FRF", "FDT", "FSV", "SRF", "SDT", "SSVM", "CNN-LSTM"])
#
#    new_df = pd.DataFrame(results_by_app.values())
#    final_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["Heartbeat Metric"])
#    final_df.to_csv(combined_table_path, index=False)
#    print(f"Saved combined model F1 scores to {combined_table_path}")

if not results_by_app:
    print("No results to update.")
    
summary_file = "../results/" + app_name + "_performance.csv"

if os.path.exists(summary_file):
    existing_df = pd.read_csv(summary_file)
else:
    existing_df = pd.DataFrame(columns=["Heartbeat Metric", "FRF", "FDT", "FSVM", "SRF", "SDT", "SSVM", "CNN-LSTM"])

new_df = pd.DataFrame(results_by_app.values())
updated_df = pd.concat([existing_df, new_df], ignore_index=True)
updated_df = updated_df.drop_duplicates(subset=["Heartbeat Metric"])

updated_df.to_csv(summary_file, index=False)
print(f"Updated summary saved to: {summary_file}")
