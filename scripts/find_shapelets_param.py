"""
Hyperparameter Search for Shapelet Learning
-------------------------------------------
This script performs a grid search to find the best combination of shapelet size (l)
and number of shapelets per class (r) using the LearningShapelets algorithm with
an SVM classifier. 5-fold cross-validation is used for evaluation.

Outputs:
    - Best shapelet parameters (l and r)
    - Best cross-validation accuracy for each dataset
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict, LearningShapelets

# === Command Line Arguments ===
parser = argparse.ArgumentParser(description="Shapelet-based time series classification")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing dataset CSV files')
parser.add_argument('--app_name', type=str, required=True, help='Application name for saving results')
args = parser.parse_args()

input_dir = args.input_dir
app_name = args.app_name

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

# === Hyperparameter Search Settings ===
param_grid = {
    "n_shapelets_per_size": [3, 4, 5, 6],
    "shapelet_sizes": [0.01, 0.02, 0.03]
}

# === Process Each File Pair ===
for prefix, file_path, label_path in csv_pairs:
    print(f"\n=== Processing {prefix} ===")

    # Read time series data
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        max_length = max(len(row) for row in reader)

    colnames = [str(i) for i in range(max_length)]
    df = pd.read_csv(file_path, header=None, names=colnames).fillna(0)

    # Read and map labels
    y = pd.read_csv(label_path, header=None)
    y.columns = ['label']
    y['label'] = y['label'].map(label_mapping)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    # Use 20% of training data for shapelet learning
    X_train_shapelet, _, y_train_shapelet, _ = train_test_split(X_train, y_train, test_size=0.7, stratify=y_train, random_state=42)
    X_train_shapelet = MinMaxScaler().fit_transform(X_train_shapelet)

    n_ts, ts_sz = X_train_shapelet.shape
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_acc = 0
    best_params = {}

    for r in param_grid["n_shapelets_per_size"]:
        for l in param_grid["shapelet_sizes"]:
            shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=4, l=l, r=r)
            shapelet_model = LearningShapelets(
                n_shapelets_per_size=shapelet_sizes,
                optimizer="sgd",
                weight_regularizer=0.01,
                max_iter=10,
                verbose=0,
                random_state=42
            )

            shapelet_model.fit(X_train_shapelet, y_train_shapelet)
            X_train_transformed = shapelet_model.transform(X_train)
            X_test_transformed = shapelet_model.transform(X_test)

            svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
            scores = cross_val_score(svm_clf, X_train_transformed, y_train, cv=kf, scoring='accuracy')
            acc = scores.mean()

            print(f"l={l}, r={r}, Accuracy={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = {"n_shapelets": r, "shapelet_size": l}

    print("Best Parameters:", best_params)
    print("Best Accuracy:", best_acc)

