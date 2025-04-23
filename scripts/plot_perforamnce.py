"""
Model Performance Visualization Tool

This script reads a CSV file containing F1 score performance metrics of various ML models
for different heartbeat features, sorts the data by CNN-LSTM, Shapelet-SVM, and Shapelet-DT
scores, and produces a grouped bar chart to visualize the model comparisons.

Usage:
    python model_performance_plot.py <app_name>__performance.csv

Example:
    python model_performance_plot.py miniFE_performance.csv

The generated plot is saved as <app_name>_model_performance.png.
"""

import csv
import os
import re
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def read_csv_to_list_of_lists(file_path: str) -> Tuple[np.ndarray, List[str], str]:
    """
    Reads a CSV file and parses it into a data matrix, list of heartbeat metrics,
    and a variable name inferred from the filename.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Tuple containing:
        - np.ndarray: Data matrix with shape (n_metrics, n_methods)
        - List[str]: List of heartbeat metric labels (1st column values)
        - str: Variable name inferred from the filename (prefix before first underscore)
    """
    data: List[List[float]] = []
    heartbeat_metrics: List[str] = []

    # Extract variable name from filename (before first underscore)
    base_name = os.path.basename(file_path)
    match = re.split(r"_", base_name)
    app_name = match[0] if match else "unknown"

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            heartbeat_metrics.append(row[0])
            data.append([float(value) for value in row[1:]])

    return np.array(data), heartbeat_metrics, app_name


def main():
    parser = argparse.ArgumentParser(description="Visualize model performance from CSV.")
    parser.add_argument("file_path", type=str, help="Path to the performance CSV file.")
    args = parser.parse_args()

    data, labels, app_name = read_csv_to_list_of_lists(args.file_path)

    methods = [
        'Feature-RF', 'Feature-DT', 'Feature-SVM',
        'Shapelet-RF', 'Shapelet-DT', 'Shapelet-SVM', 'CNN-LSTM'
    ]
    colors = ['blue', 'red', 'black', 'green', 'orange', 'grey', 'darkkhaki']

    cnn_lstm = data[:, 6]
    shapelet_svm = data[:, 5]
    shapelet_dt = data[:, 4]

    sorted_indices = np.lexsort((-shapelet_dt, -shapelet_svm, -cnn_lstm))

    data = data[sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    x = np.arange(len(labels))
    width = 0.11

    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(len(methods)):
        ax.bar(x + i * width, data[:, i], width, label=methods[i], color=colors[i])

    ax.set_ylabel('F1 Score')
    ax.set_xticks(x + width * 3)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='lower left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("../results/" + app_name + "_model_performance_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

