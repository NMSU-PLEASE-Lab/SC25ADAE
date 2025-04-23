# Import necessary libraries
import os
import argparse
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, MaxPool1D, Flatten, Dense,
                                     BatchNormalization, Input, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process time series and label files.')
parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing data files')
parser.add_argument('--app_name', type=str, required=True, help='Application name for saving results')
args = parser.parse_args()
data_dir = args.data_dir
app_name = args.app_name

# Define label mappings
label_mappings_by_app = {
    "miniAMR": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1},
    "LAMMPS": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1},
    "miniFE": {"Low Quality": 2, "Med Low Quality": 3, "Med High Quality": 1, "High Quality": 0},
    "CoMD": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 1, "High Quality": 3},
    "PENNANT": {"Low Quality": 3, "Med Low Quality": 1, "Med High Quality": 2, "High Quality": 0},
    "default": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1}
}
label_mapping = label_mappings_by_app.get(app_name, label_mappings_by_app["default"])

# Define model parameters
model_params_by_app = {
    "miniAMR": {"conv1_units": 192, "kernel_size": 7, "lstm_units": 64, "dropout_1": 0.4, "dense_units": 64, "dropout_2": 0.3},
    "LAMMPS": {"conv1_units": 192, "kernel_size": 5, "lstm_units": 64, "dropout_1": 0.3, "dense_units": 192, "dropout_2": 0.2},
    "miniFE": {"conv1_units": 128, "kernel_size": 5, "lstm_units": 64, "dropout_1": 0.2, "dense_units": 256, "dropout_2": 0.4},
    "CoMD": {"conv1_units": 128, "kernel_size": 7, "lstm_units": 128, "dropout_1": 0.2, "dense_units": 256, "dropout_2": 0.4},
    "PENNANT": {"conv1_units": 192, "kernel_size": 7, "lstm_units": 64, "dropout_1": 0.2, "dense_units": 256, "dropout_2": 0.2},
    "default": {"conv1_units": 128, "kernel_size": 7, "lstm_units": 128, "dropout_1": 0.2, "dense_units": 256, "dropout_2": 0.4}
}
model_params = model_params_by_app.get(app_name, model_params_by_app["default"])

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

all_files = os.listdir(data_dir)
csv_pairs = [(f.replace('_time_series_data.csv', ''),
              os.path.join(data_dir, f),
              os.path.join(data_dir, f.replace('_time_series_data.csv', '_labels.csv')))
             for f in all_files if f.endswith('_time_series_data.csv')]

for file_name, file_path, label_path in csv_pairs:
    print(f"Processing: {file_name}")

    max_length = 0
    count = 0
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            max_length = max(max_length, len(row))
            count += 1
    print("Maximum length of rows:", max_length)
    print(count)

    colnames = [str(i) for i in range(max_length)]
    df = pd.read_table(file_path, header=None, sep=',', names=colnames).fillna(0)
    print(df.shape)

    y = pd.read_csv(label_path, header=None)
    y.columns = ['label']
    y['label'] = y['label'].map(label_mapping)

    label_names = {v: k for k, v in label_mapping.items()}
    value_counts = y['label'].value_counts().rename(label_names)
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=value_counts.index, y=value_counts.values)
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{value_counts[i]}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)
    plt.title('Number of samples per class in train data')
    plt.xlabel('Classes')
    plt.ylabel('Number of samples')
    plt.xticks(rotation=90)
    plt.savefig(f"{file_name}_class_distribution.png")
    plt.show()

    y = y.values.ravel()
    x_train, x_val, y_train, y_val = train_test_split(df, y, test_size=0.3,
                                                      stratify=y, random_state=42)
    x_train = x_train.values.reshape(x_train.shape[0], -1, 1)
    x_val = x_val.values.reshape(x_val.shape[0], -1, 1)

    scaler = MinMaxScaler()
    X_train_2d = x_train.reshape(-1, x_train.shape[1])
    X_val_2d = x_val.reshape(-1, x_val.shape[1])
    X_train_normalized_2d = scaler.fit_transform(X_train_2d)
    X_val_normalized_2d = scaler.fit_transform(X_val_2d)
    X_train_normalized = X_train_normalized_2d.reshape(x_train.shape)
    X_val_normalized = X_val_normalized_2d.reshape(x_val.shape)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)

    model_cnn_lstm = Sequential([
        Input(shape=(X_train_normalized.shape[1:])),
        Conv1D(model_params["conv1_units"], kernel_size=model_params["kernel_size"], activation='relu'),
        MaxPool1D(pool_size=2, strides=2, padding="same"),
        BatchNormalization(),
        LSTM(model_params["lstm_units"], return_sequences=True, activation="tanh"),
        Dropout(model_params["dropout_1"]),
        Flatten(),
        Dense(model_params["dense_units"], activation='relu'),
        Dropout(model_params["dropout_2"]),
        Dense(4, activation='softmax')
    ])

    model_cnn_lstm.summary()
    model_cnn_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8),
        ReduceLROnPlateau(patience=20, monitor='val_loss', min_lr=1e-6, cool_down=20),
        ModelCheckpoint(filepath=f"../results/{file_name}_cnn_lstm_model.keras", monitor='val_loss', save_best_only=True)
    ]

    history = model_cnn_lstm.fit(X_train_normalized, y_train, epochs=30, callbacks=callbacks,
                                 batch_size=32, validation_split=0.3, verbose=1)

    train_score = model_cnn_lstm.evaluate(X_train_normalized, y_train)
    validation_score = model_cnn_lstm.evaluate(X_val_normalized, y_val)
    print('Accuracy Train data: ', train_score[1])
    print('Accuracy Validation data: ', validation_score[1])

    y_pred = model_cnn_lstm.predict(X_val_normalized)
    y_test_labels = np.argmax(y_val, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    report = classification_report(y_test_labels, y_pred_labels)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

    with open("{file_name}_model_performance.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"F1 Score (macro): {f1:.2f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    # --- Update or create summary file ---
    summary_file_path = "../results/" + app_name + "_performance.csv"
    if os.path.exists(summary_file_path):
        summary_df = pd.read_csv(summary_file_path)
    summary_df.loc[summary_df["Heartbeat Metric"] == file_name, "CNN-LSTM"] = round(f1, 2)
    
    summary_df.to_csv(summary_file_path, index=False)
    #else:
    #    summary_df = pd.DataFrame(columns=["Heartbeat Metric", "CNN-LSTM"])

    #if file_name in summary_df["Heartbeat Metric"].values:
    #    summary_df.loc[summary_df["Heartbeat Metric"] == file_name, "CNN-LSTM"] = round(f1, 4)
    #else:
    #    new_row = pd.DataFrame([{"Heartbeat Metric": file_name, "CNN-LSTM": round(f1, 3)}])
    #    summary_df = pd.concat([summary_df, new_row], ignore_index=True)

    #summary_df.to_csv(summary_file_path, index=False)
    #print(f"Updated summary saved to: {summary_file_path}")

