# This refactor adds hyperparameter tuning for the CNN-LSTM model only.
# Original model training is skipped in favor of Keras Tuner.

import os
import csv
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, LSTM, MaxPool1D, Flatten, Dense,
                                     BatchNormalization, Input, Dropout)
from tensorflow.keras.callbacks import EarlyStopping

# CLI Args
parser = argparse.ArgumentParser(description='Hyperparameter tuning for CNN-LSTM.')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--app_name', type=str, required=True)
args = parser.parse_args()
data_dir = args.data_dir
app_name = args.app_name

label_mappings_by_app = {
    "miniAMR": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1},
    "LAMMPS": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1},
    "miniFE": {"Low Quality": 2, "Med Low Quality": 3, "Med High Quality": 1, "High Quality": 0},
    "CoMD": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 1, "High Quality": 3},
    "PENNANT": {"Low Quality": 3, "Med Low Quality": 1, "Med High Quality": 2, "High Quality": 0},
    "default": {"Low Quality": 0, "Med Low Quality": 2, "Med High Quality": 3, "High Quality": 1}
}
label_mapping = label_mappings_by_app.get(app_name, label_mappings_by_app["default"])

def build_model(hp):
    model = Sequential([
        Input(shape=(input_shape)),
        Conv1D(
            filters=hp.Int('conv1_filters', min_value=64, max_value=256, step=64),
            kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]),
            activation='relu'
        ),
        MaxPool1D(pool_size=2, strides=2, padding="same"),
        BatchNormalization(),
        LSTM(
            units=hp.Int('lstm_units', min_value=64, max_value=256, step=64),
            return_sequences=True, activation='tanh'
        ),
        Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)),
        Flatten(),
        Dense(
            units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
            activation='relu'
        ),
        Dropout(hp.Float('dropout_dense', 0.2, 0.5, step=0.1)),
        Dense(4, activation='softmax')
    ])
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load file pair
all_files = os.listdir(data_dir)
csv_pairs = [(f.replace('_time_series_data.csv', ''),
              os.path.join(data_dir, f),
              os.path.join(data_dir, f.replace('_time_series_data.csv', '_labels.csv')))
             for f in all_files if f.endswith('_time_series_data.csv')]

for file_name, file_path, label_path in csv_pairs:
    # Determine max length and load CSV
    print(f"Processing metric prefix: {file_name}")
    max_length = max(len(row) for row in csv.reader(open(file_path)))
    colnames = [str(i) for i in range(max_length)]
    df = pd.read_csv(file_path, header=None, names=colnames).fillna(0)
    y = pd.read_csv(label_path, header=None)
    y.columns = ['label']
    y['label'] = y['label'].map(label_mapping)
    
    y = y.values.ravel()
    x_train, x_val, y_train, y_val = train_test_split(df, y, test_size=0.3, stratify=y, random_state=42)
    x_train = x_train.values.reshape(x_train.shape[0], -1, 1)
    x_val = x_val.values.reshape(x_val.shape[0], -1, 1)

    scaler = MinMaxScaler()
    X_train_2d = x_train.reshape(-1, x_train.shape[1])
    X_val_2d = x_val.reshape(-1, x_val.shape[1])
    X_train_normalized = scaler.fit_transform(X_train_2d).reshape(x_train.shape)
    X_val_normalized = scaler.transform(X_val_2d).reshape(x_val.shape)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)

    input_shape = X_train_normalized.shape[1:]

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory=app_name,
        project_name=f'{file_name}_cnn_lstm_tuning'
    )

    tuner.search(
        X_train_normalized, y_train,
        validation_data=(X_val_normalized, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
Best hyperparameters for {file_name}:
  conv1_filters: {best_hps.get('conv1_filters')}
  kernel_size: {best_hps.get('kernel_size')}
  lstm_units: {best_hps.get('lstm_units')}
  dropout: {best_hps.get('dropout')}
  dense_units: {best_hps.get('dense_units')}
  dropout_dense: {best_hps.get('dropout_dense')}
  optimizer: {best_hps.get('optimizer')}
""")

