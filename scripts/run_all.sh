#!/bin/bash

subfolders=($(find ../data -mindepth 1 -maxdepth 1 -type d))

for folder in "${subfolders[@]}"; do
    app_name=$(basename "$folder")
    echo "Processing: $folder"
    echo "Feature modules:"
    python3 feature_models.py \
        --app_name "$app_name" \
        --input_dir "$folder" \
        --features_file "../data/features.json"
    python3 plot_perforamnce.py \
        --file_name "../results/${app_name}_performance.csv"
    echo "shapelets modules:"

    python3 shapelets_models.py \
       --app_name "$app_name" \
       --input_dir "$folder"
    python3 cnn_lstm_model.py \
       --app_name "$app_name" \
       --data_dir_dir "$folder" 
    python3 plot_perforamnce.py \
       --file_name "../results/${app_name}_performance.csv"
    
done

python3 agg_f1_scores.py
