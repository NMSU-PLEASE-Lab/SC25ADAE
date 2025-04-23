import pandas as pd
import glob

# Define model columns
model_cols = ["FRF", "FDT", "FSVM", "SRF", "SDT", "SSVM", "CNN-LSTM"]

# Update path as needed (assumes CSV files)
file_paths = glob.glob("../results/*.csv")

# Concatenate all CSVs into a single DataFrame
all_data = []
for path in file_paths:
    df = pd.read_csv(path)
    if "Heartbeat Metric" in df.columns:
        df = df.drop(columns=["Heartbeat Metric"])
    all_data.append(df)

concat_df = pd.concat(all_data, ignore_index=True)


# Filter only model columns that exist
present_cols = [col for col in model_cols if col in concat_df.columns]
concat_df = concat_df[present_cols].dropna().astype(float)

# Prepare result storage
rows = []
for model in present_cols:
    scores = concat_df[model].tolist()
    mean_f1 = round(sum(scores) / len(scores), 3)
    top3_f1 = round(sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores)), 3)
    top5_f1 = round(sum(sorted(scores, reverse=True)[:5]) / min(5, len(scores)), 3)
    count_above_90 = sum(1 for s in scores if s > 0.90)
    fraction_above_90 = round((count_above_90 / 70), 3)
    rows.append({
        "Model": model,
        "Mean F1": mean_f1,
        "Top3 Mean F1": top3_f1,
        "Top5 Mean F1": top5_f1,
        "Count F1 > 90": count_above_90,
        "Fraction F1 > 0.9": fraction_above_90
    })

# Create final result table
result_df = pd.DataFrame(rows)
result_df = result_df.sort_values(by="Mean F1", ascending=False).reset_index(drop=True)

# Save result to CSV
result_df.to_csv("../results/model_f1_summary.csv", index=False)

# Print the result table
print(result_df)

# Print overall summary
overall_mean_f1 = result_df["Mean F1"].mean()
overall_top3_mean = result_df["Top3 Mean F1"].mean()
overall_top5_mean = result_df["Top5 Mean F1"].mean()

print("\nOverall Summary:")
print(f"Average Mean F1 across models: {overall_mean_f1:.4f}")
print(f"Average Top3 Mean F1 across models: {overall_top3_mean:.4f}")
print(f"Average Top5 Mean F1 across models: {overall_top5_mean:.4f}")

