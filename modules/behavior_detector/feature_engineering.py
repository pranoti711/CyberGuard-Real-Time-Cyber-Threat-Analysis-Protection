import os
import pandas as pd

def engineer_features(input_dir="data/insider_behavior", output_file="data/engineered_features.csv"):
    all_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            try:
                df = pd.read_csv(file_path)

                # ✅ Ensure 'label' column exists and is filled with 0 if missing
                if 'label' not in df.columns:
                    df['label'] = 0
                else:
                    df['label'] = df['label'].fillna(0).astype(int)

                # ✅ Optional: add source file name as a feature
                df['source'] = filename.replace('.csv', '')

                all_data.append(df)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"✅ Engineered features saved to: {output_file}")
    else:
        print("❌ No CSV files processed. Please check your input directory.")

if __name__ == "__main__":
    print("🔍 Running feature engineering...")
    engineer_features()

# python -m modules.behavior_detector.feature_engineering
# .\venv\Scripts\activate.ps1