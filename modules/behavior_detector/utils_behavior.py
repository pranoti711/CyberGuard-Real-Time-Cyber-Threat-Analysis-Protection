# Save as: modules/behavior_detector/utils_behavior.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_input(df, vectorizer):
    """
    Preprocess a DataFrame for model prediction:
    - Fills missing values
    - Drops label column if present
    - Converts each row to space-separated string
    - Transforms using the provided vectorizer
    """
    df = df.fillna("")  # Handle missing values

    if 'label' in df.columns:
        df = df.drop(columns=['label'])  # Drop label column if present

    # Convert all values in each row to a string and join them with spaces
    row_texts = df.astype(str).agg(' '.join, axis=1)

    # Use the fitted vectorizer to convert text to numerical features
    return vectorizer.transform(row_texts)


def split_engineered_features():
    """
    One-time dataset split: reads engineered CSV and creates
    train/valid/test feature files for training/validation/testing.
    """
    INPUT_FILE = "data/insider_behavior/engineered_features.csv"
    OUTPUT_DIR = "data/insider_behavior/features"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(f"{OUTPUT_DIR}/train_features.csv", index=False)
    valid_df.to_csv(f"{OUTPUT_DIR}/valid_features.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test_features.csv", index=False)

    print("✅ Split completed:")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test : {len(test_df)}")

