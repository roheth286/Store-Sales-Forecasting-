"""
Submission generation module for Kaggle output.
"""

import numpy as np
import pandas as pd


def generate_submission(test_df, output_path="data/processed/submission.csv"):
    """
    Extracts 'id' and 'sales' from test_df, clips negative sales to 0, and saves submission.csv.
    """
    submission = test_df[["id", "sales"]].copy()
    submission["sales"] = np.clip(submission["sales"], 0, None)
    submission.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print(f"🎉 Kaggle Submission CSV generated successfully!")
    print(f"Path: {output_path}")
    print(f"Shape: {submission.shape[0]:,} rows × {submission.shape[1]} columns")
    print(f"Any NaNs? {submission.isnull().sum().sum()}")
    print("Preview:")
    print(submission.head(5))
    print("=" * 50)

    return submission
