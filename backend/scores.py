# backend/scores.py

import os
import pandas as pd

# Base score per file type
FILE_TYPE_SCORES = {
    "csv": 10,
    "xlsx": 5,
    "xls": 5
}

def get_score_for_file(filename, file_path=None, ai_applied=False):
    """
    Returns score based on:
    - File extension
    - Dataset size (1 point per 100 rows)
    - AI usage (+5 bonus)
    """

    # Base score from extension
    ext = filename.rsplit('.', 1)[1].lower()
    score = FILE_TYPE_SCORES.get(ext, 0)

    # Bonus for dataset size
    if file_path and os.path.exists(file_path):
        try:
            if ext == "csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            row_count = len(df)
            score += row_count // 100  # 1 point for every 100 rows
        except Exception as e:
            print(f"Error reading file for scoring: {e}")

    # Bonus if AI was applied
    if ai_applied:
        score += 5

    return score
