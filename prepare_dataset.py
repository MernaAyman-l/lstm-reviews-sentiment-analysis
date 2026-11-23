"""Create a small, real sample from the original Amazon Fine Food Reviews CSV.

1. Put the downloaded original file (Reviews.csv) in this folder.
2. Run: python -m pip install pandas
3. Run: python prepare_dataset.py
4. Upload the generated Reviews_sample.csv to the GitHub repository root.
"""
from pathlib import Path
import pandas as pd

SOURCE = Path("Reviews.csv")
OUTPUT = Path("Reviews_sample.csv")
SAMPLE_ROWS = 1000
RANDOM_STATE = 42

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Reviews.csv was not found. Download the original Amazon Fine Food "
            "Reviews dataset and place Reviews.csv beside this script."
        )

    # Read only the columns used by the notebook.
    df = pd.read_csv(SOURCE, usecols=["Text", "Score"])
    df = df.dropna(subset=["Text", "Score"]).copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score"])
    df = df[df["Score"].between(1, 5)]

    if df.empty:
        raise ValueError("No valid rows with Text and Score were found.")

    # Keep the score distribution approximately representative.
    n = min(SAMPLE_ROWS, len(df))
    if len(df) > n:
        sample = (
            df.groupby("Score", group_keys=False)
              .apply(
                  lambda group: group.sample(
                      n=max(1, round(n * len(group) / len(df))),
                      random_state=RANDOM_STATE
                  )
              )
        )
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=RANDOM_STATE)
        elif len(sample) < n:
            remaining = df.drop(index=sample.index)
            extra = remaining.sample(n=n-len(sample), random_state=RANDOM_STATE)
            sample = pd.concat([sample, extra])
        df = sample.sample(frac=1, random_state=RANDOM_STATE)

    df.to_csv(OUTPUT, index=False)
    print(f"Created {OUTPUT} with {len(df):,} genuine reviews.")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.1f} KB")
    print("Upload this CSV to the root of your GitHub repository.")

if __name__ == "__main__":
    main()
