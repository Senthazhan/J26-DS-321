"""Train / validation / test split, done BY RUN (70 / 15 / 15)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def make_splits(runs: pd.DataFrame) -> pd.DataFrame:
    train_ids, temp_ids = train_test_split(
        runs["run_id"], test_size=0.30, random_state=config.SEED,
        stratify=runs["run_type"])
    temp = runs[runs["run_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp["run_id"], test_size=0.50, random_state=config.SEED,
        stratify=temp["run_type"])
    return pd.concat([
        pd.DataFrame({"run_id": train_ids, "split": "train"}),
        pd.DataFrame({"run_id": val_ids, "split": "val"}),
        pd.DataFrame({"run_id": test_ids, "split": "test"}),
    ])


def load_or_make_splits(runs: pd.DataFrame) -> pd.DataFrame:
    """Use the saved splits.csv if it exists, so every result uses the same split."""
    if config.SPLITS_CSV.exists():
        return pd.read_csv(config.SPLITS_CSV)
    split = make_splits(runs)
    split.to_csv(config.SPLITS_CSV, index=False)
    return split
