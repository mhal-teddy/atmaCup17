import numpy as np
import pandas as pd
from typing import TypeAlias

DATAFRAME: TypeAlias = pd.core.frame.DataFrame

def preprocess(df: DATAFRAME, clothing_master: DATAFRAME, is_train: bool = True) -> DATAFRAME:
    processed_df = pd.merge(df, clothing_master, on="Clothing ID", how="left")
    processed_df["text"] = "title: " + processed_df["Title"].fillna("none") + " [SEP] " + "review: " + processed_df["Review Text"].fillna("none")
    processed_df.drop(["Title", "Review Text"], axis=1, inplace=True)
    if is_train:
        processed_df["labels"] = processed_df["Recommended IND"].astype(np.int8)
        processed_df.drop(["Rating", "Recommended IND"], axis=1, inplace=True)
    return processed_df

if __name__ == "__main__":
    train_df = pd.read_csv("../data/raw/train.csv")
    test_df = pd.read_csv("../data/raw/test.csv")
    clothing_master_df = pd.read_csv("../data/raw/clothing_master.csv")

    train_df = preprocess(train_df, clothing_master_df, is_train=True)
    test_df = preprocess(test_df, clothing_master_df, is_train=False)
    train_df.to_csv("../data/processed/train.csv", header=True, index=False)
    test_df.to_csv("../data/processed/test.csv", header=True, index=False)
