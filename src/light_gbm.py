import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from typing import TypeAlias

DATAFRAME: TypeAlias = pd.core.frame.DataFrame

def label_encoding(train_df: DATAFRAME, test_df: DATAFRAME) -> tuple[DATAFRAME, DATAFRAME]:
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    cat_cols = ["Division Name", "Department Name", "Class Name"]
    ordinal_encoder = OrdinalEncoder()
    combined_df[cat_cols] = ordinal_encoder.fit_transform(combined_df[cat_cols])

    train_length = len(train_df)
    train_df = combined_df.iloc[:train_length]
    test_df = combined_df.iloc[train_length:]
    test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df

def train(train_df: DATAFRAME, test_df: DATAFRAME) -> np.ndarray:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.1,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": 0.3,
        "lambda_l2": 0.3,
        "max_depth": 6,
        "num_leaves": 128,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "seed": 42,
    }

    features = ["Age", "Positive Feedback Count", "Division Name", "Department Name", "Class Name", "probability"]
    X = train_df[features].copy()
    y = train_df["labels"].copy()

    oof = np.zeros(X.shape[0])
    preds = np.zeros(test_df.shape[0])
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params, 
            train_data, 
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )

        oof[val_index] = model.predict(X_val)
        preds += model.predict(test_df[features]) / kf.n_splits

    # model.save_model("../models/lightgbm.txt", num_iteration=model.best_iteration)
    print(f"CV score: {roc_auc_score(y, oof)}")
    return preds

def main():
    train_df = pd.read_csv("../data/processed/train.csv")
    test_df = pd.read_csv("../data/processed/test.csv")
    submission_df = pd.read_csv("../data/raw/sample_submission.csv")

    train_df, test_df = label_encoding(train_df, test_df)
    preds = train(train_df, test_df)
    submission_df["target"] = preds
    submission_df.to_csv("../data/processed/submission.csv", header=True, index=False)

if __name__ == "__main__":
    main()
