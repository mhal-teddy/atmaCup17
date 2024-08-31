import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import TypeAlias
import yaml

from preprocessing import preprocess

DATAFRAME: TypeAlias = pd.core.frame.DataFrame

with open("config.yml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbedDataset(Dataset):
    def __init__(self, texts: DATAFRAME):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], clean_up_tokenization_spaces=False)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        token = self.tokenizer(
            self.texts[idx],
            max_length=config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.squeeze(0).to(device) for key, value in token.items()}

def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

def predict_sentiment(df: DATAFRAME, model) -> DATAFRAME:
    dataset = EmbedDataset(df["text"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    probabilities_list = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            probabilities = softmax(logits)
            probabilities_list.append(probabilities[:, 1])

    probabilities_array = np.concatenate(probabilities_list, axis=0)
    df["probability"] = probabilities_array
    return df

def main():
    train_df = train_df = pd.read_csv("../data/raw/train.csv")
    test_df = pd.read_csv("../data/raw/test.csv")
    clothing_master_df = pd.read_csv("../data/raw/clothing_master.csv")

    train_df = preprocess(train_df, clothing_master_df, is_train=True)
    test_df = preprocess(test_df, clothing_master_df, is_train=False)

    model = AutoModelForSequenceClassification.from_pretrained("../models/checkpoint-375").to(device)
    model.eval()

    test_df = predict_sentiment(test_df, model)
    test_df.to_csv("../data/processed/test.csv", header=True, index=False)
    print("Test data embedded")

    train_df = predict_sentiment(train_df, model)
    train_df.to_csv("../data/processed/train.csv", header=True, index=False)
    print("Training data embedded")

if __name__ == "__main__":
    main()