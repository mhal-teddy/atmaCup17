from datasets import Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments
import yaml

from preprocessing import preprocess

def compute_metrics(p):
    preds, labels = p
    preds = torch.softmax(torch.tensor(preds), dim=1).numpy()
    score = roc_auc_score(labels, preds[:, 1])
    return {"auc": score}

def fine_tune():
    with open("config.yml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv("../data/raw/train.csv")
    clothing_master_df = pd.read_csv("../data/raw/clothing_master.csv")
    train_df = preprocess(train_df, clothing_master_df, is_train=True)

    dataset = Dataset.from_pandas(train_df[["text", "labels"]]).class_encode_column("labels")
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=config["seed"], stratify_by_column="labels")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], clean_up_tokenization_spaces=False)

    def tokenize(examples):
        return tokenizer(examples["text"], max_length=config["max_length"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize, batched=True)

    # サンプル数を制限
    train_dataset = tokenized_datasets["train"].shuffle(seed=config["seed"]).select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=config["seed"]).select(range(1000))

    training_args = TrainingArguments(
        output_dir="../models",
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        do_eval=True,
        eval_strategy="epoch",
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=config["seed"],
    )

    training_config = AutoConfig.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], config=training_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    fine_tune()