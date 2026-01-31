import argparse
import os
import random

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


LABEL_NAMES = {0: "Bad", 1: "Neutral", 2: "Good"}


def map_labels(stars):
    if stars <= 2:
        return 0
    if stars == 3:
        return 1
    return 2


def load_german_test(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path, low_memory=False)

    if "language" in df.columns:
        df = df[df["language"] == "de"]

    if "stars" not in df.columns:
        raise ValueError("test.csv must contain a 'stars' column.")

    df = df.copy()
    df["label"] = df["stars"].apply(map_labels)

    keep_cols = ["review_title", "review_body", "label"]
    if "language" in df.columns:
        keep_cols.append("language")

    return Dataset.from_pandas(df[keep_cols])


def build_tokenizer(model_path: str, fallback_model: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback_model)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def tokenize_function(examples):
        texts = []
        for i in range(len(examples["review_title"])):
            title = examples["review_title"][i] or ""
            body = examples["review_body"][i] or ""
            texts.append(f"{title} {body}".strip())

        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    remove_cols = ["review_title", "review_body"]
    if "language" in dataset.column_names:
        remove_cols.append("language")

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }


def evaluate_checkpoint(name: str, model_path: str, tokenized_test: Dataset, batch_size: int):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    eval_dir = os.path.join(model_path, "eval_output")

    args = TrainingArguments(
        output_dir=eval_dir,
        per_device_eval_batch_size=batch_size,
        report_to=[],
        do_train=False,
        do_eval=True,
        logging_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    predictions = trainer.predict(tokenized_test).predictions
    pred_labels = np.argmax(predictions, axis=1)

    print("\n" + "=" * 60)
    print(f"{name} checkpoint evaluation")
    print("=" * 60)
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Precision: {metrics['eval_precision']:.4f}")
    print(f"Recall: {metrics['eval_recall']:.4f}")
    print(f"F1: {metrics['eval_f1']:.4f}")

    return pred_labels


def print_examples(raw_dataset: Dataset, pred_labels: np.ndarray, num_examples: int, seed: int):
    random.seed(seed)
    indices = list(range(len(raw_dataset)))
    random.shuffle(indices)
    indices = indices[:num_examples]

    print("\nExamples (German reviews):")
    for idx in indices:
        row = raw_dataset[idx]
        title = row.get("review_title", "") or ""
        body = row.get("review_body", "") or ""
        text = f"{title} {body}".strip()
        true_label = LABEL_NAMES.get(int(row["label"]), str(row["label"]))
        pred_label = LABEL_NAMES.get(int(pred_labels[idx]), str(pred_labels[idx]))
        print("-" * 60)
        print(f"Text: {text[:400]}")
        print(f"True: {true_label} | Pred: {pred_label}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate single and multilingual checkpoints on German test reviews.")
    parser.add_argument("--test_csv", default="test.csv", help="Path to test.csv")
    parser.add_argument("--single_ckpt", default="/data01/aiello/results_single_language/checkpoint-12500", help="Single-language checkpoint path")
    parser.add_argument("--multi_ckpt", default="/data01/aiello/results_multilingual/checkpoint-12500", help="Multilingual checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of example predictions to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for example selection")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv = args.test_csv if os.path.isabs(args.test_csv) else os.path.join(base_dir, args.test_csv)
    single_ckpt = args.single_ckpt if os.path.isabs(args.single_ckpt) else os.path.join(base_dir, args.single_ckpt)
    multi_ckpt = args.multi_ckpt if os.path.isabs(args.multi_ckpt) else os.path.join(base_dir, args.multi_ckpt)

    raw_test = load_german_test(test_csv)
    if len(raw_test) == 0:
        raise ValueError("No German reviews found in test.csv.")

    fallback_model = "distilbert-base-multilingual-cased"
    tokenizer = build_tokenizer(multi_ckpt, fallback_model)
    tokenized_test = tokenize_dataset(raw_test, tokenizer)

    print(f"Loaded {len(raw_test)} German test reviews from {test_csv}")

    single_preds = evaluate_checkpoint("Single-language", single_ckpt, tokenized_test, args.batch_size)
    multi_preds = evaluate_checkpoint("Multilingual", multi_ckpt, tokenized_test, args.batch_size)

    print("\n" + "=" * 60)
    print("Single-language example predictions")
    print("=" * 60)
    print_examples(raw_test, single_preds, args.num_examples, args.seed)

    print("\n" + "=" * 60)
    print("Multilingual example predictions")
    print("=" * 60)
    print_examples(raw_test, multi_preds, args.num_examples, args.seed)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_DISABLED", "true")
    main()