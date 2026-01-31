import argparse
import os
import random

import matplotlib.pyplot as plt
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


def load_mixed_language_test(
    csv_path: str,
    exclude_language: str,
    sample_per_language: int | None,
    max_languages: int | None,
    seed: int,
) -> Dataset:
    df = pd.read_csv(csv_path, low_memory=False)

    if "language" not in df.columns:
        raise ValueError("test.csv must contain a 'language' column for mixed-language sampling.")

    if "stars" not in df.columns:
        raise ValueError("test.csv must contain a 'stars' column.")

    df = df[df["language"].notna()]
    if exclude_language:
        df = df[df["language"] != exclude_language]

    df = df.copy()
    df["label"] = df["stars"].apply(map_labels)

    languages = df["language"].unique().tolist()
    random.Random(seed).shuffle(languages)
    if max_languages:
        languages = languages[:max_languages]

    if sample_per_language:
        chunks = []
        for lang in languages:
            lang_df = df[df["language"] == lang]
            if len(lang_df) == 0:
                continue
            sample_size = min(sample_per_language, len(lang_df))
            chunks.append(lang_df.sample(n=sample_size, random_state=seed))
        if not chunks:
            raise ValueError("No reviews available after applying language filters.")
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = df[df["language"].isin(languages)]

    keep_cols = ["review_title", "review_body", "label", "language"]
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

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["review_title", "review_body", "language"],
    )
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

    print("\nExamples (mixed languages):")
    for idx in indices:
        row = raw_dataset[idx]
        title = row.get("review_title", "") or ""
        body = row.get("review_body", "") or ""
        text = f"{title} {body}".strip()
        lang = row.get("language", "unknown")
        true_label = LABEL_NAMES.get(int(row["label"]), str(row["label"]))
        pred_label = LABEL_NAMES.get(int(pred_labels[idx]), str(pred_labels[idx]))
        print("-" * 60)
        print(f"Language: {lang}")
        print(f"Text: {text[:400]}")
        print(f"True: {true_label} | Pred: {pred_label}")


def count_errors_by_language(raw_dataset: Dataset, pred_labels: np.ndarray):
    languages = raw_dataset["language"]
    labels = raw_dataset["label"]
    errors = {}
    totals = {}

    for lang, true_label, pred_label in zip(languages, labels, pred_labels, strict=False):
        totals[lang] = totals.get(lang, 0) + 1
        if int(true_label) != int(pred_label):
            errors[lang] = errors.get(lang, 0) + 1

    return errors, totals


def plot_errors_by_language(errors: dict, totals: dict, title: str, output_path: str):
    languages = sorted(totals.keys())
    error_counts = [errors.get(lang, 0) for lang in languages]

    plt.figure(figsize=(12, 5))
    plt.bar(languages, error_counts, color="#d9534f", alpha=0.85)
    plt.xlabel("Language")
    plt.ylabel("Errors")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single and multilingual checkpoints on mixed-language test reviews (non-German)."
    )
    parser.add_argument("--test_csv", default="test.csv", help="Path to test.csv")
    parser.add_argument("--single_ckpt", default="/data01/aiello/results_single_language/checkpoint-12500", help="Single-language checkpoint path")
    parser.add_argument("--multi_ckpt", default="/data01/aiello/results_multilingual/checkpoint-12500", help="Multilingual checkpoint path")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of example predictions to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and examples")
    parser.add_argument("--exclude_language", default="de", help="Language to exclude from sampling")
    parser.add_argument("--sample_per_language", type=int, default=200, help="Samples per language (set 0 to use all)")
    parser.add_argument("--max_languages", type=int, default=12, help="Max number of languages to sample (0 for all)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv = args.test_csv if os.path.isabs(args.test_csv) else os.path.join(base_dir, args.test_csv)
    single_ckpt = args.single_ckpt if os.path.isabs(args.single_ckpt) else os.path.join(base_dir, args.single_ckpt)
    multi_ckpt = args.multi_ckpt if os.path.isabs(args.multi_ckpt) else os.path.join(base_dir, args.multi_ckpt)

    sample_per_language = args.sample_per_language if args.sample_per_language > 0 else None
    max_languages = args.max_languages if args.max_languages > 0 else None

    raw_test = load_mixed_language_test(
        test_csv,
        exclude_language=args.exclude_language,
        sample_per_language=sample_per_language,
        max_languages=max_languages,
        seed=args.seed,
    )
    if len(raw_test) == 0:
        raise ValueError("No reviews found after applying mixed-language filters.")

    language_counts = pd.Series([row["language"] for row in raw_test]).value_counts().to_dict()
    print(f"Loaded {len(raw_test)} mixed-language test reviews from {test_csv}")
    print(f"Language distribution: {language_counts}")

    fallback_model = "distilbert-base-multilingual-cased"
    tokenizer = build_tokenizer(multi_ckpt, fallback_model)
    tokenized_test = tokenize_dataset(raw_test, tokenizer)

    single_preds = evaluate_checkpoint("Single-language", single_ckpt, tokenized_test, args.batch_size)
    multi_preds = evaluate_checkpoint("Multilingual", multi_ckpt, tokenized_test, args.batch_size)

    print("\n" + "=" * 60)
    print("Single-language example predictions")
    print("=" * 60)
    print_examples(raw_test, single_preds, args.num_examples, args.seed)

    single_errors, single_totals = count_errors_by_language(raw_test, single_preds)
    print("\nSingle-language errors by language:")
    for lang in sorted(single_totals.keys()):
        err = single_errors.get(lang, 0)
        total = single_totals[lang]
        print(f"{lang}: {err} errors / {total} samples")

    single_plot_path = os.path.join(base_dir, "errors_by_language_single.png")
    plot_errors_by_language(
        single_errors,
        single_totals,
        "Single-language errors by language",
        single_plot_path,
    )
    print(f"Single-language error plot saved to {single_plot_path}")

    print("\n" + "=" * 60)
    print("Multilingual example predictions")
    print("=" * 60)
    print_examples(raw_test, multi_preds, args.num_examples, args.seed)

    multi_errors, multi_totals = count_errors_by_language(raw_test, multi_preds)
    print("\nMultilingual errors by language:")
    for lang in sorted(multi_totals.keys()):
        err = multi_errors.get(lang, 0)
        total = multi_totals[lang]
        print(f"{lang}: {err} errors / {total} samples")

    multi_plot_path = os.path.join(base_dir, "errors_by_language_multilingual.png")
    plot_errors_by_language(
        multi_errors,
        multi_totals,
        "Multilingual errors by language",
        multi_plot_path,
    )
    print(f"Multilingual error plot saved to {multi_plot_path}")


if __name__ == "__main__":
    os.environ.setdefault("WANDB_DISABLED", "true")
    main()