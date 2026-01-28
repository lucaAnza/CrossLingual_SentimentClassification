import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR

from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Ensure repo root is on sys.path so `from experiments...` imports work when running this file directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.utils import preprocessing


load_dotenv(dotenv_path=os.path.join(REPO_ROOT, ".env"))  # take environment variables from .env (if present).


# ==================== PARAMS ====================
dataset_path = os.getenv("DATASET_PATH")
model_name = "distilbert-base-multilingual-cased"

languages_raw = os.getenv("LANGUAGES", "en,de,fr")
LANGUAGES = [l.strip() for l in languages_raw.split(",") if l.strip()]

sample_size_raw = os.getenv("SAMPLE_SIZE", "200000")
SAMPLE_SIZE = int(sample_size_raw) if sample_size_raw else None

run_prefix = os.getenv("RUN_PREFIX", "exp3_regression")
# Where to save trained models/checkpoints.
# In Colab, set e.g. `os.environ["SAVE_ROOT"] = "/content/drive/MyDrive/crosslingual_models"`
# before running this script to write directly to Google Drive.
base_output_dir = os.path.abspath(os.path.expanduser(os.getenv("SAVE_ROOT", os.path.join(REPO_ROOT, "models"))))

use_wandb = bool(os.getenv("WANDB_API_KEY"))
report_list = ["wandb"] if use_wandb else ["none"]

max_length_raw = os.getenv("MAX_LENGTH", "64")
MAX_LENGTH = int(max_length_raw) if max_length_raw else None


# ==================== UTILS ====================
def filter_by_language(db, lang):
    return db.filter(lambda x: x["language"] == lang)


def bin3_stars(x):
    """
    stars 1,2 -> 0 (negative)
    star  3   -> 1 (neutral)
    stars 4,5 -> 2 (positive)
    """
    x = np.array(x)
    return np.where(x <= 2, 0, np.where(x == 3, 1, 2))


def spearman_corr(a, b):
    """Spearman correlation without scipy: rank then Pearson."""
    a = np.asarray(a)
    b = np.asarray(b)

    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)

    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum())
    return float((ra * rb).sum() / denom) if denom != 0 else 0.0


def qwk(y_true, y_pred, min_rating=1, max_rating=5):
    """Quadratic Weighted Kappa (ordinal agreement)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n = max_rating - min_rating + 1
    O = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t - min_rating, p - min_rating] += 1

    act_hist = np.bincount(y_true - min_rating, minlength=n).astype(np.float64)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() if E.sum() != 0 else E

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    O = O / O.sum() if O.sum() != 0 else O
    num = (W * O).sum()
    den = (W * E).sum()
    return float(1.0 - num / den) if den != 0 else 0.0


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Flatten (Trainer may add extra dimensions)
    preds = predictions.reshape(-1)
    labels = labels.reshape(-1)

    # ---- Undo normalization: [0,1] -> [1,5] ----
    preds = preds * 4 + 1
    labels = labels * 4 + 1

    # ---- Regression metrics ----
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    # ---- Discretize to stars (1..5) for classification-like metrics ----
    preds_star = np.clip(np.rint(preds), 1, 5).astype(int)
    labels_star = np.clip(np.rint(labels), 1, 5).astype(int)

    # ---- 5-class rounded accuracy (exact star match) ----
    rounded_accuracy = np.mean(preds_star == labels_star)

    # ---- 3-class rounded accuracy ----
    preds_3 = bin3_stars(preds_star)
    labels_3 = bin3_stars(labels_star)
    rounded_accuracy_3 = np.mean(preds_3 == labels_3)

    # ---- Ordinal metrics ----
    spearmanr = spearman_corr(preds, labels)
    qwk_score = qwk(labels_star, preds_star, 1, 5)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rounded_accuracy": float(rounded_accuracy),
        "rounded_accuracy_3": float(rounded_accuracy_3),
        "spearmanr": float(spearmanr),
        "qwk": float(qwk_score),
    }


# ==================== LOAD DATASET ====================
amazon_db = load_dataset(
    "csv",
    data_files={
        "train": dataset_path + "/train.csv",
        "test": dataset_path + "/test.csv",
        "validation": dataset_path + "/validation.csv",
    },
)


# ==================== TRAIN PER LANGUAGE ====================
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

results = []

for lang in LANGUAGES:
    run_name = f"{run_prefix}_{lang}_{SAMPLE_SIZE or 'full'}"
    output_dir = os.path.join(base_output_dir, run_name)

    if use_wandb:
        wandb.init(project="CrossLingual-Sentiment-GPU(SDU)", name=run_name)

    lang_db = filter_by_language(amazon_db, lang)
    amazon_db_tokenized = preprocessing(
        lang_db,
        tokenizer,
        k=SAMPLE_SIZE,
        task="Regression",
        max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        report_to=report_list,
        gradient_checkpointing=True,
    )

    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=amazon_db_tokenized["train"],
        eval_dataset=amazon_db_tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ==================== SET THE DEVICE ====================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration available)")

    model.to(device)
    print(f"Starting training for language: {lang}")

    resume_env = os.getenv("RESUME", "").strip().lower()
    if resume_env in {"y", "yes", "true", "1"}:
        resume = True
    elif resume_env in {"n", "no", "false", "0"}:
        resume = False
    else:
        try:
            print("Do you want to resume from a checkpoint? (y/n)")
            resume = input().strip().lower() == "y"
        except EOFError:
            resume = False
    trainer.train(resume_from_checkpoint=resume)

    # Save a loadable model to `output_dir`.
    # With `load_best_model_at_end=True`, this persists the best weights for eval.
    trainer.save_model(output_dir)

    test_metrics = trainer.evaluate(eval_dataset=amazon_db_tokenized["test"], metric_key_prefix="test")
    test_metrics["language"] = lang
    results.append(test_metrics)

    if use_wandb:
        wandb.finish()


print("\nPer-language test metrics:")
for row in results:
    print(row)

