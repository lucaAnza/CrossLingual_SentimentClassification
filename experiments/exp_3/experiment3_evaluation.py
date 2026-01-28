import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Ensure repo root is on sys.path so `from experiments...` imports work when running this file directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.utils import preprocessing

load_dotenv(dotenv_path=os.path.join(REPO_ROOT, ".env"))  # loads .env (if present)


# =========  PARAMS =========
dataset_path = os.getenv("DATASET_PATH")
run_prefix = os.getenv("RUN_PREFIX", "exp3_regression")
languages_raw = os.getenv("LANGUAGES", "en,de,fr")
LANGUAGES = [l.strip() for l in languages_raw.split(",") if l.strip()]

sample_size_raw = os.getenv("SAMPLE_SIZE", "200000")
SAMPLE_SIZE = int(sample_size_raw) if sample_size_raw else None
max_length_raw = os.getenv("MAX_LENGTH", "64")
MAX_LENGTH = int(max_length_raw) if max_length_raw else None


def find_latest_checkpoint(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Expected a directory created by training. Run:\n"
            f"  python experiments/exp_3/experiment3.py\n"
            f"from the repo root (or set your PYTHONPATH accordingly)."
        )
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        # Fallback: allow loading directly from the run folder if a final model was saved there.
        # (Our training script calls `trainer.save_model(output_dir)` and `tokenizer.save_pretrained(output_dir)`.)
        has_config = os.path.isfile(os.path.join(model_dir, "config.json"))
        has_weights = any(
            os.path.isfile(os.path.join(model_dir, f))
            for f in ("model.safetensors", "pytorch_model.bin")
        )
        if has_config and has_weights:
            return model_dir

        contents = sorted(os.listdir(model_dir))
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}\n"
            f"Directory contents: {contents}\n"
            f"Fix: run training so it saves either a `checkpoint-*` folder (per epoch) or a final model.\n"
            f"Also ensure env vars match between train and eval (RUN_PREFIX, SAMPLE_SIZE, LANGUAGES)."
        )
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(model_dir, checkpoints[-1])


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
if not dataset_path:
    raise ValueError(
        "DATASET_PATH is not set. Put it in your .env or export it, e.g.\n"
        "  DATASET_PATH=/absolute/path/to/dataset_folder\n"
        "The folder must contain train.csv, test.csv, validation.csv."
    )

amazon_db = load_dataset(
    "csv",
    data_files={
        "train": dataset_path + "/train.csv",
        "test": dataset_path + "/test.csv",
        "validation": dataset_path + "/validation.csv",
    },
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

results = []

for lang in LANGUAGES:
    run_name = f"{run_prefix}_{lang}_{SAMPLE_SIZE or 'full'}"
    model_dir = os.path.join(REPO_ROOT, "models", run_name)
    checkpoint_path = find_latest_checkpoint(model_dir)

    lang_db = filter_by_language(amazon_db, lang)
    amazon_db_tokenized = preprocessing(
        lang_db,
        tokenizer,
        k=SAMPLE_SIZE,
        task="Regression",
        max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=128,
        report_to=None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        eval_dataset=amazon_db_tokenized["test"],
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    metrics["language"] = lang
    results.append(metrics)

print("\nPer-language metrics:")
for row in results:
    print(row)

