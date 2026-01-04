from transformers import AutoTokenizer , DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
from dotenv import load_dotenv
from datasets import load_dataset, Value
import numpy as np
import evaluate
load_dotenv()  # loads .env 
from utils import preprocessing


# =========  PARAMS =========
model_name = "models/exp2_regression_1.2m/checkpoint-37500"
dataset_path = os.getenv('DATASET_PATH')

# ==================== LOAD DATASET ====================
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )

# ==================== PREPROCESSING ====================
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = preprocessing(amazon_db , tokenizer , task = 'Regression' , k = 10)




# ==================== EVALUATION DEFINITION ====================
# 5-class → 3-class mapping (expects 0..4)
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

f1_metric = evaluate.load("f1")

# EXAMPLE
## predictions :  [[0.08773001]]   # value from 0 -> 1
## labels :  [0.25]       # 0.00 = ★ ;  # 0.25 = ★★ ; ...
## preds :  [0.08773001]
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Flatten (Trainer may add extra dimensions)
    preds = predictions.reshape(-1)
    labels = labels.reshape(-1)

    # ---- Undo normalization: [0,1] → [1,5] ----
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

    # ---- F1 metrics (macro) ----
    f1_5 = f1_metric.compute(
        predictions=preds_star.tolist(),
        references=labels_star.tolist(),
        average="macro",
        labels=[1, 2, 3, 4, 5],
    )["f1"]

    classes_3 = sorted(set(labels_3.tolist()) | set(preds_3.tolist()))
    f1_3 = f1_metric.compute(
        predictions=preds_3.tolist(),
        references=labels_3.tolist(),
        average="macro",
        labels=classes_3,
    )["f1"]

    # ---- Ordinal metrics ----
    spearmanr = spearman_corr(preds, labels)          # on continuous (1..5 floats)
    qwk_score = qwk(labels_star, preds_star, 1, 5)    # on rounded stars (1..5 ints)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rounded_accuracy": float(rounded_accuracy),
        "rounded_accuracy_3": float(rounded_accuracy_3),
        "spearmanr": float(spearmanr),
        "qwk": float(qwk_score),
        "f1_5": float(f1_5),
        "f1_3": float(f1_3),
    }


# =========  LOAD THE MODEL =========
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamically padding for token list (so we can batch different length inputs) [used in training]
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="./tmp_eval",
    per_device_eval_batch_size=128,
)
trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    eval_dataset=amazon_db_tokenized["test"],
    compute_metrics=compute_metrics,   
)


# =========  RECOMPUTE THE METRICS =========
print("\n⚙ Recomputing the metrics for the datasets!")
metrics = trainer.evaluate()
print("\n✅ Metring re-computed with success!")
print("Metrics : ")
for k,v in metrics.items():
    print("\t",k,":"," ",v)


