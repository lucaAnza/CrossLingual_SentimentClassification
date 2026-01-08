from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import evaluate
from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np
load_dotenv()  # loads .env 
from utils import preprocessing


# =========  PARAMS =========
model_name = "models/exp1_classification_1.2m/checkpoint-28125"
dataset_path = os.getenv('DATASET_PATH')

# ==================== LOAD DATASET ====================
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )

# ==================== PREPROCESSING ====================
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = preprocessing(amazon_db , tokenizer , task = 'Classification')

# ===================== DISABLE WANDB =====================
if "WANDB_API_KEY" in os.environ:
    del os.environ["WANDB_API_KEY"]

# =========  DEFINE THE METRICS =========
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

def softmax_np(x):
    """Stable softmax for (N, C) logits."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def qwk_classification(y_true, y_pred, num_classes=None, min_label=None):
    """
    Quadratic Weighted Kappa for ORDINAL classification.
    y_true, y_pred: arrays of class indices (e.g., 0..4 or 1..5).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    if min_label is None:
        min_label = int(min(y_true.min(), y_pred.min()))
    if num_classes is None:
        max_label = int(max(y_true.max(), y_pred.max()))
        num_classes = max_label - min_label + 1

    # shift to 0..n-1
    yt = y_true - min_label
    yp = y_pred - min_label

    # safety: clip in range (useful if some preds are out of bounds)
    yt = np.clip(yt, 0, num_classes - 1)
    yp = np.clip(yp, 0, num_classes - 1)

    # Observed matrix O
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(O, (yt, yp), 1.0)

    N = O.sum()
    if N == 0:
        return 0.0

    # Convert to probabilities
    O = O / N
    act_hist = O.sum(axis=1)          # P(true = i)
    pred_hist = O.sum(axis=0)         # P(pred = j)
    E = np.outer(act_hist, pred_hist) # expected probs under independence

    # Weight matrix W (quadratic)
    idx = np.arange(num_classes, dtype=np.float64)
    W = ((idx[:, None] - idx[None, :]) ** 2) / ((num_classes - 1) ** 2)

    num = np.sum(W * O)
    den = np.sum(W * E)
    return float(1.0 - num / den) if den != 0 else 0.0


# 5-class → 3-class mapping
def bin3_array(x):
    """
    0,1 -> 0  (negative)
    2   -> 1  (neutral)
    3,4 -> 2  (positive)
    """
    x = np.array(x)
    return np.where(
        x <= 1, 0,
        np.where(x == 2, 1, 2)
    )
# Example:
# logits :  [[ 3.598274   2.2076087 -0.9162292 -4.189072  -5.289907 ]]
# labels :  [1]
# preds : 0
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # ----- 5-class metrics -----
    accuracy_5 = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro_5 = f1.compute(predictions=preds, references=labels, average="macro")["f1"]

    # QWK 5-class 
    qwk_5 = qwk_classification(labels, preds, num_classes=5, min_label=0)

    # ----- Spearman (classification) -----
    # Option A: use argmax predictions (many ties -> sometimes less informative)
    spearman_argmax = spearman_corr(preds, labels)

    # Option B (recommended): expected rating from probabilities (fewer ties)
    probs = softmax_np(logits)                         # (N, 5)
    expected = (probs * np.arange(5)).sum(axis=1)      # values in [0,4]
    spearman_expected = spearman_corr(expected, labels)

    # ----- 3-class metrics -----
    labels_3 = bin3_array(labels)
    preds_3 = bin3_array(preds)

    accuracy_3 = accuracy.compute(predictions=preds_3, references=labels_3)["accuracy"]
    f1_macro_3 = f1.compute(predictions=preds_3, references=labels_3, average="macro")["f1"]

    qwk_3 = qwk_classification(labels_3, preds_3, num_classes=3, min_label=0)

    return {
        "accuracy": accuracy_5,
        "f1_macro": f1_macro_5,
        "qwk": qwk_5,
        "accuracy_3": accuracy_3,
        "f1_macro_3": f1_macro_3,
        "qwk_3": qwk_3,
        "spearman_argmax": float(spearman_argmax),
        "spearman_expected": float(spearman_expected),
        "qwk_5": qwk_5,
    }

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


# =========  LOAD THE MODEL =========
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./tmp_eval",    # Where the model is saved after the revaluation
    per_device_eval_batch_size=128,
    report_to=None,
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

