from transformers import AutoTokenizer , DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
from dotenv import load_dotenv
from datasets import load_dataset, Value
import numpy as np
load_dotenv()  # loads .env 


# =========  PARAMS =========
run_name = "models/exp2_regression_300k/checkpoint-12504"
dataset_path = os.getenv('DATASET_PATH')

# ==================== LOAD DATASET ====================
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )

# ==================== PREPROCESSING ====================
# Reduce dataset size for faster experimentation 
k = 200000
amazon_db['train'] = amazon_db['train'].shuffle(seed=42).select(range(k))
amazon_db['test'] = amazon_db['test'].shuffle(seed=42).select(range(min(30000 , k//6)))
amazon_db['validation'] = amazon_db['validation'].shuffle(seed=42).select(range(min(30000 , k//6)))

# Rename columns and remove unnecessary ones
amazon_db = amazon_db.rename_column("stars", "label")
amazon_db = amazon_db.rename_column("review_body", "text")
amazon_db = amazon_db.remove_columns(["Unnamed: 0", 'review_id', 'product_id', 'reviewer_id', 'review_title', 'language', 'product_category'])  # Remove unnecessary index column
amazon_db = amazon_db.cast_column("label", Value("float32"))  # regression needs continuous labels

# Tokenization
# Add ids column + mask column
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True) # TODO : Instead of use review_body alone use also review_title

tokenizer = AutoTokenizer.from_pretrained(run_name)
amazon_db_tokenized = amazon_db.map(preprocess_function, batched=True) # features: ['text', 'label' , 'ids' , 'mask']  (batched=True for speed up the mapping process)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Defines how batches are created during training (uses dynamic padding)
print("\n✅ Preprocessing completed. Db struct : " , amazon_db_tokenized)



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
    }


# =========  LOAD THE MODEL =========
model = AutoModelForSequenceClassification.from_pretrained(run_name)
tokenizer = AutoTokenizer.from_pretrained(run_name)

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


