import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from datasets import load_dataset
import numpy as np
import wandb
from transformers import EarlyStoppingCallback , AutoTokenizer , DataCollatorWithPadding 
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from utils import preprocessing
#from huggingface_hub import notebook_login



# ==================== Get env variables ====================
dataset_path = os.getenv('DATASET_PATH')
model_name = 'distilbert-base-multilingual-cased'
wandb_api_key = os.getenv("WANDB_API_KEY")
report_list = ["none"]  # default no reporting

# ==================== SETUP MODEL CHECK POINT DIRECTORIES ====================
base_output_dir = "models"
run_name = "exp2_regression_200k"  # define the name of the /models/<run_name> 
output_dir = os.path.join(base_output_dir, run_name)

# ==================== SETUP WANDB ====================
report_list = ["wandb"] 
wandb.init(
    project="CrossLingual-Sentiment-GPU(SDU)",   
    name=run_name + "_wandb",  
)


# ==================== LOAD DATASET ====================
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )


# ==================== PREPROCESSING ====================
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = preprocessing(amazon_db , tokenizer , k = 200000 , task = 'Regression')


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

# ==================== MODEL TRAINING ====================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamically padding for token list (so we can batch different length inputs) [used in training]


# default loss function : torch.nn.MSELoss()
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    problem_type="regression",
)

# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir=output_dir,                 
    learning_rate=2e-5,                    
    per_device_train_batch_size=32,        # Moderate batch for stability; avoids GPU OOM and helps generalization
    per_device_eval_batch_size=128,        # Larger eval batch to speed up evaluation (no backprop, less memory pressure)
    gradient_accumulation_steps=4,         # Simulates a bigger batch (32*4=128) while keeping memory usage low
    num_train_epochs=8,                    
    weight_decay=0.01,                     # Regularization to reduce overfitting on noisy review ratings
    lr_scheduler_type="cosine",            # Smooth LR decay often improves final performance vs constant LR
    warmup_ratio=0.06,                     # Warmup avoids unstable updates at the start of fine-tuning
    max_grad_norm=1.0,                     # Gradient clipping prevents exploding gradients (more stable training)
    eval_strategy="epoch",                 
    save_strategy="epoch",                 
    load_best_model_at_end=True,           
    metric_for_best_model="rmse",          # Choose the model that minimizes RMSE (main regression metric)
    greater_is_better=False,               # RMSE must be minimized (lower = better)
    report_to=["wandb"],                   
)

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
# Check if MPS is available (for Mac with M1/M2/M3 chips)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚠️ Using MPS device")
# Check if CUDA is available (for NVIDIA GPUs)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (no GPU acceleration available)")

model.to(device)
print("✅ Trainer is set up. Starting training...")


print("Do you want to resume from a checkpoint? (y/n)")
if input().strip().lower() == "y":
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
