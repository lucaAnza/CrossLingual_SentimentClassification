from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from datasets import load_dataset, Value


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

tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = amazon_db.map(preprocess_function, batched=True) # features: ['text', 'label' , 'ids' , 'mask']  (batched=True for speed up the mapping process)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Defines how batches are created during training (uses dynamic padding)
print("\n✅ Preprocessing completed. Db struct : " , amazon_db_tokenized)


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
    tokenizer=tokenizer,
    eval_dataset=amazon_db_tokenized["test"],
    compute_metrics=compute_metrics,   # same function you used before
)

# =========  COMPUTE THE METRICS =========
def compute_metrics(eval_pred):
    predictions, labels = eval_pred   

    preds = predictions.reshape(-1)
    labels = labels.reshape(-1)

    # --- Regression metrics ---
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    # --- Accuracy metrics ---
    preds_rounded = np.rint(preds)   # Round predictions to nearest integer ex : 2.3→2, 2.8→3
    preds_rounded = np.clip(preds_rounded, 1, 5) # Clamp values to valid range (1–5) if needed
    accuracy = np.mean(preds_rounded == labels) # Compute accuracy

    return {
        "mae": mae.item(),     # item() to convert numpy types to native Python types
        "rmse": rmse.item(),
        "accuracy": accuracy.item()
    }

metrics = trainer.evaluate()
print(metrics)
