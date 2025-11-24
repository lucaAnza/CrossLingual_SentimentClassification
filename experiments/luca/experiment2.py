import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from datasets import load_dataset, Value
import numpy as np
import wandb
from transformers import AutoTokenizer, EarlyStoppingCallback , DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
#from huggingface_hub import notebook_login




# ==================== Get env variables ====================
dataset_path = os.getenv('DATASET_PATH')
model_name = 'distilbert-base-multilingual-cased'
wandb_api_key = os.getenv("WANDB_API_KEY")
report_list = ["none"]  # default no reporting

# ==================== SETUP MODEL CHECK POINT DIRECTORIES ====================
base_output_dir = "models"
run_name = "experiment2_regression"  # define the name of the /models/<run_name> 
output_dir = os.path.join(base_output_dir, run_name)

"""# ==================== SETUP WANDB ====================
report_list = ["wandb"] 
wandb.init(
    project="CrossLingual-Sentiment-GPU(SDU)",   
    name=run_name + "_wandb",  
)
"""

# ==================== LOAD DATASET ====================
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )




# ==================== PREPROCESSING ====================
# Reduce dataset size for faster experimentation 
k = 150000
amazon_db['train'] = amazon_db['train'].shuffle(seed=42).select(range(k))
amazon_db['test'] = amazon_db['test'].shuffle(seed=42).select(range(min(30000 , k//6)))
amazon_db['validation'] = amazon_db['validation'].shuffle(seed=42).select(range(min(30000 , k//6)))

# Add ids column + mask column
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) # TODO : Instead of use review_body alone use also review_title

# Rename columns and remove unnecessary ones
amazon_db = amazon_db.rename_column("stars", "label")
amazon_db = amazon_db.rename_column("review_body", "text")
amazon_db = amazon_db.remove_columns(["Unnamed: 0", 'review_id', 'product_id', 'reviewer_id', 'review_title', 'language', 'product_category'])  # Remove unnecessary index column
amazon_db = amazon_db.cast_column("label", Value("float32"))  # regression needs continuous labels

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = amazon_db.map(preprocess_function, batched=True) # features: ['text', 'label' , 'ids' , 'mask']

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamically padding for token list (so we can batch different length inputs)
print("\n✅ Preprocessing completed. Db struct : " , amazon_db_tokenized)





# ==================== EVALUATION DEFINITION ====================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.reshape(-1)
    labels = labels.reshape(-1)
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return {"mae": mae.item(), "rmse": rmse.item()}



# ==================== MODEL TRAINING ====================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    problem_type="regression",
)
# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=8, # usually 2-5 epochs are sufficient for finetuning
    weight_decay=0.01, # penalize large weights, regularization, helps generalization
    eval_strategy="epoch", # other options: "no", "steps"
    save_strategy="epoch", # when the model is saved
    load_best_model_at_end=True,
    push_to_hub=False,  # Huggingface hub integration
    report_to=report_list, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=amazon_db_tokenized["train"],
    eval_dataset=amazon_db_tokenized["test"],
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





## ==================== MODEL EVALUATION ====================
trainer.evaluate(metric_key_prefix="test")
