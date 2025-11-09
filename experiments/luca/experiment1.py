import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from datasets import load_dataset
import evaluate
import numpy as np
import wandb
from transformers import AutoTokenizer, EarlyStoppingCallback , DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
#from huggingface_hub import notebook_login




# ==================== Get env variables ====================
dataset_path = os.getenv('DATASET_PATH')
model_name = os.getenv('MODEL_NAME')

# ==================== SETUP MODEL CHECK POINT DIRECTORIES ====================
base_output_dir = "models"
run_name = "multilingual-distilbert-finetuned-amazon-reviews"
output_dir = os.path.join(base_output_dir, run_name)

# ==================== SETUP WANDB ====================
wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "ai509"  # Set the env variable so wandb che read it
wandb.init(name="multilingual-distilbert (EXP1- reduced data)")


# ==================== LOAD DATASET ====================
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )




# ==================== PREPROCESSING ====================
# Reduce dataset size for faster experimentation 
k = 1000
amazon_db['train'] = amazon_db['train'].shuffle(seed=42).select(range(k))
amazon_db['test'] = amazon_db['test'].shuffle(seed=42).select(range(k//6))
amazon_db['validation'] = amazon_db['validation'].shuffle(seed=42).select(range(k//6))    
   

# Fix labels to start from 0
def adjust_label(example):
    example['label'] = example['label'] - 1
    return example

# Add ids column + mask column
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) # TODO : Instead of use review_body alone use also review_title

# Rename columns and remove unnecessary ones
amazon_db = amazon_db.rename_column("stars", "label")
amazon_db = amazon_db.rename_column("review_body", "text")
amazon_db = amazon_db.remove_columns(["Unnamed: 0", 'review_id', 'product_id', 'reviewer_id', 'review_title', 'language', 'product_category'])  # Remove unnecessary index column

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = amazon_db.map(preprocess_function, batched=True) # features: ['text', 'label' , 'ids' , 'mask']


# Fix labels to start from 0
amazon_db_tokenized = amazon_db_tokenized.map(adjust_label)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamically padding for token list (so we can batch different length inputs)
print("\n✅ Preprocessing completed. Db struct : " , amazon_db_tokenized)





# ==================== EVALUATION DEFINITION ====================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print("\n\n" , predictions.shape)
    print(labels.shape)
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

accuracy = evaluate.load("accuracy")



# ==================== MODEL TRAINING ====================
id2label = {0: "VERY NEGATIVE", 1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE", 4: "VERY POSITIVE"}
label2id = {"VERY NEGATIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2, "POSITIVE": 3, "VERY POSITIVE": 4}
model = AutoModelForSequenceClassification.from_pretrained( model_name , num_labels=5 , id2label=id2label , label2id=label2id )
# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=2,   # it is useful when you have memory issues (e.g. OOM errors) to simulate larger batch sizes
    num_train_epochs=5, # usually 2-5 epochs are sufficient for finetuning
    weight_decay=0.01, # penalize large weights, regularization, helps generalization
    eval_strategy="epoch", # other options: "no", "steps"
    save_strategy="epoch", # when the model is saved
    load_best_model_at_end=True,
    push_to_hub=False,  # Huggingface hub integration
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=amazon_db_tokenized["train"],
    eval_dataset=amazon_db_tokenized["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # need to switch from epochs to steps and set eval_steps for this to work
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
trainer.train()





## ==================== MODEL EVALUATION ====================
trainer.evaluate(metric_key_prefix="test")

