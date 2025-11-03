import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
hf_token = os.getenv("HF_TOKEN")
from datasets import load_dataset
import evaluate
import numpy as np
import wandb
from transformers import AutoTokenizer, EarlyStoppingCallback , DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
#from huggingface_hub import notebook_login






# ==================== SETUP WANDB ====================
base_output_dir = "models"
run_name = "multilingual-distilbert-finetuned-amazon-reviews (Experiment1)"
output_dir = os.path.join(base_output_dir, run_name)
wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "ai509"  # Set the env variable so wandb che read it
wandb.init(name=run_name)



# ==================== LOAD DATASET ====================
dataset_path = os.getenv('DATASET_PATH')
amazon_db = load_dataset( 'csv' , data_files={ 'train': dataset_path + '/train.csv', 'test': dataset_path + '/test.csv'  , 'validation': dataset_path + '/validation.csv' } )




# ==================== PREPROCESSING ====================
model_name = os.getenv('MODEL_NAME')  
def preprocess_function(examples):
    return tokenizer(examples["review_body"], padding="max_length", truncation=True) # TODO : Instead of use review_body alone use also review_title

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = amazon_db.map(preprocess_function, batched=True) # features: ['text', 'label' , 'ids' , 'mask']

# Rename columns and remove unnecessary ones
amazon_db_tokenized = amazon_db_tokenized.rename_column("stars", "label")
amazon_db_tokenized = amazon_db_tokenized.rename_column("review_body", "text")
amazon_db_tokenized = amazon_db_tokenized.remove_columns(["Unnamed: 0", 'review_id', 'product_id', 'reviewer_id', 'review_title', 'language', 'product_category'])  # Remove unnecessary index column

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamically padding for token list (so we can batch different length inputs)





# ==================== EVALUATION DEFINITION ====================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions.shape)
    print(labels.shape)
    print(predictions)
    print(labels)
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

accuracy = evaluate.load("accuracy")



# ==================== MODEL TRAINING ====================
id2label = {1: "VERY NEGATIVE", 2: "NEGATIVE", 3: "NEUTRAL", 4: "POSITIVE", 5: "VERY POSITIVE"}
label2id = {"VERY NEGATIVE": 1, "NEGATIVE": 2, "NEUTRAL": 3, "POSITIVE": 4, "VERY POSITIVE": 5}
model = AutoModelForSequenceClassification.from_pretrained( model_name , num_labels=5 , id2label=id2label , label2id=label2id )
# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # gradient_accumulation_steps=2,
    num_train_epochs=2, # usually 2-5 epochs are sufficient for finetuning
    weight_decay=0.01, # penalize large weights, regularization, helps generalization
    eval_strategy="epoch", # other options: "no", "steps"
    save_strategy="epoch",
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
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # need to switch from epochs to steps and set eval_steps for this to work
)

trainer.train()



# ==================== SET THE DEVICE ====================
# Check if MPS is available (for Mac with M1/M2/M3 chips)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
# Check if CUDA is available (for NVIDIA GPUs)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU acceleration available)")

model.to(device)



## ==================== MODEL EVALUATION ====================
trainer.evaluate(metric_key_prefix="test")

