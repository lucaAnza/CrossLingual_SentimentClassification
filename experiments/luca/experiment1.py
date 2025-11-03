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
from huggingface_hub import notebook_login
import torch




# LOGIN TO HUGGINGFACE HUB
# notebook_login()


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
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

model_name = os.getenv('MODEL_NAME')
tokenizer = AutoTokenizer.from_pretrained(model_name)
amazon_db_tokenized = amazon_db.map(preprocess_function, batched=True) # features: ['text', 'label' , 'ids' , 'mask']
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
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model = AutoModelForSequenceClassification.from_pretrained( model_name , num_labels=2 , id2label=id2label , label2id=label2id )



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