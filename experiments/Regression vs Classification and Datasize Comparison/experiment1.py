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
from utils import preprocessing



# ==================== Get env variables ====================
dataset_path = os.getenv('DATASET_PATH')
model_name = 'distilbert-base-multilingual-cased'
wandb_api_key = os.getenv("WANDB_API_KEY")
report_list = ["none"]  # default no reporting

# ==================== SETUP MODEL CHECK POINT DIRECTORIES ====================
base_output_dir = "models"
run_name = "exp1_classification_1.2m_wandb-"  # define the name of the /models/<run_name> 
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
amazon_db_tokenized = preprocessing(amazon_db , tokenizer , k = 2000 , task = 'Regression')



# ==================== EVALUATION DEFINITION ====================
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
    accuracy_5 = accuracy.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    f1_macro_5 = f1.compute(
        predictions=preds,
        references=labels,
        average="macro"
    )["f1"]

    # ----- 3-class metrics -----
    labels_3 = bin3_array(labels)
    preds_3 = bin3_array(preds)

    accuracy_3 = accuracy.compute(
        predictions=preds_3,
        references=labels_3
    )["accuracy"]

    f1_macro_3 = f1.compute(
        predictions=preds_3,
        references=labels_3,
        average="macro"
    )["f1"]

    return {
        "accuracy": accuracy_5,
        "f1_macro": f1_macro_5,
        "accuracy_3": accuracy_3,
        "f1_macro_3": f1_macro_3,
    }


accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")



# ==================== MODEL TRAINING ====================
id2label = {0: "VERY NEGATIVE", 1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE", 4: "VERY POSITIVE"}
label2id = {"VERY NEGATIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2, "POSITIVE": 3, "VERY POSITIVE": 4}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # dynamically padding for token list (so we can batch different length inputs)

model = AutoModelForSequenceClassification.from_pretrained( model_name , num_labels=5 , id2label=id2label , label2id=label2id )
# TRAINING ARGUMENTS (loss function by default : categorical Cross-Entropy)
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    warmup_ratio = 0.1, # 10% of total training steps are used to increase the learning rate linearly from 0 → base LR
    lr_scheduler_type = "linear", # Increase and decrease aftet warmup is linear
    per_device_train_batch_size=64, # Better to keep it smaller (better generalization)
    gradient_accumulation_steps=2,  # In this way accumulate the gradient so is like 128 (best throughput)
    per_device_eval_batch_size=128,  # I can leave big (gradient not computed into evaluation)
    num_train_epochs=6, # every epoch should have 
    weight_decay=0.01, # penalize large weights, regularization, helps generalization
    eval_strategy="epoch", # other options: "no", "steps"
    save_strategy="epoch", # when the model is saved
    load_best_model_at_end=True,
    report_to=report_list, 
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



## ==================== MODEL EVALUATION ====================
trainer.evaluate(metric_key_prefix="test")
