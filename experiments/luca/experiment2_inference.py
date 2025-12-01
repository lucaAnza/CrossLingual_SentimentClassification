import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


# ==========================================
# 0. Params
run_name = 'exp2_regression_300k'
# ==========================================


# ==========================================
# 1. Ask model name
# ==========================================

if(not(run_name)):
    print("Enter the model directory (inside /models):")
    run_name = input().strip()

model_dir = os.path.join("models", run_name)

if not os.path.exists(model_dir):
    raise ValueError(f"❌ Directory {model_dir} does not exist.")

# Automatically find the best checkpoint
checkpoints = [
    os.path.join(model_dir, d)
    for d in os.listdir(model_dir)
    if "checkpoint" in d
]

if not checkpoints:
    print("⚠️ No checkpoints found. Using the base model directory.")
    best_checkpoint = model_dir
else:
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    best_checkpoint = checkpoints[-1]

print(f"📂 Loading best checkpoint: {best_checkpoint}")

# ==========================================
# 2. Load tokenizer + model
# ==========================================
# TOKENIZATION
tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)

# MODEL LOAD
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device) # Moves the tensors to device
model.eval()   # puts the model in evaluation mode.
print(f"🔧 Using device: {device}")
print("✅ Model loaded.")


# ==========================================
# 3. Inference function
# ==========================================
def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device) # Returns PyTorch tensors instead of Python lists.

    with torch.no_grad():  # Disable gradient computation
        outputs = model(**inputs)
    
    # Regression output → a single value
    score = outputs.logits.squeeze().cpu().item() # tensor([[3.2332]] , dev = cuda) --> tensor(3.2332 , ...) --> tensor([[3.2332]]) --> 3.2332

    # Clamp + round
    rounded_score = int(np.clip(np.rint(score), 1, 5))

    return score, rounded_score


# ==========================================
# 4. Interactive loop
# ==========================================
print("\nType a review and press Enter. Type 'exit' to quit.")

while True:
    text = input("\n📝 Review: ").strip()

    if text.lower() == "exit":
        print("👋 Bye!")
        break

    raw, rounded = predict_sentiment(text)

    print(f"➡️ Raw regression score: {raw:.4f}")
    print(f"⭐ Predicted stars: {rounded}/5")
