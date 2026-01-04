import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 0. Params
# ==========================================
BASE_MODELS_DIR = "models"
task_type = "classification"
# Path inside /models 
# run_name = "exp2_regression_1.2m/checkpoint-37500"    # REGRESSION
run_name = "exp1_classification_1.2m/checkpoint-28125"  # CLASSIFICATION




# ==========================================
# 1. Define model path
# ==========================================
model_path = os.path.join(BASE_MODELS_DIR , run_name)

# ==========================================
# 2. Load tokenizer + model
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"🔧 Using device: {device}")
print(f"✅ Model loaded from: {model_path}")
print(f"🧠 Task type: {task_type}")


# ==========================================
# 3. Inference functions
# ==========================================
def predict_regression(text: str) -> tuple[float, int]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Regression output → single scalar
    score = outputs.logits.squeeze().cpu().item()

    # [0→1] TO [1→5]
    score = (score * 4) + 1

    # Round + clamp to stars [1..5]
    stars = int(np.clip(np.rint(score), 1, 5))
    return score, stars


def predict_classification(text: str) -> tuple[int, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze(0)  # shape: [num_labels]
    probs = torch.softmax(logits, dim=-1)

    pred_class = int(torch.argmax(probs).cpu().item())
    confidence = float(probs[pred_class].cpu().item())

    # [0-4] -> [1-5]
    stars = pred_class + 1
    return stars, confidence


def predict(text: str):
    if task_type.lower() == "regression":
        raw, stars = predict_regression(text)
        return {"raw": raw, "stars": stars}
    elif task_type.lower() == "classification":
        stars, conf = predict_classification(text)
        return {"stars": stars, "confidence": conf}
    else:
        raise ValueError('task_type must be either "regression" or "classification".')


# ==========================================
# 4. Interactive loop
# ==========================================
print("\nType a review and press Enter. Type 'exit' to quit.")

while True:
    text = input("\n📝 Review: ").strip()

    if text.lower() == "exit":
        print("👋 Bye!")
        break

    out = predict(text)

    if task_type.lower() == "regression":
        print(f"➡️ Regression score normalized: {out['raw']:.4f}")
        print(f"⭐ Predicted stars using rounding: {out['stars']}/5")
    else:
        print(f"⭐ Predicted stars: {out['stars']}/5")
        print(f"✅ Confidence (softmax probability predicted class): {out['confidence']:.4f}")
