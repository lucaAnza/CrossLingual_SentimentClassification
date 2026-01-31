from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import sys

# Import preprocessing function
import datapreprocessing

# Get mode from command line or use default
mode = 'multilingual'  # default
if len(sys.argv) > 1:
    mode = sys.argv[1].lower()
    if mode not in ['multilingual', 'single_language']:
        print(f"Warning: Invalid mode '{mode}', using 'multilingual'")
        mode = 'multilingual'

# Direct storage under /data01/aiello to avoid local disk usage
BASE_STORAGE_DIR = '/data01/aiello'
wandb_dir = os.path.join(BASE_STORAGE_DIR, 'wandb')
output_dir = os.path.join(BASE_STORAGE_DIR, f'results_{mode}')
logging_dir = os.path.join(BASE_STORAGE_DIR, f'logs_{mode}')
wandb_cache_dir = os.path.join(BASE_STORAGE_DIR, 'wandb_cache')
wandb_config_dir = os.path.join(BASE_STORAGE_DIR, 'wandb_config')
hf_home = os.path.join(BASE_STORAGE_DIR, 'hf_home')
hf_datasets_cache = os.path.join(BASE_STORAGE_DIR, 'hf_datasets_cache')
hf_transformers_cache = os.path.join(BASE_STORAGE_DIR, 'hf_transformers_cache')
mpl_config_dir = os.path.join(BASE_STORAGE_DIR, 'mpl_cache')
tmp_dir = os.path.join(BASE_STORAGE_DIR, 'tmp')

for path in (
    wandb_dir,
    output_dir,
    logging_dir,
    wandb_cache_dir,
    wandb_config_dir,
    hf_home,
    hf_datasets_cache,
    hf_transformers_cache,
    mpl_config_dir,
    tmp_dir,
):
    os.makedirs(path, exist_ok=True)

os.environ['WANDB_DIR'] = wandb_dir
os.environ.setdefault('WANDB_CACHE_DIR', wandb_cache_dir)
os.environ.setdefault('WANDB_CONFIG_DIR', wandb_config_dir)
os.environ.setdefault('HF_HOME', hf_home)
os.environ.setdefault('HF_DATASETS_CACHE', hf_datasets_cache)
os.environ.setdefault('TRANSFORMERS_CACHE', hf_transformers_cache)
os.environ.setdefault('MPLCONFIGDIR', mpl_config_dir)
os.environ.setdefault('TMPDIR', tmp_dir)
os.environ.setdefault('TEMP', tmp_dir)
os.environ.setdefault('TMP', tmp_dir)

# Preprocess data (without tokenization)
print(f"Preprocessing data in {mode} mode...")
train_dataset, val_dataset, model_name, _, language_info = datapreprocessing.preprocess_data(mode)
if language_info:
    print(f"Language selection: {language_info}")

# Now load tokenizer and apply tokenization
print(f"\nLoading tokenizer: {model_name}")
try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("\nTrying alternative approach...")
    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
        raise RuntimeError("Cannot load tokenizer due to library compatibility issues. Please update your conda environment.")

# Tokenization function
def tokenize_function(examples):
    texts = []
    for i in range(len(examples['review_title'])):
        title = examples['review_title'][i] if examples['review_title'][i] else ""
        body = examples['review_body'][i] if examples['review_body'][i] else ""
        combined = f"{title} {body}".strip()
        texts.append(combined)
    
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=512
    )

# Apply tokenization
print("Tokenizing datasets...")
remove_cols = ['review_title', 'review_body']
if 'language' in train_dataset.column_names:
    remove_cols.append('language')

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print("Tokenization complete!")

# Load the pre-trained model for classification
print(f"Loading model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,  # Three classes: "Bad" (0), "Neutral" (1), "Good" (2)
)

# Set training arguments with mode-specific output directories
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",                  # Evaluate every epoch
    save_strategy="steps",                  # Save every 5000 steps
    save_steps=5000,                        # Save every 5000 steps
    logging_dir=logging_dir,                # Directory for logs
    logging_strategy="steps",              # Log by steps (not epochs)
    logging_steps=1,                        # Log every step
    num_train_epochs=3,                     # Number of epochs
    per_device_train_batch_size=8,          # Reduced batch size for training
    per_device_eval_batch_size=32,          # Reduced batch size for evaluation
    weight_decay=0.01,                      # Regularization to avoid overfitting
    save_total_limit=2,                     # Keep only the last 2 checkpoints
    load_best_model_at_end=True,            # Load best model at the end of training
    metric_for_best_model="eval_loss",      # Metric to determine best model
    greater_is_better=False,                # Lower loss is better
    report_to=["wandb"],                   # Ensure W&B logging is enabled
)

# Evaluation function (metrics)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1": report['macro avg']['f1-score'],
        "eval_loss": eval_pred[0].mean().item()
    }

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
print("Starting training...")
trainer.train()

# Extract training history for plotting
history = trainer.state.log_history

# Extract training and validation losses
train_losses = []
val_losses = []
train_steps = []
val_steps = []

for log in history:
    if 'loss' in log and 'eval_loss' not in log:
        train_losses.append(log['loss'])
        train_steps.append(log.get('step', len(train_losses)))
    if 'eval_loss' in log:
        val_losses.append(log['eval_loss'])
        val_steps.append(log.get('step', len(val_losses)))

# Plot training and validation loss
print("Plotting training and validation loss...")
plt.figure(figsize=(12, 5))

# Plot 1: Loss vs Steps
plt.subplot(1, 2, 1)
if train_steps and train_losses:
    plt.plot(train_steps, train_losses, label='Training Loss', marker='o', markersize=3)
if val_steps and val_losses:
    plt.plot(val_steps, val_losses, label='Validation Loss', marker='s', markersize=3)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs Steps')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss vs Epochs
plt.subplot(1, 2, 2)
epoch_train_losses = []
epoch_val_losses = []
epochs = []

# Group by epoch
current_epoch = 0
epoch_train = []
epoch_val = []

for log in history:
    if 'epoch' in log:
        if log['epoch'] != current_epoch:
            if epoch_train:
                epoch_train_losses.append(np.mean(epoch_train))
            if epoch_val:
                epoch_val_losses.append(np.mean(epoch_val))
            epochs.append(current_epoch)
            current_epoch = log['epoch']
            epoch_train = []
            epoch_val = []

    if 'loss' in log and 'eval_loss' not in log:
        epoch_train.append(log['loss'])
    if 'eval_loss' in log:
        epoch_val.append(log['eval_loss'])

# Add last epoch
if epoch_train:
    epoch_train_losses.append(np.mean(epoch_train))
if epoch_val:
    epoch_val_losses.append(np.mean(epoch_val))
if epochs:
    epochs.append(current_epoch)

if epochs and epoch_train_losses:
    plt.plot(epochs[:len(epoch_train_losses)], epoch_train_losses, label='Training Loss', marker='o')
if epochs and epoch_val_losses:
    plt.plot(epochs[:len(epoch_val_losses)], epoch_val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, f'training_validation_loss_{mode}.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.close()  # Close instead of show() for headless environments

print("\nTraining completed!")
print(f"Mode: {mode.upper()}")
print(f"Best model saved in: {training_args.output_dir}")
print(f"Loss plot saved in: {plot_path}")
