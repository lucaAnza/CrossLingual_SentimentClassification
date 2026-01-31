"""
Main script to run the complete pipeline:
1. Download/load dataset
2. Preprocess data (200k multilingual OR 200k same language)
3. Train model with plotting

Usage:
    python main.py [multilingual|single_language]
    
    Examples:
        python main.py multilingual      # Train with 200k reviews across languages
        python main.py single_language   # Train with 200k reviews from one language
        python main.py                   # Will prompt for choice
"""

import sys
import os
import argparse

# Force all caches/artifacts to /data01/aiello to avoid local disk usage
BASE_STORAGE_DIR = '/data01/aiello'
wandb_dir = os.path.join(BASE_STORAGE_DIR, 'wandb')
wandb_cache_dir = os.path.join(BASE_STORAGE_DIR, 'wandb_cache')
wandb_config_dir = os.path.join(BASE_STORAGE_DIR, 'wandb_config')
hf_home = os.path.join(BASE_STORAGE_DIR, 'hf_home')
hf_datasets_cache = os.path.join(BASE_STORAGE_DIR, 'hf_datasets_cache')
hf_transformers_cache = os.path.join(BASE_STORAGE_DIR, 'hf_transformers_cache')
mpl_config_dir = os.path.join(BASE_STORAGE_DIR, 'mpl_cache')
tmp_dir = os.path.join(BASE_STORAGE_DIR, 'tmp')

for path in (
    wandb_dir,
    wandb_cache_dir,
    wandb_config_dir,
    hf_home,
    hf_datasets_cache,
    hf_transformers_cache,
    mpl_config_dir,
    tmp_dir,
):
    os.makedirs(path, exist_ok=True)

os.environ.setdefault('WANDB_DIR', wandb_dir)
os.environ.setdefault('WANDB_CACHE_DIR', wandb_cache_dir)
os.environ.setdefault('WANDB_CONFIG_DIR', wandb_config_dir)
os.environ.setdefault('HF_HOME', hf_home)
os.environ.setdefault('HF_DATASETS_CACHE', hf_datasets_cache)
os.environ.setdefault('TRANSFORMERS_CACHE', hf_transformers_cache)
os.environ.setdefault('MPLCONFIGDIR', mpl_config_dir)
os.environ.setdefault('TMPDIR', tmp_dir)
os.environ.setdefault('TEMP', tmp_dir)
os.environ.setdefault('TMP', tmp_dir)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train sentiment analysis model')
parser.add_argument('mode', nargs='?', choices=['multilingual', 'single_language'],
                    help='Training mode: multilingual (200k across languages) or single_language (200k from one language)')
args = parser.parse_args()

# Get mode from command line or prompt user
mode = args.mode
if mode is None:
    print("\n" + "=" * 60)
    print("Please choose training mode:")
    print("=" * 60)
    print("1. multilingual - 200k reviews split across different languages")
    print("2. single_language - 200k reviews from the same language")
    print("\nYou can also run: python main.py [multilingual|single_language]")
    
    choice = input("\nEnter choice (1 or 2, or 'multilingual'/'single_language'): ").strip().lower()
    
    if choice in ['1', 'multilingual']:
        mode = 'multilingual'
    elif choice in ['2', 'single_language', 'single']:
        mode = 'single_language'
    else:
        print("Invalid choice. Defaulting to 'multilingual'")
        mode = 'multilingual'

print("\n" + "=" * 60)
print(f"Selected mode: {mode.upper()}")
print("=" * 60)

# Step 1: Load dataset (optional, preprocessing will load it if needed)
print("\n" + "=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)
try:
    from dataset import dataset, val_dataset as val_dataset_raw, test_dataset
    print("Dataset module loaded successfully")
except:
    print("Note: Dataset will be loaded during preprocessing")

# Step 2: Preprocess data
print("\n" + "=" * 60)
print("STEP 2: Preprocessing Data")
print("=" * 60)
import datapreprocessing

# Preprocess with the selected mode (without tokenization)
train_dataset, val_dataset, model_name, _, language_info = datapreprocessing.preprocess_data(mode)
if language_info:
    print(f"Language selection: {language_info}")

# Step 3: Tokenize and Train model
print("\n" + "=" * 60)
print("STEP 3: Tokenizing Data and Training Model")
print("=" * 60)

# Workaround for scipy import issue: patch scipy before importing transformers
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to prevent scipy import issues
try:
    # Create a mock scipy.stats module to prevent import errors
    class MockScipyStats:
        def pearsonr(self, *args, **kwargs):
            raise NotImplementedError("scipy.stats.pearsonr not available")
        def spearmanr(self, *args, **kwargs):
            raise NotImplementedError("scipy.stats.spearmanr not available")
    
    # Only patch if scipy import would fail
    import importlib.util
    spec = importlib.util.find_spec("scipy.stats")
    if spec is None:
        import types
        mock_module = types.ModuleType("scipy.stats")
        mock_module.pearsonr = lambda *args, **kwargs: None
        mock_module.spearmanr = lambda *args, **kwargs: None
        sys.modules["scipy.stats"] = mock_module
except:
    pass

# Now try to import transformers components
# Import Trainer separately to handle scipy import issues
Trainer = None
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
    # Try to import Trainer - this may fail due to scipy
    try:
        # Set environment to avoid some imports
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        # Try lazy import by patching sys.modules temporarily
        import sys
        original_scipy = sys.modules.get('scipy', None)
        try:
            from transformers import Trainer
            print("Successfully imported Trainer")
        except (RuntimeError, ImportError) as e:
            if "scipy" in str(e).lower() or "GLIBCXX" in str(e) or "libstdc++" in str(e).lower():
                print("Warning: Cannot import Trainer due to scipy/libstdc++ compatibility issue.")
                print("Will use a custom PyTorch training loop instead.")
                Trainer = None
            else:
                raise
    except Exception as e:
        print(f"Error importing Trainer: {e}")
        Trainer = None
except Exception as e:
    print(f"Error importing transformers: {e}")
    raise

import numpy as np
try:
    from sklearn.metrics import accuracy_score, classification_report
except ImportError:
    # Use custom implementations if sklearn not available
    def accuracy_score(y_true, y_pred):
        return np.mean(np.array(y_true) == np.array(y_pred))
    def classification_report(y_true, y_pred, output_dict=False, zero_division=0):
        # Simple classification report
        from collections import Counter
        acc = accuracy_score(y_true, y_pred)
        if output_dict:
            return {
                'macro avg': {'precision': acc, 'recall': acc, 'f1-score': acc}
            }
        return f"Accuracy: {acc}"

import matplotlib.pyplot as plt

# Load tokenizer and apply tokenization
print(f"Loading tokenizer: {model_name}")
try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Trying alternative approach...")
    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
        raise RuntimeError("Cannot load tokenizer due to library compatibility issues.")

# Tokenization function
def tokenize_function(examples):
    texts = []
    for i in range(len(examples['review_title'])):
        title = examples['review_title'][i] if examples['review_title'][i] else ""
        body = examples['review_body'][i] if examples['review_body'][i] else ""
        combined = f"{title} {body}".strip()
        texts.append(combined)
    return tokenizer(texts, truncation=True, padding='max_length', max_length=512)

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
print(f"\nLoading model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,  # Three classes: "Bad" (0), "Neutral" (1), "Good" (2)
)

# Set training arguments with mode-specific output directories
BASE_STORAGE_DIR = '/data01/aiello'
output_dir = os.path.join(BASE_STORAGE_DIR, f'results_{mode}')
logging_dir = os.path.join(BASE_STORAGE_DIR, f'logs_{mode}')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

print(f"Results will be saved to: {output_dir}")
print(f"Logs will be saved to: {logging_dir}")

# Try both parameter names for compatibility with different transformers versions
# Note: save_strategy must match eval_strategy when load_best_model_at_end=True
try:
    training_args = TrainingArguments(
        output_dir=output_dir,                  # directory to save results (mode-specific)
        eval_strategy="epoch",                  # evaluate every epoch (newer versions)
        save_strategy="epoch",                  # save every epoch (must match eval_strategy)
        logging_dir=logging_dir,                # directory for logs (mode-specific)
        logging_steps=500,                      # log every 500 steps
        num_train_epochs=3,                     # number of epochs
        per_device_train_batch_size=16,         # batch size for training
        per_device_eval_batch_size=64,          # batch size for evaluation
        weight_decay=0.01,                      # regularization to avoid overfitting
        save_total_limit=2,                     # keep only the last 2 checkpoints
        load_best_model_at_end=True,            # load best model at end of training
        metric_for_best_model="eval_loss",      # metric to determine best model
        greater_is_better=False,                # lower loss is better
    )
except TypeError:
    # Fallback for older versions that use evaluation_strategy
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",        # evaluate every epoch (older versions)
            save_strategy="epoch",              # save every epoch (must match evaluation_strategy)
            logging_dir=logging_dir,
            logging_steps=500,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    except TypeError:
        # Even older versions might not have save_strategy, use save_steps but disable load_best_model_at_end
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            logging_dir=logging_dir,
            logging_steps=500,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            save_steps=10_000,
            save_total_limit=2,
            load_best_model_at_end=False,       # Disable if save_strategy not available
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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
        "f1": report['macro avg']['f1-score']
    }

# Use Trainer if available, otherwise use custom training loop
if Trainer is not None:
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    print("Starting training with Hugging Face Trainer...")
    trainer.train()

    # Extract training history for plotting
    history = trainer.state.log_history
else:
    # Custom training loop using PyTorch directly
    print("Using custom PyTorch training loop (Trainer not available due to library issues)...")
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=training_args.weight_decay)
    
    # Setup scheduler
    num_training_steps = len(train_loader) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # Training history
    history = []
    train_losses = []
    val_losses = []
    global_step = 0
    
    # Training loop
    for epoch in range(training_args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
        
        # Training phase
        model.train()
        epoch_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % training_args.logging_steps == 0:
                avg_loss = epoch_train_loss / (batch_idx + 1)
                print(f"Step {global_step}, Loss: {avg_loss:.4f}")
                history.append({'step': global_step, 'loss': avg_loss, 'epoch': epoch})
                train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        history.append({
            'step': global_step,
            'epoch': epoch,
            'eval_loss': avg_val_loss,
            'eval_accuracy': val_accuracy
        })
        val_losses.append(avg_val_loss)
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:  # Save every epoch
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    print("\nTraining completed!")

# Extract training and validation losses
if Trainer is not None:
    # History from Trainer
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
else:
    # History from custom training loop (already extracted during training)
    # train_losses and val_losses are already populated
    train_steps = [log.get('step', i) for i, log in enumerate(history) if 'loss' in log and 'eval_loss' not in log]
    val_steps = [log.get('step', i) for i, log in enumerate(history) if 'eval_loss' in log]

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
plot_path = f'{output_dir}/training_validation_loss_{mode}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.close()  # Close instead of show() for headless environments

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)
print(f"Mode: {mode.upper()}")
print(f"Best model saved in: {training_args.output_dir}")
print(f"Loss plot saved in: {plot_path}")
