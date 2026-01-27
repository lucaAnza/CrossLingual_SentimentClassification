# Experiment 3 – Per-language Performance Ranking (Regression)

## How to setup a run

1. Install Conda
   - https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

2. Create the conda environment:

   ```bash
   conda env create -n NLP -f environment.yml
   ```

3. Create a `.env` file with:

   ```bash
   DATASET_PATH=/path/to/amazon_reviews
   WANDB_API_KEY=your_wandb_key
   LANGUAGES=en,de,fr
   SAMPLE_SIZE=200000
   RUN_PREFIX=exp3_regression
   MAX_LENGTH=64
   ```

---

## Experiment setup

### Dataset

- **Dataset:** Amazon Reviews (multi-domain sentiment dataset)
- We train and evaluate **separate models per language** using the same data size and hyperparameters.
- Each language uses a fixed number of training examples (`SAMPLE_SIZE`). Validation and test splits are sampled proportionally.

### Model and task

- **Backbone:** `distilbert-base-multilingual-cased`
- **Task:** **Regression**
  - Predicts a continuous sentiment score in `[0,1]`, mapped back to `[1,5]` for evaluation.

### Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Spearman correlation** (rank agreement)
- **Quadratic Weighted Kappa (QWK)** (ordinal agreement)
- **Rounded Accuracy (5-class)**
- **Rounded Accuracy (3-class)**

---

## Training configuration

- `learning_rate`: `2e-5`
- `num_train_epochs`: `8`
- `per_device_train_batch_size`: `32`
- `gradient_accumulation_steps`: `4` (effective batch size = 128)
- `per_device_eval_batch_size`: `128`
- `weight_decay`: `0.01`
- `lr_scheduler_type`: `cosine`
- `warmup_ratio`: `0.06`
- `max_grad_norm`: `1.0`
- Evaluation and checkpointing: **per epoch**
- Best model selection based on **RMSE**

---

## How to run

Train all languages listed in `LANGUAGES`:

```bash
python experiments/exp_3/experiment3.py
```

Evaluate all languages (loads latest checkpoint per language):

```bash
python experiments/exp_3/experiment3_evaluation.py
```

---

## Expected output

A per-language table of metrics. Rank languages by QWK (higher is better) or RMSE (lower is better) to identify the easiest and hardest languages.

