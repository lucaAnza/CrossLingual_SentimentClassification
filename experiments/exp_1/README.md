# Experiment 1 – Multiclass Classification

## How to set up a run

1. Install Conda
   👉 [Conda official webpage](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

   * [Linux installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)

2. Create the conda environment:

   ```bash
   conda env create -n NLP -f environment.yml
   ```

   *If you add new packages, export again:*

   ```bash
   conda env export > environment.yml
   ```

3. Create a `.env` file containing:

   ```bash
   DATASET_PATH=/path/to/amazon_reviews
   WANDB_API_KEY=your_wandb_key
   ```

---

## Experiment setup

### Dataset

- **Dataset:** Amazon Reviews (multi-domain sentiment dataset)  [(Dataset)](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi/data)
  - **Training set**: `1.200.000 samples`
  - **Validation set**: `30.000 samples `
  - **Test set**: `30.000 samples `

The full dataset is used to maximize performance and evaluate large-scale training behavior.

---

### Model and task

- **Base Model:** `distilbert/distilbert-base-uncased` [(Model)](https://huggingface.co/distilbert/distilbert-base-uncased)

* **Task:** **5-class multiclass classification**

  * VERY NEGATIVE
  * NEGATIVE
  * NEUTRAL
  * POSITIVE
  * VERY POSITIVE

The task treats sentiment labels as independent categories, without explicitly modeling their ordinal nature.

---

### Metrics

Both fine-grained and coarse-grained metrics are reported:

* **Accuracy (5-class)**
* **Macro F1-score (5-class)**
* **Accuracy (3-class)**
  (Negative / Neutral / Positive)
* **Macro F1-score (3-class)**

---

## Training configuration

Main hyperparameters:

* `learning_rate`: `2e-5`
* `warmup_ratio`: `0.1`
* `lr_scheduler_type`: `linear`
* `per_device_train_batch_size`: `64`
* `gradient_accumulation_steps`: `2`
  *(effective batch size = 128)*
* `per_device_eval_batch_size`: `128`
* `num_train_epochs`: `6`
* `weight_decay`: `0.01`
* Evaluation and checkpointing: **per epoch**
* Best model selection based on validation metrics

---

## Experimental results

At epoch **3**, the model achieves the following validation performance:

* **Accuracy (5-class):** **61.65%**
* **Macro F1 (5-class):** **0.614**
* **Accuracy (3-class):** **79.18%**
* **Macro F1 (3-class):** **0.746**

These results show that the model successfully learns high-level sentiment distinctions, particularly when collapsing labels into three sentiment polarities.

---

## Discussion and possible improvement

While multiclass classification provides reasonable performance, it exhibits a fundamental limitation: **all misclassifications are treated equally**. For example, predicting *POSITIVE* instead of *VERY POSITIVE* is penalized as much as predicting *VERY NEGATIVE*, despite the ordinal structure of star ratings.

This limitation becomes especially evident when distinguishing adjacent classes (e.g., 3 vs. 4 stars or 4 vs. 5 stars), where confusion remains frequent even with large-scale training.
