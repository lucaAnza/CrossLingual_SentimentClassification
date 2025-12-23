# Experiment1

## How to setup a run

1. Install conda [Conda official webpage](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

   * [Linux installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
2. Create a conda environment `conda env create -n NLP -f environment.yml`

   * If you add some package export again `conda env export > environment.yml`

## Experiments setup

### Dataset

- **Dataset:** Amazon Reviews (multi-domain sentiment dataset)  [(Dataset)](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi/data)
  - **Training set**: `150,000 samples`
  - **Validation set**: `25,000 samples `
  - **Test set**: `25,000 samples `

### Model and task

- **Backbone**: `distilbert/distilbert-base-uncased` [(Model)](https://huggingface.co/distilbert/distilbert-base-uncased)
- **Task**: **5-class multiclass classification** (VERY NEGATIVE → VERY POSITIVE)

### Metrics
- Accuracy

## Training configuration

* `learning_rate`: 2e-5
* `per_device_train_batch_size`: 32
* `per_device_eval_batch_size`: 32
* `num_train_epochs`: 5
* `weight_decay`: 0.01 *(L2 regularization to penalize large weights and improve generalization)*

## Experiments results

The model reached **~56% accuracy**, meaning it learns useful sentiment patterns but still struggles to separate adjacent star ratings consistently (e.g., 3 vs 4, 4 vs 5). A key limitation is that **multiclass classification treats the 5 labels as independent categories**, while star ratings are **ordinal** (1 < 2 < 3 < 4 < 5): predicting 1 instead of 2 is “less wrong” than predicting 1 instead of 5, but standard classification accuracy does not capture this distance-aware structure.

## Possible improvement

A promising next step is to **reframe the task as regression** (as done in Experiment2): predict a continuous score in ([1,5]) so the model can naturally learn that small star shifts are less severe than large ones. Evaluation can then rely on **MAE/RMSE**.
