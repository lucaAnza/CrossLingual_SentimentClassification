# Experiment2

## How to setup a run

1. Install conda [Conda official webpage](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
   - [Linux installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
2. Create a conda environment `conda env create -n NLP -f environment.yml`
   - If you add some package export again `conda env export > environment.yml`

## Experiments setup

* **Training set**: 300,000 samples
* **Validation set**: 30,000 samples
* **Test set**: 30,000 samples

**Training configuration:**

* `learning_rate`: 2e-5
* `per_device_train_batch_size`: 64
* `per_device_eval_batch_size`: 64
* `num_train_epochs`: 8
  *(although 2–5 epochs are usually sufficient for fine-tuning, a higher number was used to ensure convergence)*
* `weight_decay`: 0.01
  *(L2 regularization to penalize large weights and improve generalization)*

## Experiments results

The regression model reaches **53.2%** accuracy when predictions are rounded to the nearest star, showing it captures sentiment reasonably well but still struggles with fine-grained distinctions between adjacent ratings (e.g., 4 vs 5). Since the task is ordinal, regression is a more natural formulation than multiclass classification: predicting a continuous value encourages the model to learn that errors of 1 star are less severe than errors of 3 stars. This is reflected by **MAE = 0.61** and **RMSE = 0.83**. When compressing ratings into 3 sentiment bins, performance increases to **73.9%**, indicating the model is often correct on the broader polarity even when the exact star is off by one.

| Metric                         | Value      |
|--------------------------------|------------|
| Eval loss                      | 0.6896     |
| Mean Absolute Error (MAE)      | 0.6096     |
| Root Mean Squared Error (RMSE) | 0.8304     |
| Accuracy (5 classes)           | 0.5320     |
| Accuracy (3 classes)           | 0.7388     |


$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \hat{y}_i \right|$
* **MAE ≈ 0.61** ⇒ on average, the model’s predictions are off by about **0.6 stars**.

$\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2 }$
* **RMSE ≈ 0.83** ⇒ there are some **larger errors**, because RMSE penalizes bigger mistakes more heavily than MAE.



