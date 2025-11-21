# Experiment1

## How to setup a run

1. Install conda [Conda official webpage](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
   - [Linux installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
2. Create a conda environment `conda env create -n NLP -f environment.yml`
   - If you add some package export again `conda env export > environment.yml`

## Experiments results

The model reached **56% accuracy**, indicating that it captures some sentiment patterns but struggles to generalize across the full range of classes (from very negative to very positive).
A potential reason is that the star ratings are ordinal — they have a natural numerical order (1 to 5) — but our model treats them as independent categorical labels.

## Possible Improvement
A promising next step is to reframe the task as a regression problem instead of multi-class classification.

Using a linear regression head (predicting a continuous star value between 1–5) could better model the ordinal relationships between sentiment levels.
Evaluation could then use **Mean Squared Error (MSE)** or **Spearman correlation** instead of accuracy.
