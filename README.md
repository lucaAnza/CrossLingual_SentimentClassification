# CrossLingual_SentimentClassification

## Overview
This project focuses on **benchmarking and comparing sentiment analysis models** across different **languages** and **datasets**.  
The goal is to evaluate and understand the performance differences between **cross-lingual** and **monolingual** approaches in sentiment classification tasks.

We explore multiple datasets covering various linguistic domains and cultural contexts, aiming to identify how well pretrained language models generalize sentiment understanding beyond their training language.

## Experiments

### 1. Fine-Tuning for multiclass classification using **Distilbert-base-multilingual-cased**

The core training script is located in [`experiment1.py`](/experiments/exp_1/experiment1.py).

![Experiment1](src/exp1.png)

In this file, we perform **fine-tuning** using:
- **Dataset:** Amazon Reviews (multi-domain sentiment dataset)  [(Dataset)](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi/data)
- **Base Model:** `distilbert/distilbert-base-uncased` [(Model)](https://huggingface.co/distilbert/distilbert-base-uncased)


### 2. Fine-Tuning for regression using **Distilbert-base-multilingual-cased**

All settings are the same of [Experiment1](#1.fine-tuningformulticlassclassificationusing**Distilbert-base-multilingual-cased**) except for the **preprocessing** and **evaluation**.
