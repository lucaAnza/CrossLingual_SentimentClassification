#!/bin/bash
# Script to run training with the correct conda environment

# Activate the nlp-hf environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nlp-hf

# Change to the script directory
cd /home/aiello/dlwpt-code/SentAnal

# Ensure a writable temp directory on /data01
export TMPDIR=/data01/aiello/tmp
export TEMP=/data01/aiello/tmp
export TMP=/data01/aiello/tmp
mkdir -p /data01/aiello/tmp

# Force W&B to write to /data01
export WANDB_DIR=/data01/aiello/wandb
export WANDB_CACHE_DIR=/data01/aiello/wandb_cache
export WANDB_CONFIG_DIR=/data01/aiello/wandb_config
mkdir -p /data01/aiello/wandb /data01/aiello/wandb_cache /data01/aiello/wandb_config

# Force HF/transformers/datasets caches and matplotlib config to /data01
export HF_HOME=/data01/aiello/hf_home
export HF_DATASETS_CACHE=/data01/aiello/hf_datasets_cache
export TRANSFORMERS_CACHE=/data01/aiello/hf_transformers_cache
export MPLCONFIGDIR=/data01/aiello/mpl_cache
mkdir -p /data01/aiello/hf_home /data01/aiello/hf_datasets_cache /data01/aiello/hf_transformers_cache /data01/aiello/mpl_cache

# Run the training with the specified mode (default: multilingual)
MODE=${1:-multilingual}

echo "Running training in $MODE mode with nlp-hf environment..."
python main.py $MODE
