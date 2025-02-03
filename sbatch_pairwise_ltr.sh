#!/bin/bash
# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=isg
#SBATCH --constraint=vram80+ 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:4
#SBATCH --job-name=ltr
#SBATCH -o learning_to_rank_bigbird_log-%N.%j.out
#SBATCH --time=48:0:0

# MODIFY THESE OPTIONS
# **************************

source /mnt/isgnas/home/anl31/miniconda3/etc/profile.d/conda.sh
conda activate ai-vul-detect-pair-ltr

# Commands related to general project setup
SEED=42

# Commands related to loading the dataset
PRIMEVUL_PAIRED_TRAIN_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl"
PRIMEVUL_PAIRED_VALID_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired.jsonl"
PRIMEVUL_SINGLE_INPUT_VALID_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid.jsonl"
PRIMEVUL_SINGLE_INPUT_TEST_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test.jsonl"
PRIMEVUL_PAIRED_TEST_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test_paired.jsonl"

# PRIMEVUL_PAIRED_TRAIN_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired_16pts.jsonl"
# PRIMEVUL_PAIRED_VALID_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired_16pts.jsonl"
# PRIMEVUL_SINGLE_INPUT_VALID_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_16pts.jsonl"
# PRIMEVUL_SINGLE_INPUT_TEST_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_16pts.jsonl"
# PRIMEVUL_PAIRED_TEST_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test_paired.jsonl"

# Commands related to the models
# HUGGINGFACE_EMBEDDER_NAME="google/bigbird-roberta-large"
HUGGINGFACE_EMBEDDER_NAME="microsoft/codebert-base"
OUTPUT_DIR="/mnt/isgnas/home/anl31/documents/code/ai-vul-detect-pairwise-learning-to-rank/model_checkpoints"

# Commands related to training
LEARNING_RATE=5e-5
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=128
NB_EPOCHS=10

python ./src/main.py \
    --seed=${SEED} \
    --primevul_paired_train_data_file=${PRIMEVUL_PAIRED_TRAIN_DATA_FILE} \
    --primevul_paired_test_data_file=${PRIMEVUL_PAIRED_TEST_DATA_FILE} \
    --primevul_paired_valid_data_file=${PRIMEVUL_PAIRED_VALID_DATA_FILE} \
    --primevul_single_input_valid_dataset=${PRIMEVUL_SINGLE_INPUT_VALID_DATA_FILE} \
    --primevul_single_input_test_dataset=${PRIMEVUL_SINGLE_INPUT_TEST_DATA_FILE} \
    --huggingface_embedder_name=${HUGGINGFACE_EMBEDDER_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --do_train \
    --do_test \
    --evaluate_during_training \
    --learning_rate=${LEARNING_RATE} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --nb_epochs=${NB_EPOCHS} \
    --max_patience=2