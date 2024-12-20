# Commands related to general project setup
SEED=42

# Commands related to loading the dataset
PRIMEVUL_PAIRED_TRAIN_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl"
PRIMEVUL_PAIRED_TEST_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test_paired.jsonl"
PRIMEVUL_PAIRED_VALID_DATA_FILE="/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired.jsonl"

# Commands related to the models
HUGGINGFACE_EMBEDDER_NAME="microsoft/codebert-base"
OUTPUT_DIR="/mnt/isgnas/home/anl31/documents/code/ai-vul-detect-pairwise-learning-to-rank/model_checkpoints"

# Commands related to training
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
NB_EPOCHS=5

python ./src/main.py \
    --seed=${SEED} \
    --primevul_paired_train_data_file=${PRIMEVUL_PAIRED_TRAIN_DATA_FILE} \
    --primevul_paired_test_data_file=${PRIMEVUL_PAIRED_TEST_DATA_FILE} \
    --primevul_paired_valid_data_file=${PRIMEVUL_PAIRED_VALID_DATA_FILE} \
    --huggingface_embedder_name=${HUGGINGFACE_EMBEDDER_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --do_train \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --nb_epochs=${NB_EPOCHS}