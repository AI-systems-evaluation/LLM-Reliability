#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "=========================================="
echo " Starting LLM-Reliability Pipeline "
echo "=========================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Datasets
PROMPT_DATASET="AI-systems-evaluation/paraphrased_prompt"
RESPONSE_DATASET="AI-systems-evaluation/paraphrased_prompt_response"
SPLIT="train"

# Directories
EMBEDDINGS_DIR="../embeddings"
INITIAL_DIR="../resample2k3k-initial"
OUTPUT_DIR="../resample2k3k"
REL_INDICES_DIR="../outputs-rel_indices"

# Sampling
SAMPLE_SIZE=2000
SEED_START=1001
SEED_COUNT=16

# K-Means K values
K_VALUES=(10 20 30 40)
# K value used for the final Q-Model inference
QMODEL_K=30

# Anchor Expert for Q-Model (must be one of: MMLU-PRO, GPQA, MATH)
ANCHOR_EXPERT="MMLU-PRO"

# -----------------------------------------------------------------------------
# Step 1: Prepare Shape Descriptors (Embeddings + Initial Sampling + Clustering)
# -----------------------------------------------------------------------------
echo ""
echo "[Step 1/4] Preparing shape descriptors..."
python prepare_shape_descriptors.py \
    --prompt_dataset "${PROMPT_DATASET}" \
    --response_dataset "${RESPONSE_DATASET}" \
    --split "${SPLIT}" \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --initial_dir "${INITIAL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample_size ${SAMPLE_SIZE} \
    --seed_start ${SEED_START} \
    --seed_count ${SEED_COUNT} \
    --k_values "${K_VALUES[@]}"


# -----------------------------------------------------------------------------
# Step 2: Compute Reliability Indices for each K
# -----------------------------------------------------------------------------
echo ""
echo "[Step 2/4] Computing reliability indices..."
for K in "${K_VALUES[@]}"; do
    echo "  -> Computing for K=${K}"
    python run_algorithm1.py compute \
        --k ${K} \
        --prefix ".." \
        --outdir "${REL_INDICES_DIR}" \
        --seed_start ${SEED_START} \
        --seed_count ${SEED_COUNT}
done


# -----------------------------------------------------------------------------
# Step 3: Aggregate Reliability Indices
# -----------------------------------------------------------------------------
echo ""
echo "[Step 3/4] Aggregating reliability indices..."
python run_algorithm1.py aggregate \
    --k_values "${K_VALUES[@]}" \
    --outdir "${REL_INDICES_DIR}"


# -----------------------------------------------------------------------------
# Step 4: Run Q-Model Inference for Bayesian alignement inference
# -----------------------------------------------------------------------------
echo ""
echo "[Step 4/4] Running Q-Model inference..."
python run_Qmodel.py \
    --anchor_E_name "${ANCHOR_EXPERT}" \
    --k ${QMODEL_K} \
    --prefix "${REL_INDICES_DIR}"

echo ""
echo "=========================================="
echo " Pipeline execution completed successfully! "
echo "=========================================="
