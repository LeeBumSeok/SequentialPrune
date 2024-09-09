#!/bin/bash
TORCH_USE_CUDA_DSA=1
CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
MODEL_CONFIG_PATH=configs/model_configs.yaml
gpu_num=1
for model in llava-v1.5-7b; do
    for dataset_name in DocVQA; do
        BATCH_SIZE=1
        mkdir -p logs/${model}
        # Start generating
        python ${GEN_SCRIPT_PATH} \
            --data_dir ${DATA_DIR} \
            --dataset_name ${dataset_name}  \
            --model_name ${model} \
            --output_dir outputs \
            --batch-image ${BATCH_SIZE} \
            --model_configs ${MODEL_CONFIG_PATH} \
            --overwrite \
            --combine_image 1 \
            --use-fast-v \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length 576 \
            --fast-v-attention-rank 72 \
            --fast-v-agg-layer 2 
        # Start evaluating
        # python ${EVAL_SCRIPT_PATH} \
        #     --data-dir ${DATA_DIR} \
        #     --dataset ${dataset_name} \
        #     --result-dir outputs/${model}
    done
done