export TOKENIZERS_PARALLELISM=false
GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
CUDA_LAUNCH_BLOCKING=0
gpu_num=1
ATTENTION_RANKS=(0.5 0.75 0.9)
AGG_LAYERS=(1 2 3 4 5 6)
PRUNE_STEPS=(5 10 15 20 25)
for model in llava-v1.5-13b; do
    if [ "$model" == "llava-v1.5-7b" ]; then
        MODEL_CONFIG_PATH=configs/model_configs.yaml
    else
        MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
    fi

    for prune_step in "${PRUNE_STEPS[@]}"; do
        for attention_rank in "${ATTENTION_RANKS[@]}"; do
            for agg_layer in "${AGG_LAYERS[@]}"; do
                for dataset_name in Rouge_OCR_VQA; do
                    BATCH_SIZE=1
                    OUTPUT_DIR=outputs_ablation_1/${model}_rank_${attention_rank}_layer_${agg_layer}_step_${prune_step}_seq
                    # LOG_DIR=logs_ablation_1/${model}_rank_${attention_rank}_layer_${agg_layer}_seq
                    
                    mkdir -p ${OUTPUT_DIR}
                    
                    # Start generating
                    python ${GEN_SCRIPT_PATH} \
                        --data_dir ${DATA_DIR} \
                        --dataset_name ${dataset_name}  \
                        --model_name ${model} \
                        --output_dir ${OUTPUT_DIR} \
                        --batch-image ${BATCH_SIZE} \
                        --model_configs ${MODEL_CONFIG_PATH} \
                        --overwrite \
                        --combine_image 1 \
                        --use-fast-v \
                        --fast-v-inplace \
                        --fast-v-sys-length 36 \
                        --fast-v-image-token-length 576 \
                        --fast-v-attention-rank ${attention_rank} \
                        --fast-v-agg-layer ${agg_layer} \
                        --fast-v-sequential-prune \
                        --prune-step ${prune_step}

                    # Start evaluating
                    python ${EVAL_SCRIPT_PATH} \
                        --data-dir ${DATA_DIR} \
                        --dataset ${dataset_name} \
                        --result-dir ${OUTPUT_DIR} 
                done
            done
        done
    done
done