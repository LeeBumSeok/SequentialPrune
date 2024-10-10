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
                    python score.py \
                        --result-dir outputs_ablation_1 \
                        --models ${model}_rank_${attention_rank}_layer_${agg_layer}_step_${prune_step}_seq
                done
            done
        done
    done
done