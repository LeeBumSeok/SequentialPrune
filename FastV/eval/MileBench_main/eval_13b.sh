# export TOKENIZERS_PARALLELISM=false
# GEN_SCRIPT_PATH=generate.py
# EVAL_SCRIPT_PATH=evaluate.py
# DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
# CUDA_LAUNCH_BLOCKING=0
# gpu_num=1
# ATTENTION_RANKS=(0.5 0.75 0.9)
# AGG_LAYERS=(2 3 5)
# for model in llava-v1.5-13b; do
#     if [ "$model" == "llava-v1.5-7b" ]; then
#         MODEL_CONFIG_PATH=configs/model_configs.yaml
#     else
#         MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
#     fi

#     for attention_rank in "${ATTENTION_RANKS[@]}"; do
#         for agg_layer in "${AGG_LAYERS[@]}"; do
#             for dataset_name in WikiVQA; do
#                 BATCH_SIZE=1
#                 OUTPUT_DIR=outputs/${model}_rank_${attention_rank}_layer_${agg_layer}
#                 LOG_DIR=logs/${model}_rank_${attention_rank}_layer_${agg_layer}
                
#                 mkdir -p ${LOG_DIR}
#                 mkdir -p ${OUTPUT_DIR}
                
#                 # Start generating
#                 python ${GEN_SCRIPT_PATH} \
#                     --data_dir ${DATA_DIR} \
#                     --dataset_name ${dataset_name}  \
#                     --model_name ${model} \
#                     --output_dir ${OUTPUT_DIR} \
#                     --batch-image ${BATCH_SIZE} \
#                     --model_configs ${MODEL_CONFIG_PATH} \
#                     --overwrite \
#                     --combine_image 1 \
#                     --use-fast-v \
#                     --fast-v-inplace \
#                     --fast-v-sys-length 36 \
#                     --fast-v-image-token-length 576 \
#                     --fast-v-attention-rank ${attention_rank} \
#                     --fast-v-agg-layer ${agg_layer}

#                 # Start evaluating
#                 python ${EVAL_SCRIPT_PATH} \
#                     --data-dir ${DATA_DIR} \
#                     --dataset ${dataset_name} \
#                     --result-dir ${OUTPUT_DIR} 
#             done
#         done
#     done
# done

# export TOKENIZERS_PARALLELISM=false
# GEN_SCRIPT_PATH=generate.py
# EVAL_SCRIPT_PATH=evaluate.py
# DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
# CUDA_LAUNCH_BLOCKING=0
# gpu_num=1
# ATTENTION_RANKS=(0.5 0.75 0.9)
# AGG_LAYERS=(2 3 5)
# for model in llava-v1.5-13b; do
#     if [ "$model" == "llava-v1.5-7b" ]; then
#         MODEL_CONFIG_PATH=configs/model_configs.yaml
#     else
#         MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
#     fi

#     for attention_rank in "${ATTENTION_RANKS[@]}"; do
#         for agg_layer in "${AGG_LAYERS[@]}"; do
#             for dataset_name in WikiVQA; do
#                 BATCH_SIZE=1
#                 OUTPUT_DIR=outputs_inplace/${model}_rank_${attention_rank}_layer_${agg_layer}_seq
#                 LOG_DIR=logs_inplace/${model}_rank_${attention_rank}_layer_${agg_layer}_seq
                
#                 mkdir -p ${LOG_DIR}
#                 mkdir -p ${OUTPUT_DIR}
                
#                 # Start generating
#                 python ${GEN_SCRIPT_PATH} \
#                     --data_dir ${DATA_DIR} \
#                     --dataset_name ${dataset_name}  \
#                     --model_name ${model} \
#                     --output_dir ${OUTPUT_DIR} \
#                     --batch-image ${BATCH_SIZE} \
#                     --model_configs ${MODEL_CONFIG_PATH} \
#                     --overwrite \
#                     --combine_image 1 \
#                     --use-fast-v \
#                     --fast-v-inplace \
#                     --fast-v-sys-length 36 \
#                     --fast-v-image-token-length 576 \
#                     --fast-v-attention-rank ${attention_rank} \
#                     --fast-v-agg-layer ${agg_layer} \
#                     --fast-v-sequential-prune

#                 # Start evaluating
#                 python ${EVAL_SCRIPT_PATH} \
#                     --data-dir ${DATA_DIR} \
#                     --dataset ${dataset_name} \
#                     --result-dir ${OUTPUT_DIR} 
#             done
#         done
#     done
# done


# export TOKENIZERS_PARALLELISM=false
# GEN_SCRIPT_PATH=generate.py
# EVAL_SCRIPT_PATH=evaluate.py
# DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
# CUDA_LAUNCH_BLOCKING=0
# gpu_num=1
# for model in llava-v1.5-13b; do
#     if [ "$model" == "llava-v1.5-7b" ]; then
#         MODEL_CONFIG_PATH=configs/model_configs.yaml
#     else
#         MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
#     fi

#     for dataset_name in WikiVQA; do
#         BATCH_SIZE=1
#         OUTPUT_DIR=outputs/llava-v1.5-13b
#         LOG_DIR=logs/
        
#         mkdir -p ${LOG_DIR}
#         mkdir -p ${OUTPUT_DIR}
        
#         # Start generating
#         # python ${GEN_SCRIPT_PATH} \
#         #     --data_dir ${DATA_DIR} \
#         #     --dataset_name ${dataset_name}  \
#         #     --model_name ${model} \
#         #     --output_dir ${OUTPUT_DIR} \
#         #     --batch-image ${BATCH_SIZE} \
#         #     --model_configs ${MODEL_CONFIG_PATH} \
#         #     --overwrite \
#         #     --combine_image 1

#         python ${EVAL_SCRIPT_PATH} \
#                     --data-dir ${DATA_DIR} \
#                     --dataset ${dataset_name} \
#                     --result-dir ${OUTPUT_DIR} 
#     done
# done