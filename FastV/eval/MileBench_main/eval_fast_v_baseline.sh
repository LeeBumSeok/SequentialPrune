#!/bin/bash
export TOKENIZERS_PARALLELISM=false
GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
CUDA_LAUNCH_BLOCKING=0
gpu_num=1

# ATTENTION_RANKS=(0.75)
# AGG_LAYERS=(3)
# for model in llava-v1.5-7b; do
#     if [ "$model" == "llava-v1.5-7b" ]; then
#         MODEL_CONFIG_PATH=configs/model_configs.yaml
#     else
#         MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
#     fi

#     for attention_rank in "${ATTENTION_RANKS[@]}"; do
#         for agg_layer in "${AGG_LAYERS[@]}"; do
#             for dataset_name in ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA WebQA WikiVQA nuscenes; do
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
#                     --fast-v-sys-length 36 \
#                     --fast-v-image-token-length 576 \
#                     --fast-v-attention-rank ${attention_rank} \
#                     --fast-v-agg-layer ${agg_layer} \
#                     > ${LOG_DIR}/${dataset_name}.log
                
#                 # Start evaluating
#                 python ${EVAL_SCRIPT_PATH} \
#                     --data-dir ${DATA_DIR} \
#                     --dataset ${dataset_name} \
#                     --result-dir ${OUTPUT_DIR} \
#                     >> ${LOG_DIR}/${dataset_name}.log
#             done
#             python score.py \
#                 --result-dir outputs \
#                 --models ${model}_rank_${attention_rank}_layer_${agg_layer}
#         done
#     done
# done


ATTENTION_RANKS=(0.75)
AGG_LAYERS=(5)
for model in llava-v1.5-7b; do
    if [ "$model" == "llava-v1.5-7b" ]; then
        MODEL_CONFIG_PATH=configs/model_configs.yaml
    else
        MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
    fi

    for attention_rank in "${ATTENTION_RANKS[@]}"; do
        for agg_layer in "${AGG_LAYERS[@]}"; do
            for dataset_name in Rouge_OCR_VQA ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation IEdit MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA WebQA WikiVQA nuscenes; do
                BATCH_SIZE=1
                OUTPUT_DIR=outputs/${model}_rank_${attention_rank}_layer_${agg_layer}
                LOG_DIR=logs/${model}_rank_${attention_rank}_layer_${agg_layer}
                
                mkdir -p ${LOG_DIR}
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
                    --fast-v-sys-length 36 \
                    --fast-v-image-token-length 576 \
                    --fast-v-attention-rank ${attention_rank} \
                    --fast-v-agg-layer ${agg_layer} \
                    > ${LOG_DIR}/${dataset_name}.log
                
                # Start evaluating
                python ${EVAL_SCRIPT_PATH} \
                    --data-dir ${DATA_DIR} \
                    --dataset ${dataset_name} \
                    --result-dir ${OUTPUT_DIR} \
                    >> ${LOG_DIR}/${dataset_name}.log
            done
            python score.py \
                --result-dir outputs \
                --models ${model}_rank_${attention_rank}_layer_${agg_layer}
        done
    done
done

bash eval_baseline.sh

ATTENTION_RANKS=(0.9)
AGG_LAYERS=(2 3 5)
for model in llava-v1.5-7b; do
    if [ "$model" == "llava-v1.5-7b" ]; then
        MODEL_CONFIG_PATH=configs/model_configs.yaml
    else
        MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
    fi

    for attention_rank in "${ATTENTION_RANKS[@]}"; do
        for agg_layer in "${AGG_LAYERS[@]}"; do
            for dataset_name in Rouge_OCR_VQA ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation IEdit MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA WebQA WikiVQA nuscenes; do
                BATCH_SIZE=1
                OUTPUT_DIR=outputs/${model}_rank_${attention_rank}_layer_${agg_layer}
                LOG_DIR=logs/${model}_rank_${attention_rank}_layer_${agg_layer}
                
                mkdir -p ${LOG_DIR}
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
                    --fast-v-sys-length 36 \
                    --fast-v-image-token-length 576 \
                    --fast-v-attention-rank ${attention_rank} \
                    --fast-v-agg-layer ${agg_layer} \
                    > ${LOG_DIR}/${dataset_name}.log
                
                # Start evaluating
                python ${EVAL_SCRIPT_PATH} \
                    --data-dir ${DATA_DIR} \
                    --dataset ${dataset_name} \
                    --result-dir ${OUTPUT_DIR} \
                    >> ${LOG_DIR}/${dataset_name}.log
            done
            python score.py \
                --result-dir outputs \
                --models ${model}_rank_${attention_rank}_layer_${agg_layer}
        done
    done
done
# Rouge_OCR_VQA ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation IEdit MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA WebQA WikiVQA nuscenes