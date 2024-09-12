#!/bin/bash
export TOKENIZERS_PARALLELISM=false
GEN_SCRIPT_PATH=time_eval.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
MODEL_CONFIG_PATH=configs/model_configs.yaml
CUDA_LAUNCH_BLOCKING=0
gpu_num=1
for model in llava-v1.5-7b; do
    for dataset_name in ALFRED; do
        BATCH_SIZE=1
        mkdir -p logs/${model}
        # Start generating
        python ${GEN_SCRIPT_PATH} \
            --data_dir ${DATA_DIR} \
            --dataset_name ${dataset_name}  \
            --model_name ${model} \
            --output_dir test \
            --batch-image ${BATCH_SIZE} \
            --model_configs ${MODEL_CONFIG_PATH} \
            --overwrite \
            --combine_image 1 
            
        # Start evaluating
        # python ${EVAL_SCRIPT_PATH} \
        #     --data-dir ${DATA_DIR} \
        #     --dataset ${dataset_name} \
        #     --result-dir test/${model} \
        #     >> logs/${model}/${dataset_name}.log
    done
done


# python score.py \
#     --result-dir outputs \
#     --models llava-v1.5-7b

# bash eval_fast_v_baseline.sh

# export TOKENIZERS_PARALLELISM=false
# GEN_SCRIPT_PATH=generate.py
# EVAL_SCRIPT_PATH=evaluate.py
# DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
# MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
# CUDA_LAUNCH_BLOCKING=0
# gpu_num=1
# for model in llava-v1.5-13b; do
#     for dataset_name in Rouge_OCR_VQA ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA TextNeedleInAHaystack WebQA WikiVQA nuscenes; do
#         BATCH_SIZE=1
#         mkdir -p logs/${model}
#         # Start generating
#         python ${GEN_SCRIPT_PATH} \
#             --data_dir ${DATA_DIR} \
#             --dataset_name ${dataset_name}  \
#             --model_name ${model} \
#             --output_dir outputs \
#             --batch-image ${BATCH_SIZE} \
#             --model_configs ${MODEL_CONFIG_PATH} \
#             --overwrite \
#             --combine_image 1 \
#             > logs/${model}/${dataset_name}.log
#         # Start evaluating
#         python ${EVAL_SCRIPT_PATH} \
#             --data-dir ${DATA_DIR} \
#             --dataset ${dataset_name} \
#             --result-dir outputs/${model} \
#             >> logs/${model}/${dataset_name}.log
#     done
# done
# python score.py \
#     --result-dir outputs \
#     --models llava-v1.5-13b

# export TOKENIZERS_PARALLELISM=false
# GEN_SCRIPT_PATH=generate.py
# EVAL_SCRIPT_PATH=evaluate.py
# DATA_DIR=/home/work/workspace_bum/Tokenpruning/FastV/data/MileBench
# MODEL_CONFIG_PATH=configs/internvl_model_configs.yaml
# CUDA_LAUNCH_BLOCKING=0
# gpu_num=1
# for model in InternVL-LLava-7B; do
#     for dataset_name in Rouge_OCR_VQA ANLS_DocVQA ALFRED ActionLocalization ActionPrediction ActionSequence CLEVR-Change CharacterOrder CounterfactualInference DocVQA EgocentricNavigation GPR1200 IEdit ImageNeedleInAHaystack MMCoQA MovingAttribute MovingDirection MultiModalQA OCR-VQA ObjectExistence ObjectInteraction ObjectShuffle SceneTransition SlideVQA Spot-the-Diff StateChange TQA TextNeedleInAHaystack WebQA WikiVQA nuscenes; do
#         BATCH_SIZE=1
#         mkdir -p logs/${model}
#         # Start generating
#         python ${GEN_SCRIPT_PATH} \
#             --data_dir ${DATA_DIR} \
#             --dataset_name ${dataset_name}  \
#             --model_name ${model} \
#             --output_dir outputs \
#             --batch-image ${BATCH_SIZE} \
#             --model_configs ${MODEL_CONFIG_PATH} \
#             --overwrite \
#             > logs/${model}/${dataset_name}.log

#         # Start evaluating
#         python ${EVAL_SCRIPT_PATH} \
#             --data-dir ${DATA_DIR} \
#             --dataset ${dataset_name} \
#             --result-dir outputs/${model} \
#             >> logs/${model}/${dataset_name}.log
#     done
# done
# python score.py \
#     --result-dir outputs \
#     --models InternVL-LLava-7B