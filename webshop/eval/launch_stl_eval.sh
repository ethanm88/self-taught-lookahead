#!/bin/bash

# Default values for arguments
SELF_IMPROVEMENT_ITERATION=1
MODEL_NAME=gpt-35-turbo
VALUE_MODEL_NAME=model
VALUE_MODEL_NO=1
EVAL_TASK_START_IDX=0
EVAL_TASK_END_IDX=50
ALGORITHM=greedy
MAX_DEPTH=5
run_name="none"
PORT=8001

# Parse command-line arguments for overrides
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --self_improvement_iteration)
      SELF_IMPROVEMENT_ITERATION="$2"
      shift
      shift
      ;;
    --backend)
      MODEL_NAME="$2"
      shift
      shift
      ;;
    --eval_start_task_index)
      EVAL_TASK_START_IDX="$2"
      shift
      shift
      ;;
    --eval_end_task_index)
        EVAL_TASK_END_IDX="$2"
        shift
        shift
        ;;
    --value_model)
        VALUE_MODEL_NAME="$2"
        shift
        shift
        ;;
    --value_model_no)
        VALUE_MODEL_NO="$2"
        shift
        shift
        ;;
    --run_name)
        run_name="$2"
        shift
        shift
        ;;
    --port)
        PORT="$2"
        shift
        shift
        ;;
    *)
      # Unknown option, pass through
      break
      ;;
  esac
done
echo MODEL NAME: $MODEL_NAME
model_name_cleaned=${VALUE_MODEL_NAME//\//_}
log_file=logs/log_${MODEL_NAME}_${model_name_cleaned}
echo START IDX ${EVAL_TASK_START_IDX}
echo END IDX ${EVAL_TASK_END_IDX}
python run.py \
    --backend ${MODEL_NAME} \
    --value_model ${VALUE_MODEL_NAME} \
    --task_start_index ${EVAL_TASK_START_IDX} \
    --task_end_index ${EVAL_TASK_END_IDX} \
    --n_generate_sample 5 \
    --n_evaluate_sample 5 \
    --prompt_sample cot \
    --temperature 1.0 \
    --iterations 3 \
    --self_improvement_iteration ${SELF_IMPROVEMENT_ITERATION} \
    --evaluate \
    --log ${log_file} \
    --ignore_reflections \
    --search_algorithm ${ALGORITHM} \
    --select \
    --max_depth ${MAX_DEPTH} \
    --value_model_no ${VALUE_MODEL_NO} \
    --run_name ${run_name} \
    --port ${PORT} \
    ${@}

# remember to change the url in lats.py to your local instance of WebShop 

