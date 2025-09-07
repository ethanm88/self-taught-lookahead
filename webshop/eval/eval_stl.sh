#!/bin/bash

# Accept backend argument from the command line, default to "gpt-35-turbo"
backend="${1:-gpt-35-turbo}"

get_value_model_name() {
    local depth="$1"  # Explicitly name the parameter
    echo "../stl/stl_value_models/model_unsloth/Meta-Llama-3.1-8B-Instruct_${backend}_0_1_start=1500_num_samples=50_depth=${depth}_loss_"
}

EVAL_TASK_START_IDX=0
EVAL_TASK_END_IDX=1

MAX_DEPTH=5
DEVICE=1
port=8020
run_name=eval_stl_${backend}

# Main iteration loop
for (( i=1; i<=MAX_DEPTH; i++ )); do 
    echo "Starting iteration $i..."

    # Launch vllm server
    if [ $i -eq 5 ]; then
        current_value_model="unsloth/Meta-Llama-3.1-8B-Instruct"
    else
        current_value_model=$(get_value_model_name $i)
    fi
    echo "Launching initial vllm serve process..."
    tmux kill-session -t stl_value_model
    echo current_value_model $current_value_model
    tmux new-session -d -s "stl_value_model" \
    "source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate stl && \
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    vllm serve '${current_value_model}' --dtype auto \
    --api-key jackets --trust-remote-code \
    --max_model_len=8192 \
    --gpu-memory-utilization 0.97 \
    --port=${port} ; bash"

    sleep 20

    # Run eval
    bash launch_stl_eval.sh --value_model "$current_value_model" --value_model_no $i --run_name "${run_name}" --backend "${backend}" --port $port --eval_start_task_index ${EVAL_TASK_START_IDX} --eval_end_task_index ${EVAL_TASK_END_IDX}
done
