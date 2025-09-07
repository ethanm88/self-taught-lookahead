import json
import numpy as np
import os
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metric', choices=['average_reward', 'success_rate'], default='average_reward')
parser.add_argument('--num_instances', type=int, default=500)
args = parser.parse_args()

def load_rewards(file, num):
    with open(file, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
        return [entry["true_reward"] for entry in data if "true_reward" in entry][:num]


def compute_metric(results, metric, num):
    if metric == "average_reward":
        return results[:num].sum() / num
    elif metric == "success_rate":
        return (results[:num] == 1).sum() / num
    else:
        raise ValueError("Invalid metric. Choose either 'average_reward' or 'success_rate'.")

def compute_p_value(args):
    model, results, baseline_results, baseline_metric, delta, b, metric = args

    # Generate b bootstrap samples with NumPy
    bootstrap_samples = np.random.choice(baseline_results, (b, len(baseline_results)), replace=True)

    if metric == "average_reward":
        bootstrap_metrics = bootstrap_samples.mean(axis=1)
    elif metric == "success_rate":
        bootstrap_metrics = (bootstrap_samples == 1).mean(axis=1)
    else:
        raise ValueError("Invalid metric. Choose either 'average_reward' or 'success_rate'.")

    # Compute deltas
    bootstrap_deltas = bootstrap_metrics - baseline_metric

    # Calculate p-value
    p_value = np.mean(bootstrap_deltas > delta * 2)
    return model, p_value

RESULTS_DIR = 'saved_results'
POLICY_MODELS = ['gpt-35-turbo', 'gpt-4o']
VALUE_MODELS = ['unsloth_meta-llama-3.1-8b-instruct', 'deepseek-ai_deepseek-r1-distill-llama-8b', 'gpt-35-turbo', 'gpt-4o', 'stl_unsloth_meta-llama-3.1-8b-instruct']

if __name__=="__main__":
    # we compare to using a llama value model with a gpt-3.5-turbo policy
    baseline_results_file = f"{RESULTS_DIR}/greedy_evaluation_results_unsloth_meta-llama-3.1-8b-instruct_gpt-35-turbo_0-500.jsonl"
    eval_files = [
        {'results_file': f"{RESULTS_DIR}/greedy_evaluation_results_{value}_{policy}_0-500.jsonl",
        'value_model': value,
        'policy_model': policy}
        for value in VALUE_MODELS for policy in POLICY_MODELS
    ]
    eval_files.append( # add MCTS
        {'results_file': f"{RESULTS_DIR}/mcts_results_unsloth_meta-llama-3.1-8b-instruct_gpt-35-turbo_0-50.jsonl",
        'value_model': "unsloth_meta-llama-3.1-8b-instruct",
        'policy_model': "mcts_gpt-35-turbo"}
    )

    # Load baseline and comparison results
    baseline_results = np.array(load_rewards(baseline_results_file, args.num_instances))
    comparison_results = {f"{config['value_model']}|{config['policy_model']}": np.array(load_rewards(config['results_file'], args.num_instances)) for config in eval_files}


    all_results = {}
    # Compute metric for each model
    for model, results in comparison_results.items():
        value = compute_metric(results, args.metric, args.num_instances)
        all_results[model] = value


    b = 10**6 # use a million bootstrap samples
    baseline_metric = compute_metric(baseline_results, args.metric, args.num_instances)

    # Prepare arguments for parallel computation
    args_list = []
    for model, results in comparison_results.items():
        comparison_metric = compute_metric(results, args.metric, args.num_instances)
        delta = comparison_metric - baseline_metric
        args_list.append((model, results, baseline_results, baseline_metric, delta, b, args.metric))

    with Pool() as pool:
        p_values = dict(pool.map(compute_p_value, args_list))

    print(f'Bootstrap p values for metric={args.metric}')
    for model, p_value in p_values.items():
        stars = 0
        if p_value < 0.05:
            stars = 1
        if p_value < 0.01:
            stars = 2
        if p_value < 0.001:
            stars = 3
        print(f"policy: {model.split('|')[1]} value_model: {model.split('|')[0]} {args.metric}: {all_results[model]} p-value: {p_value} {'*' * stars}")
