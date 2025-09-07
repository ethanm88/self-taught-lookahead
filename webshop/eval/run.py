import os
import json
import argparse
import logging
import traceback
from models import gpt_usage
from search import greedy_best_first_search_multi
from webshop import WebShopTask

import asyncio

from concurrent.futures import ThreadPoolExecutor

# Configuring the logging


def process_task(args, task, task_idx):
    if args.search_algorithm == 'greedy':
        if args.value_model_no != args.max_depth:
            num_expanded_nodes = greedy_best_first_search_multi(args, task, f'fixed_{task_idx}')
            result = {
                "task_idx": task_idx,
                "reward": 0,
                "true_reward": 0,
                "success": False,
                "true_rewards_list": [],
                "num_expanded_nodes": num_expanded_nodes,
                "usage_so_far": gpt_usage()
            }
        else:
            state, value, reward, _, true_reward, true_rewards_list, num_expanded_nodes = greedy_best_first_search_multi(args, task, f'fixed_{task_idx}')
            result = {
                "task_idx": task_idx,
                "reward": reward,
                "true_reward": true_reward,
                "success": true_reward == 1,
                "true_rewards_list": true_rewards_list,
                "num_expanded_nodes": num_expanded_nodes,
                "usage_so_far": gpt_usage()
            }
    else:
        raise ValueError(f"Invalid search algorithm: {args.search_algorithm}")

    return result

EVAL_LOG_DIR = '../saved_results'
def run(args):
    if not args.select:
        raise ValueError("This script is only for select mode")
    if args.select:
        eval_file = f"{EVAL_LOG_DIR}/{args.search_algorithm}_evaluation_results_stl_{args.value_model.replace('/', '_').lower()}_{args.backend.replace('/', '_').lower()}_{args.task_start_index}-{args.task_end_index}.jsonl"

    task = WebShopTask()

    logging.basicConfig(
        filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a'
    )

    task_accs = []
    task_acks_normal = []

    # Use ThreadPoolExecutor for multithreading with batch processing
    batch_size = 1
    for batch_start in range(args.task_start_index, args.task_end_index, batch_size):
        # empty the log
        with open(args.log, "w") as f:
            f.write("")
        
        batch_end = min(batch_start + batch_size, args.task_end_index)
        batch_indices = list(range(batch_start, batch_end))
        print(f"Processing batch {batch_indices}")
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(process_task, args, task, i): i for i in batch_indices}

            batch_results = []
            for future in futures:
                task_idx = futures[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                    task_accs.append(result["true_reward"])
                    task_acks_normal.append(result["reward"])
                    # print(f"Task {task_idx}: true_reward = {result['true_reward']}")
                except Exception as e:
                    print(f"Task {task_idx} failed with exception: {e}")
                    # print traceback
                    traceback.print_exc()

        # Save batch results to file to maintain order
        if args.evaluate and args.value_model_no == args.max_depth:
            with open(eval_file, "a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")
        else:
            # log the results instead of writing to file
            for result in batch_results:
                logging.info(json.dumps(result))

    # Compute aggregate metrics
    average_true_reward = sum(task_accs) / len(task_accs) if task_accs else 0
    average_reward = sum(task_acks_normal) / len(task_acks_normal) if task_acks_normal else 0
    success_rate = len([_ for _ in task_accs if _ == 1]) / len(task_accs) if task_accs else 0

    print('average_reward', average_reward, 'average_true_reward', average_true_reward, 'success_rate', success_rate)
    logging.info(f"RESULTS: {average_reward}, SUCCESS RATE: {success_rate}")

    if args.evaluate and args.value_model_no == args.max_depth:
        with open(eval_file, "a") as f:
            f.write(json.dumps({
                "starting_task_idx": args.task_start_index,
                "ending_task_idx": args.task_end_index,
                "average_true_reward": average_true_reward,
                "average_reward": average_reward,
                "success_rate": success_rate,
                "usage": gpt_usage()
            }) + "\n")
    else:
        # log the results instead of writing to file
        logging.info(f"RESULTS: {average_reward}, SUCCESS RATE: {success_rate}")
        logging.info(f"USAGE: {gpt_usage()}")

    
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=30)
    args.add_argument('--evaluate', action='store_true')
    args.add_argument('--ignore_reflections', action='store_true')
    args.add_argument('--self_improvement_iteration', type=int, default=0)
    args.add_argument('--log', type=str)
    args.add_argument('--ground_truth_rewards', action='store_true')
    args.add_argument('--value_model', type=str, default='gpt-35-turbo')
    args.add_argument('--value_model_dict_str', type=str, default='')
    args.add_argument('--search_algorithm', type=str, default='lats')
    args.add_argument('--select', action='store_true')
    args.add_argument('--value_model_no', type=int, default=1)
    args.add_argument('--max_depth', type=int, default=5)
    args.add_argument('--run_name', type=str, default='none')
    args.add_argument('--port', type=int, default=8001)
    args.add_argument('--prompt_lookahead', action='store_true')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)