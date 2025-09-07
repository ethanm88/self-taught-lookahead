import os
import json

RESULTS_DIR = 'saved_results'
POLICY_MODELS = ['gpt-35-turbo']
VALUE_MODELS = ['unsloth_meta-llama-3.1-8b-instruct', 'gpt-35-turbo', 'gpt-4o', 'stl_unsloth_meta-llama-3.1-8b-instruct']

if __name__=="__main__":
    eval_files = [
        {'results_file': f"{RESULTS_DIR}/greedy_evaluation_results_{value}_{policy}_0-500.jsonl",
        'value_model': value,
        'policy_model': policy}
        for value in VALUE_MODELS for policy in POLICY_MODELS
    ]
    print(eval_files)
    for file_dict in eval_files:
        file, value, policy = file_dict['results_file'], file_dict['value_model'], file_dict['policy_model']
        if not os.path.exists(file):
            continue
        print(f'policy={policy}',f'value={value}')
        results = []
        cost_dict = {}
        total_expanded_states = 0
        with open(file) as json_file:
            # iterate over the file
            data = [json.loads(line) for line in json_file if 'task_idx' in json.loads(line)]
            new_d = data[-1]
            for d in data:
                while int(d['task_idx']) != len(results):
                    results.append(0)
                results.append((int)(d['success']))
                total_expanded_states += d["num_expanded_nodes"]
            for vm in new_d["usage_so_far"]:
                if vm not in cost_dict:
                    cost_dict[vm] = {'completion_tokens': 0, 'prompt_tokens': 0, 'cost': 0}
                cost_dict[vm]['completion_tokens'] += new_d["usage_so_far"][vm]["completion_tokens"]
                cost_dict[vm]['prompt_tokens'] += new_d["usage_so_far"][vm]["prompt_tokens"]
                cost_dict[vm]['cost'] += new_d["usage_so_far"][vm]["cost"]
        print('Total expanded states:', total_expanded_states)
        for end in [50, 500]:
            truncated_results = results[:end]
            if len(truncated_results) == end:
                print(f'Average true reward (first {end}):', sum(truncated_results)/len(truncated_results))
                print(f'Success rate (first {end}):', sum([1 for r in truncated_results if r == 1.0])/len(truncated_results))
        print('='*75)