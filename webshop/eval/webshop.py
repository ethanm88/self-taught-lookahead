# Credit: this code is adapted from https://github.com/lapisrocks/LanguageAgentTreeSearch

import os
import re
from base import Task
from prompt import *
from models import gpt, gpt4
import logging
import random
from typing import List
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def get_token_length(text):
    return len(tokenizer.encode(text))

max_token_length = 15000

class WebShopTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}
        self.reflections = []
    
    def test_output(self, idx: int, output: str):
        output = output.split('Action:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model='gpt-4')
        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*correctness score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        # print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    
    @staticmethod
    def extract_actions(text):
        """
        Extract possible actions from a webshop interaction string.
        Finds the most recent observation that contains bracketed terms.
        
        Args:
            text (str): Full interaction string including all observations
        
        Returns:
            list: List of possible actions in the format "click[term]"
        """
        
        # Split into sections by "Action:" markers
        sections = re.split(r'Action:', text)
        
        # Get the observations (they come after the action descriptions)
        observations = []
        for section in sections:  # Skip the first section (instructions)
            # Split each action section to separate action from observation
            parts = section.split('\n', 1)  # Split on first newline
            if len(parts) > 1:
                observations.append(parts[1])
        
        # Work backwards through observations to find the first one with bracketed terms
        for obs in reversed(observations):
            # Skip observations that are just confirmations of clicks
            if 'You have clicked' in obs:
                continue
                
            # Extract all bracketed terms
            brackets = re.findall(r'\[(.*?)\]', obs)
            if brackets:
                actions = []
                
                # Convert bracketed terms to click actions
                for term in brackets:
                    # Skip if it's just a number
                    if term.isdigit():
                        continue
                        
                    # Add as clickable action
                    actions.append(f"click[{term}]")
                
                return sorted(list(set(actions)))  # Remove duplicates and sort
                
        return []  # Return empty list if no actions found

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[], ignore_reflection=False) -> str:
        question = x
        input = x + y
        trajectories = ""
        
        if reflection_mapping_list and not ignore_reflection:
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "Reflection: " + reflection_mapping['reflection'] + "\n"
                trajectories += traj_with_reflection
            
            prompt = prompt_cot_feedback.format(trajectories=trajectories, input=input)
            return prompt
        else:
            # do not use thinking:
            prompt = prompt_cot_no_think.format(input=input)
            return prompt
        
    @staticmethod
    def cot_select_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[], ignore_reflection=False, possible_actions=None, not_allowed_actions=None) -> str:
        input = x + y
        if possible_actions is None:
            possible_actions = WebShopTask.extract_actions(input)
        if not_allowed_actions is None or len(not_allowed_actions) == 0:
            not_allowed_actions = ['None - all actions are allowed']
        prompt = prompt_cot_no_think_selection.format(task=input, possible_actions='\n'.join(possible_actions), not_allowed_actions='\n'.join(not_allowed_actions))
        return prompt, possible_actions


        
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = score_prompt + "\n" + x + "\n\n"
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best trajectory is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        
        # Extract the last Action for each trajectory
        last_actions = []
        for y in ys:
            # Split by line and reverse to start from the end
            lines = y.split('\n')[::-1]
            for line in lines:
                # Check for an Action line and get its content
                if "Action" in line:
                    last_actions.append(line.split('Action')[-1].strip(': '))
                    break

        assert len(last_actions) == 2, 'Expected to find 2 Actions'

        # Construct the prompt with the extracted Actions
        prompt = compare_prompt + f'Action 1:{last_actions[0]}\n\nAction 2:{last_actions[1]}\n'
        return prompt

    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more correct trajectory is 1' in compare_output:
            return 0
        elif 'more correct trajectory is 2' in compare_output:
            return 1
        elif "two trajectories are similarly correct" in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = [], ignore_reflection=False, terminal=False, finetuned=False, attribute_click=False, args=None) -> str:
        if len(z) != 0 and not ignore_reflection:
            failed_trajectories = ""
            for traj, ref in zip(z, reflections):
                score = int(traj['r'] * 10) / 2
                trajectory = traj['trajectory']
                split_trajectory = trajectory.split('Action: ')
                first_part = split_trajectory[0]  # This part will not be modified

                # Remove the first 'Action' and corresponding 'Observation'
                remaining_parts = split_trajectory[2:]

                # Reconstruct the trajectory string
                new_trajectory = 'Action: '.join([first_part] + remaining_parts)
                traj['trajectory'] = new_trajectory
                failed_trajectories += f"{y}\n{traj}\nReflection: {ref['reflection']}\nThus the correctness score is {score}\n"
            
            inp = y + "\n\nReflection: "
            prompt = score_prompt_feedback.format(s="", trajectories=failed_trajectories, input=inp)
        else:
            # inp = y + "\n\nReflection: "
            inp = y
            current_observation = inp.split('Observation: ')[-1]
            action_index = inp.rindex('Action:')
            last_action = inp[action_index:].strip()
            if finetuned:
                current_action = inp.split('Action: ')[-1].split('Observation:')[0]
                # truncate the trajectory
                action_index = inp.rindex('Action:')
                truncated_trajectory = inp[:action_index].strip()
                if terminal:
                    # need to replace the observation in order to not give the model the
                    prompt = score_prompt_finetuned_terminal.format(truncated_trajectory=truncated_trajectory, current_action=current_action, current_observation="Terminal State")
                else:
                    prompt = score_prompt_finetuned.format(truncated_trajectory=truncated_trajectory, current_action=current_action, current_observation=current_observation)
                # system_prompt = score_prompt_system_finetuned
            elif terminal:
                # get index of last Observation
                last_observation_idx = inp.rfind('Observation:')
                truncated_inp = inp[:last_observation_idx].strip()
                prompt = score_prompt_terminal.format(input=truncated_inp)
            else:
                # print('last_action:', last_action)
                print('prompt_lookahead:', args.prompt_lookahead)
                if args.prompt_lookahead:
                    prompt = score_prompt_lookahead.format(input=inp, last_action=last_action)
                else:
                    prompt = score_prompt.format(input=inp, last_action=last_action)
                
        return prompt


    @staticmethod
    def value_outputs_unwrap(evaluate_prompts: List[str], terminal=False, attribute_click=False, product_score=0, finetuned=False, metric='mean', args=None) -> float:
        logging.info(f'evaluate_prompts sample: {random.sample(evaluate_prompts, 1)}')
        def compute_mean(scores):
            return sum(scores) / len(scores) if scores else 0
        def get_success_proportion(text):
            # Use regex to find all steps that start with a number followed by a period and space (e.g., '1. ')
            steps = re.findall(r'\d+\.\s.*?(?=\n\d+\.\s|\Z)', text, re.DOTALL)
            success_count = 0
            total_steps = 0

            for step in steps:
                if step.strip():  # Ensure the step is not empty
                    total_steps += 1
                    # Check if there is at least one [success] in the step
                    if "[success]" in step:
                        success_count += 1

            # Calculate the proportion of success
            if total_steps == 0:
                return 0  # Avoid division by zero
            return success_count / total_steps
        def evaluate_score(evaluate_prompt):
            if finetuned:
                score_prefix = 'the correctness score is '
                if score_prefix not in evaluate_prompt:
                    return -1
                else:
                    evaluate_prompt = evaluate_prompt.split(score_prefix)[-1]
                if '/' not in evaluate_prompt:
                    return -1
                score_str = evaluate_prompt.split('/')[0].strip()
                try:
                    ret_score = float(score_str) / 10.0
                    return ret_score
                except:
                    return -1
            else:
                if terminal and not args.evaluate:
                    score = get_success_proportion(evaluate_prompt)
                    return score
                score_prefix = 'the correctness score is '
                if score_prefix not in evaluate_prompt:
                    return -1
                else:
                    evaluate_prompt = evaluate_prompt.split(score_prefix)[-1]
                score_str = evaluate_prompt
                try:
                    ret_score = float(score_str) / 10.0
                    return ret_score
                except:
                    return -1
            
        scores_rationale = []
        # TODO: Potential error here
        for i in range(len(evaluate_prompts)):
            cur_evaluate_prompt = evaluate_prompts[i]
            score = evaluate_score(cur_evaluate_prompt.lower())
            if score < 0:

                cur_evaluate_prompt = ''
            scores_rationale.append((score, cur_evaluate_prompt))
        # get median score
        scores = [[score, rationale] for score, rationale in scores_rationale if score >= 0]
        # sort by first element
        sorted_scores = sorted(scores, key=lambda x: x[0])
        if len(sorted_scores) == 0:
            return -1, ''
        median_score_tuple = sorted_scores[len(sorted_scores) // 2]
        
        metric_func_selector = {
            'mean': compute_mean
        }
        metric_func = metric_func_selector.get(metric, compute_mean)
        
        # add to product
        if attribute_click and not finetuned and not args.evaluate:
            convert_score_dict = {0.1: -0.2, 0.2: -0.1, 0.3: 0.1, 0.4: 0.2}
            attribute_add_ons = [convert_score_dict[score] for score, _ in scores]
            new_scores = [min(max(0, product_score + attribute_add_on), 1) for attribute_add_on in attribute_add_ons]
            ret_score = metric_func(new_scores)

        else:
            ret_score = metric_func([score for score, _ in scores])
        return ret_score, median_score_tuple[1]
                
