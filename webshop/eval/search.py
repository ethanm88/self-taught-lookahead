
# Credit: this code is adapted from https://github.com/lapisrocks/LanguageAgentTreeSearch


import os
import re
import json
import pickle
from tqdm import tqdm
import openai
import backoff
import sys
import copy
import itertools
import numpy as np
from functools import partial
from models import gpt
import requests
import logging
import random
import time
import asyncio
import traceback
completion_tokens = prompt_tokens = 0
openai.api_key = os.environ["OPENAI_API_KEY"]

import hashlib

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

WEBSHOP_URL = "http://127.0.0.1:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )
def clean_url(url):
    return url.replace('#', '')

def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    url = clean_url(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                # enable debugger
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                # ONLY ALLOW 5 PRODUCTS
                if prod_cnt >= 5:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                # NOTE: Added the next line back in to only include a few observations
                if just_prod <= 2 and prod_cnt >= 6: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
  def __init__(self):
    self.sessions = {}

  def clone_state(self):
    return copy.deepcopy(self.sessions)
  def clean_query(self, query):
        query = query.replace('#', '')
        return query
  def step(self, session, action):
    done = False
    observation_ = None
    # logging.info(self.sessions)
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      query = self.clean_query(query)
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        #done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        #assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          #assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
        else:
            assert False
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    
    # NOTE: If reward is in info it means it was found on the page, so the current state gives reward, so the previous state will be counted as terminal
    done_heuristic = 'reward' in info
    # if reward != 0.0:
    #     #print(f"Current Session State: {self.sessions[session]}")
    #     #print(f"Action being processed: {action}")
    #     # print(f"Resulting Observation: {observation}")
    if reward == 1.0:
        done = True
        # print("done")
    return observation, reward, done, done_heuristic

env = webshopEnv()

global reflection_map
global failed_trajectories
global action_outcome_rationales
global finetuned_model
global expanded_states
expanded_states = 0
reflection_map = []
failed_trajectories = []
action_outcome_rationales = {}
finetuned_model = False


import numpy as np

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

def select_node_softmax(node, temperature=1.0):
    while node and node.children:
        uct_values = [child.uct() for child in node.children if not child.is_terminal]
        if not uct_values:
            return None  # All children are terminal

        probabilities = softmax(np.array(uct_values), temperature)
        selected_child = np.random.choice([child for child in node.children if not child.is_terminal], p=probabilities)
        
        node = selected_child
        
    return node

def normalize_key(key):
    """
    Normalize a given key to make it uniform.
    Steps:
    1. Remove special characters and extra spaces.
    2. Convert to lowercase.
    3. Sort words alphabetically.
    """
    # Step 1: Remove special characters (except spaces) and normalize spaces
    key = re.sub(r'[^\w\s]', '', key)  # Remove special characters
    key = re.sub(r'\s+', ' ', key.strip())  # Normalize spaces
 
    # Step 2: Convert to lowercase
    key = key.lower()
 
    # Step 3: Tokenize, sort, and rejoin
    tokens = key.split()  # Split into words
    tokens.sort()  # Sort alphabetically
    normalized_key = ' '.join(tokens)  # Rejoin into a single string
 
    return normalized_key

def get_last_non_attribute_action_score(y, value_cache):
    # get indicies of all Action:
    action_indices = [m.start() for m in re.finditer('Action:', y)]
    for ending_idx in reversed(action_indices):
        if check_product_click(y[:ending_idx]):
            last_action_observation = y[:ending_idx].split('Action:')[-1].strip()
            normalized_key = normalize_key(last_action_observation)
            if normalized_key not in value_cache:
                raise ValueError(f"Value not found in cache: {normalized_key}")
            # print('prev_action_score', value_cache[normalized_key])
            return value_cache[normalized_key]

def check_product_click(y):
    last_action_observation = y.split('Action:')[-1].strip()
    last_action = last_action_observation.split('Observation')[0].strip()
    if 'click' in last_action:
        button_clicked = last_action.split('click')[-1].strip()[1:-1]
        # print('check product', button_clicked)
        if button_clicked.lower() != button_clicked and button_clicked != 'Buy Now':
            return True
    return False

special_keys = ['Back to Search', 'Next >', '< Prev', 'Attributes', 'Description', 'Features', 'Reviews']
def check_attribute_click(y):
    last_action_observation = y.split('Action:')[-1].strip()
    last_action = last_action_observation.split('Observation')[0].strip()
    if 'click' in last_action:
        button_clicked = last_action.split('click')[-1].strip()[1:-1]
        # print(button_clicked, button_clicked.lower() == button_clicked and button_clicked != 'Buy Now')
        if button_clicked.lower() == button_clicked and button_clicked not in special_keys and button_clicked != 'Buy Now':
            return True
    return False

def check_btn_clicked_special(y):
    last_action_observation = y.split('Action:')[-1].strip()
    last_action = last_action_observation.split('Observation')[0].strip()
    if 'click' in last_action:
        button_clicked = last_action.split('click')[-1].strip()[1:-1]
        if button_clicked in special_keys:
            return True
    return False

async def get_value(task, x, y, n_evaluate_sample, cache_value=True, ignore_reflections=False, terminal=False, args=None, model_no=0):
    global reflection_map
    global failed_trajectories
    global finetuned_model
    global value_gpt_dict

    last_action_observation = y.split('Action:')[-1].strip()
    normalized_key = normalize_key(last_action_observation)
    
    last_action = last_action_observation.split('Observation')[0].strip()

    # First: check that current action has not previously occurred in trajectory
    action_start_idx = y.rindex('Action:')
    truncated_trajectory = y[:action_start_idx].strip()

    # Do not allow special buttons to be clicked at all
    if check_btn_clicked_special(y):
        return -1, 'Invalid action - special button clicked'
    
    # Next, check if the last action has already occurred in the trajectory
    if last_action in truncated_trajectory:
        return -1, 'Invalid action - action has already occurred in trajectory'


    # Next, check cache (for terminal all actions and successor states will be the same, so we do not cache)
    if cache_value and normalized_key in task.value_cache and not terminal:
        return task.value_cache[normalized_key]

    
    # fetch relevant value model
    if len(value_gpt_dict) == 1:
        value_gpt = value_gpt_dict[0]
    else:
        value_gpt = value_gpt_dict[model_no]
    
    if finetuned_model:
        value_prompt = task.value_prompt_wrap(x, y, failed_trajectories, reflection_map, ignore_reflections, terminal, finetuned_model, attribute_click=None, args=args)

        if 'Invalid action' in value_prompt:
            return -1, 'Invalid action'
    
        value_outputs = await asyncio.to_thread(
            value_gpt, value_prompt, n=n_evaluate_sample
        )
        logging.info(f"VALUE OUTPUTS: {value_outputs}")

        value, rationale = task.value_outputs_unwrap(value_outputs, finetuned=True, args=args)

    else:
        # NOTE: we only do this if we are not using the finetuned model
        # this checks whether attribute are selected since all buttons are all captial
        attribute_click = check_attribute_click(y)
        
        # Next, we need to get the value of the product click
        product_score = 0
        if attribute_click:
            if get_last_non_attribute_action_score(y, task.value_cache) is None:
                return -1, 'Invalid action - no product click before attribute click'
            product_score, _ = get_last_non_attribute_action_score(y, task.value_cache)
        
        value_prompt = task.value_prompt_wrap(x, y, failed_trajectories, reflection_map, ignore_reflections, terminal, finetuned_model, attribute_click=attribute_click, args=args)

        if 'Invalid action' in value_prompt:
            return -1, 'Invalid action'
    
        value_outputs = await asyncio.to_thread(
            value_gpt, value_prompt, n=n_evaluate_sample # NOTE: REMOVED STOP=NONE
        )
        value, rationale = task.value_outputs_unwrap(value_outputs, attribute_click=attribute_click, product_score=product_score, terminal=terminal, args=args)

    if True: # always cache so can be used for attribute scores
        task.value_cache[normalized_key] = (value, rationale)

    return value, rationale


def get_values(task, x, ys, n_evaluate_sample, cache_value=True, ignore_reflections=False, terminal=False, args=None, non_terminal_multiplier=0.75, model_no=0):
    global finetuned_model
    async def compute_value(y):
        value, value_outputs = await get_value(
            task, x, y, n_evaluate_sample, cache_value=cache_value, ignore_reflections=ignore_reflections, terminal=terminal, args=args, model_no=model_no
        )
        return value, value_outputs

    async def process_all():
        # Create tasks for all values
        tasks = [compute_value(y) for y in ys]
        results = await asyncio.gather(*tasks)
        return results

    try:
        # Run the event loop and retrieve results
        verbose_values = asyncio.run(process_all())
        values_only = [value for value, _ in verbose_values]
        # print('VALUES:', values_only)
        rationales_only = [rationale for _, rationale in verbose_values]
        if not terminal and not finetuned_model and not args.evaluate:
            # If terminal, we do not need to adjust the values
            values_only = [value * non_terminal_multiplier for value in values_only]
        return values_only, rationales_only
    except Exception as e:
        # Debugging information
        print(f"Error occurred during async execution: {e}")
        # print traceback
        import traceback
        traceback.print_exc()
        
        return None

def get_terminal_values_from_rewards(nodes):
    values = []
    rationales = []
    for node in nodes:
        r = node.reward
        values.append(r * 10)
        reward_mapping = ['perfectly captures', 'imperfectly, but mostly captures', 'partially captures', 'does not capture at all']
        def map_reward(current_reward):
            if current_reward == 1:
                return reward_mapping[0]
            elif current_reward >= 0.5:
                return reward_mapping[1]
            elif current_reward > 0:
                return reward_mapping[2]
            else:
                return reward_mapping[3]
        # NOTE: we actually do not need this when we only use 
        # automatically construct rationale to use
        rationales.append(f'Reflection: The last action is to buy the selected item. The selected item {map_reward(r)} the user intent. Thus the correctness score is {values[-1]}.')
    return values, rationales

def get_terminal_values_from_llm(nodes):
    values = []
    rationales = []
    for node in nodes:
        values.append(node.reward)
        rationales.append(node.reward_rationale)
    return values, rationales
    
def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, ignore_reflections=False, args=None):
    global reflection_map
    global failed_trajectories

    if len(failed_trajectories) > len(reflection_map) and len(failed_trajectories) < 4 and not ignore_reflections:
        print("generating reflections")
        print(len(failed_trajectories))
        print(len(reflection_map))
        reflection_map = task.generate_self_reflection(failed_trajectories, x)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map, ignore_reflections)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)

    actions = [y + _.split('Action:')[-1].strip() for _ in samples if _ is not None]
    rationales = [_.split('Action:')[0].strip() for _ in samples if _ is not None]
    return actions, rationales

def get_samples_select(task, x, y, n_generate_sample, prompt_sample, stop, ignore_reflections=False, args=None):
    actions, rationales = [], []
    possible_actions = task.extract_actions(x + y)
    if len(possible_actions) == 1 and 'click[Search]' in possible_actions:
        # For search it is not possible to enumerate all possible actions
        return get_samples(task, x, y, n_generate_sample, prompt_sample, stop, ignore_reflections, args)
    if len(possible_actions) == 0:
        raise ValueError('No possible actions found')
    if len(possible_actions) <= n_generate_sample:
        # If there are fewer possible actions than the number of samples to generate, just generate all possible actions
        actions = [y + action for action in possible_actions]
        rationales = ['' for _ in range(len(actions))]
        return actions, rationales
    # NOTE: Need to still generate rationales even if fewer than n_generate_sample actions are possible
    for i in range(min(n_generate_sample, len(possible_actions))):
        tries = 3
        while True:
            if tries <= 0:
                logging.error('Failed to generate sample, skipping...')
                print('Failed to generate sample, skipping...')
                # set random seed to ensure same results
                random.seed(42)
                actions += random.choices(possible_actions, k=n_generate_sample - len(actions))
                rationales += ['' for _ in range(n_generate_sample - len(rationales))]
                break
            try:
                prompt, _ = task.cot_select_prompt_wrap(x, y, possible_actions=possible_actions, not_allowed_actions=actions)
                samples = gpt(prompt, n=1, stop=stop)
                cur_action = [_.split('Action:')[-1].strip() for _ in samples if _ is not None]
                cur_rationale = [_.split('Action:')[0].strip() for _ in samples if _ is not None]

                assert cur_action[0] in possible_actions
                # remove current action from possible actions
                possible_actions.remove(cur_action[0])
                actions.extend(cur_action)
                rationales.extend(cur_rationale)
            except AssertionError:
                logging.error('Failed to generate sample, retrying...')
                print('Failed to generate sample, retrying...')
                tries -= 1
                print('Tries', tries)
                continue
            except Exception as e:
                logging.error(f"Failed to generate sample: {e}")
                print(f"Failed to generate sample: {e}")
                tries -= 1
                continue
            break
    actions = [y + action for action in actions]
    print('Actions', actions)
    return actions, rationales

def get_unique_trajectories(failed_trajectories, num=3):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories

class Node:
    def __init__(self, state, question, env_state=None, parent=None, depth=0):
        self.state = {'action': '', 'observation': '', 'rationale': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = depth if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.true_reward = 0
        self.reward_rationale = ''
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.env_state = env_state

    def uct(self):
        if self.visits == 0 and self.value >= 0:
            return float('inf')
            #return self.value * 2
        elif self.visits == 0 and self.value < 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }
    
def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def collect_actions_to_node(node):
    actions = []
    while node:
        if node.state['action']:
            actions.append(node.state['action'])
        node = node.parent
    return list(reversed(actions))


def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes

def collect_trajectory(node):
    trajectory = []
    #print("collecting traj", node)
    
    # Append the question from the root node
    trajectory.append(node.question)
    
    # Collect action and observation from each node till the root
    while node:
        if node.state and 'action' in node.state and node.state['action'] and node.parent:
            trajectory.append(f"Action: {node.state['action']}")
        else:
            logging.warning(f"Missing or empty action in node at depth {node.depth}")
            
        if node.state and 'observation' in node.state and node.state['observation'] and node.parent:
            trajectory.append(f"Observation: {node.state['observation']}\n")
        else:
            logging.warning(f"Missing or empty observation in node at depth {node.depth}")
            
        node = node.parent
    return '\n'.join(trajectory)


def serialize_node(node, file_name):
    try:
        with open(file_name, 'wb') as file:
            pickle.dump(node, file)
        # print(f"Node object successfully serialized to {file_name}")
    except Exception as e:
        print(f"An error occurred while serializing the Node object: {e}")    
        raise e

def load_node(file_name):
    try:
        with open(file_name, 'rb') as file:
            try:
                node = pickle.load(file)
            except EOFError:
                # remove the file if it is empty
                os.remove(file_name)
                return None
            except Exception as e:
                return None
        # print(f"Node object successfully loaded from {file_name}")
        return node
    except Exception as e:
        print(f"An error occurred while loading the Node object: {e}")
        raise e

def save_file_name(idx, current_depth, trial_no, run_name, eval_save_dir='intermediate_eval_results'):
    # make dir for current depth
    current_depth_save_dir = f"{eval_save_dir}/{run_name}/depth_{current_depth}"
    os.makedirs(current_depth_save_dir, exist_ok=True)
    # return file name
    return os.path.join(current_depth_save_dir, f"task={idx}_trial={trial_no}.pkl")
    
def greedy_best_first_search_multi(args, task, idx, to_print=True):
    global gpt
    global value_gpt_dict
    global failed_trajectories
    global reflection_map
    global finetuned_model
    global expanded_states
    
    # reset expanded states
    expanded_states = 0
    print('RUN NAME:', args.run_name)
    action = 'reset'
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    
    # check if we have multiple value models
    if args.value_model_dict_str:
        value_model_dict = json.loads(args.value_model_dict_str)
        value_gpt_dict = {model_no: partial(gpt, model=value_model_name, temperature=args.temperature, port=port, stop="<|") for model_no, (value_model_name, port) in value_model_dict.items()}
    else: # otherwise use default port and value model name
        value_gpt_dict = {0: partial(gpt, model=args.value_model, temperature=args.temperature, port=args.port, stop="<|")}
    
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    #env.sessions[idx] = {'session': idx, 'page_type': 'init'}
    x = env.step(idx, action)[0]

    finetuned_model = args.self_improvement_iteration > 0
    root = Node(state=None, question=x)
    root.env_state = copy.deepcopy(env.sessions)
    failed_trajectories = []
    reflection_map = []
    
    best_nodes = []
    
    # print('='*50)
    # print('VALUE MODEL IN USE:', args.value_model)
    # print('VALUE MODEL NO:', args.value_model_no)
    # print('='*50)
    # Main Loop
    # if idx == "fixed_35":
    #     breakpoint()
    num_terminal = 0
    for i in tqdm(range(args.iterations)):
        logging.info(f"Iteration {i + 1}")
        # print('Iteration:', i)
        
        # check if node already exists
        if os.path.exists(save_file_name(idx, args.value_model_no, i, args.run_name)):
            # print('Loading node from file...')
            cur_node = load_node(save_file_name(idx, args.value_model_no, i, args.run_name))
            if cur_node is None:
                logging.info(f"Failed to load node from file, starting from the root node")
                continue
            if cur_node.is_terminal:
                num_terminal += 1
                best_nodes.append(cur_node)
            continue
        if args.value_model_no == 1:
            cur_node = root
        else: # NOTE: Need correct indexing
            cur_node = load_node(save_file_name(idx, args.value_model_no - 1, i, args.run_name))
            if cur_node is None:
                logging.info(f"Failed to load node from file, starting from the root node")
                raise ValueError("Failed to load node from file")
            
        # check if node is terminal
        if cur_node.is_terminal:
            # NOTE: save the current node as new node - will be processed at the end
            print('Current node is terminal, skipping...')
            # save the current node
            serialize_node(cur_node, save_file_name(idx, args.value_model_no, i, args.run_name))
            # add to best nodes
            best_nodes.append(cur_node)
            continue
            
        # NOTE: depth is 0 indexed
        depth = args.value_model_no - 1
        
        # NOTE: We add buy now to ensure task is completed
        if depth != args.max_depth - 1:
            expand_node(cur_node, args, task, idx, add_buy_now=True)  # Expand current node
            tries = 3
            while len(cur_node.children) == 0 or (len(cur_node.children) == 1 and cur_node.children[0].state['action'] == 'click[Buy Now]'):
                if tries == 0:
                    break
                tries -= 1
                cur_node.children = []
                expand_node(cur_node, args, task, idx, add_buy_now=True)
            if not cur_node.children:
                print('BREAKING: No children found')
                continue
                # break  # If no child can be generated, break
        else: # need to ensure that the last action is buy now
            expand_node(cur_node, args, task, idx, add_buy_now=True, only_terminal=True)
            # breakpoint()
            # remove all non-terminal nodes
            cur_node.children = [child for child in cur_node.children if child.is_terminal]
            print('Num Terminal Children:', len(cur_node.children))
            # NOTE: MAYBE THIS IS THE PROBLEM:
            # cur_node.children = [Node(state={'action': 'click[Buy Now]', 'observation': '', 'rationale': ''}, question=cur_node.question, parent=cur_node, depth=cur_node.depth + 1)]
        # Select the best child node
        cur_node.children = random.sample(cur_node.children, len(cur_node.children))
        print('Children:', [child.state['action'] for child in cur_node.children])        
        terminal_nodes = [child for child in cur_node.children if child.is_terminal and child is not None]
        non_terminal_nodes = [child for child in cur_node.children if not child.is_terminal and child is not None]
        all_nodes = non_terminal_nodes + terminal_nodes

        non_terminal_child_prompts = [generate_prompt(child) for child in non_terminal_nodes]
        if not args.value_model == 'random':
            if len(all_nodes) == 1:
                cur_node = all_nodes[0]
                print('Selected Action:', cur_node.state['action'])
            else:
                # note we do not cache the values for eval
                values, _ = get_values(task, cur_node.question, non_terminal_child_prompts, args.n_evaluate_sample, cache_value=False, ignore_reflections=args.ignore_reflections, args=args, model_no=depth + 1)
                terminal_values, _ = get_terminal_values_from_llm(terminal_nodes)
                values = values + terminal_values
                nodes_values = [(node, value) for node, value in zip(all_nodes, values)]
                if len(nodes_values) == 0:
                    continue
                cur_node = max(nodes_values, key=lambda x: x[1])[0]
                print('Selected Action:', cur_node.state['action'])
            
            # save the current node
            serialize_node(cur_node, save_file_name(idx, args.value_model_no, i, args.run_name))

        if args.value_model_no == args.max_depth:
            best_nodes.append(cur_node)
            print('Current Best Reward:', cur_node.true_reward)
            root.children = []

    if args.value_model_no == args.max_depth:
        # Post-process: select the best trajectory (based on true reward)
        best_node = max(best_nodes, key=lambda x: x.true_reward)
        print('Best True Reward:', best_node.true_reward)
        rewards_true_rewards = [(node.reward, node.true_reward) for node in best_nodes]
        # sorted_rewards_true_rewards = sorted(rewards_true_rewards, key=lambda x: x[1], reverse=True)
        return best_node.state, best_node.value, best_node.reward, best_node.em, best_node.true_reward, rewards_true_rewards, expanded_states
    print('Num Terminal:', num_terminal)
    return expanded_states


def expand_node(node, args, task, idx, add_buy_now=False, set_n=None, only_terminal=False):
    n = args.n_generate_sample
    if node.depth >= args.max_depth:
        logging.info("Depth limit reached")
        return
    if node.depth == 0:
        n *= 2
    if set_n:
        n = set_n
    new_nodes, _ = generate_new_states(node, args, task, idx, n, add_buy_now=add_buy_now, only_terminal=only_terminal)
    if node.depth == 0 and args.evaluate:
        # randomly select n nodes as children
        node.children = random.sample(new_nodes, min(n, len(new_nodes)))
    else:
        node.children.extend(new_nodes)

def generate_new_states(node, args, task, idx, n, connect_to_parent=True, add_buy_now=False, only_terminal=False):
    global failed_trajectories
    global expanded_states
    # check if node already has children
    # if node.children:
    #     logging.info("node already has children")
    #     unique_state_nodes = list(set(node.children))
    #     return unique_state_nodes
        
    prompt = generate_prompt(node)
    #print(prompt)
    if only_terminal:
        sampled_actions = []
        sampled_rationales =[]
    elif args.select:
        sampled_actions, sampled_rationales = get_samples_select(task, prompt, "\nAction: ", n, prompt_sample=args.prompt_sample, stop="Observation", ignore_reflections=args.ignore_reflections, args=args)
    else:
        sampled_actions, sampled_rationales = get_samples(task, prompt, "\nAction: ", n, prompt_sample=args.prompt_sample, stop="Observation", ignore_reflections=args.ignore_reflections, args=args)
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    unique_states = {}  # Store unique states here
    added = False
    last_action_idx = prompt.rfind("Action:")
    def add_node(action, rationale):
        # reference the global added and unique_states
        nonlocal added
        nonlocal unique_states 
        nonlocal prompt
        global expanded_states
        if action in prompt and action.strip() not in ['Action: click[Back to Search]', 'Action: click[Attributes]', 'Action: click[Description]', 'Action: click[Reviews]', 'Action: click[Prev]', 'Action: click[Next]', 'Action: click[Buy Now]']:
            # Skip if the action is already in the trajectory
            return
        local_sessions = copy.deepcopy(node.env_state)
        env.sessions = local_sessions
        # logging.info(env.sessions)
        new_state = node.state.copy()  # Make a copy of the parent node's state
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"
        
        # if unique_key in unique_states:
        #     continue  # Skip if this state already exists

        if action_line:
            try:
                #print("res", res)
                res = env.step(idx, action_line)
                expanded_states += 1
                obs = res[0]
                r = res[1]
                done = res[2]
                done_heuristic = res[3]
            except AssertionError:
                obs = 'Invalid action!'
                # print("err")
                r = -1
                done = False
                done_heuristic = False
                # NOTE: This is new! We skip invalid actions 
                # TODO: NEEDS TO BE CHANGED IF WE SWITCH BACK TO NORMAL
                return
            
            if action.startswith('think'):
                observation = 'OK.'

            if obs == 'Invalid action!' or obs == '':
                # Skip if the action
                return
            # Update the new state dictionary
            new_state['action'] = action_line
            new_state['rationale'] = rationale
            new_state['observation'] = obs
            
            env_state_clone = env.clone_state()  # Clone current environment state


            if connect_to_parent:
                new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
            else: # used to generate states for one step lookahead without effecting the rest of the search
                # TODO: change this in the same code used for game of 24
                # need to set depth > 0 in order to print correct
                new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, depth=node.depth + 1)
            new_node.env_state = local_sessions

            if args.ground_truth_rewards:
                print('FAILURE - SHOULD NOT BE ACCESSING THIS')
                if r > 0 or done or action_line == 'click[Buy Now]':
                    logging.info(f"reward:{r}")
                    new_node.is_terminal = True
                    #print("rew", r)
                new_node.reward = r
                new_node.value = r
                new_node.true_reward = r
                unique_states[unique_key] = new_node  # Add this state to unique_states
                
            else: #TODO: CHECK THIS!!!!
                if r > 0 or done or action_line == 'click[Buy Now]':
                    logging.info(f"true reward:{r}")
                    new_node.is_terminal = True
                    
                    # evaluate node with LLM
                    if args.search_algorithm != 'zero_shot':
                        values, rationales = get_values(task, node.question, [generate_prompt(new_node)], args.n_evaluate_sample, ignore_reflections=args.ignore_reflections, terminal=True, args=args, model_no=node.depth + 1)
                        logging.info(f'proxy reward: {values[0]}, {rationales[0]}')
                        
                        # set reward using LLM values
                        new_node.reward = values[0]
                        new_node.value = values[0]
                        new_node.reward_rationale = rationales[0]
                        new_node.true_reward = r
                    else:
                        new_node.reward = 0
                        new_node.value = 0
                        new_node.reward_rationale = ""
                        new_node.true_reward = r
                    
                else:
                    new_node.reward = 0
                    new_node.value = 0
                    new_node.reward_rationale = ""
                    new_node.true_reward = 0
                unique_states[unique_key] = new_node

            if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                trajectory = collect_trajectory(new_node)

                # Check if there is already a failed trajectory with the same reward
                existing_rewards = [t['r'] for t in failed_trajectories]

                if r not in existing_rewards:
                    # print("adding to failed")
                    added = True
                    failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})
    for m, action in enumerate(sampled_actions):
        rationale = sampled_rationales[m]
        add_node(action, rationale)
    
    if add_buy_now and 'Observation:' in prompt:
        # if len(unique_states) == 0:
        if True: # modify to always add because we are using value function for LLM
            logging.info("No new states generated, so we try adding the buy now action.")
            action_new = 'Action: click[Buy Now]'
            rationale_new = "Reflection: The next action is to click on the 'Buy Now' button to proceed with purchasing the selected item."
            add_node(action_new, rationale_new)
            if len(unique_states) > 0:
                logging.info("Buy now action added and used.")
                return list(unique_states.values()), True
        return list(unique_states.values()), False
    return list(unique_states.values()), False


def evaluate_node(node, args, task, idx, ignore_reflections=False):
    #actions_to_node = collect_actions_to_node(node)
    #env.restore_state(actions_to_node, idx)
    
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]

    if args.evaluate:
        votes, _ = get_values(task, node.question, child_prompts, args.n_evaluate_sample, ignore_reflections=args.ignore_reflections, args=args, cache_value=False, model_no=node.depth + 1)
    else:
        votes, _ = get_values(task, node.question, child_prompts, args.n_evaluate_sample, ignore_reflections=args.ignore_reflections, args=args, cache_value=True, model_no=node.depth + 1)

    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    
    max_vote = max(votes) if votes else 1
    if max_vote == 0:
        max_vote = 1  # Avoid division by zero
    terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    for i, condition in enumerate(terminal_conditions):
        if condition == 1:
            votes[i] = max_vote + 1
    
    for i, child in enumerate(node.children):
        child.value = votes[i] / max_vote  # Now safe from division by zero
    
    return sum(votes) / len(votes) if votes else 0

def generate_prompt(ori_node):
    trajectory = []
    node = copy.deepcopy(ori_node)
    question = node.question
    while node:
        new_segment = []
        if node.state['action']:
            new_segment.append(f"Action: {node.state['action']}")
            # print('Action', node.state['action'])
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n\n'.join(reversed(trajectory))