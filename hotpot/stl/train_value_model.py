import os
import json
import random
import argparse
from llama_training_utils import get_llama_tokenizer, load_llama_model, train_llama_model, fetch_formatted_training_file_name
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--training_data_dir', type=str, default='improvement_data')
parser.add_argument('--iteration', type=int, default=0)
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--split', type=float, default=0.1)
parser.add_argument('--max_depth', type=int, default=4)
parser.add_argument('--model_name', type=str, default='unsloth/Meta-Llama-3.1-8B-Instruct')
parser.add_argument('--backend', type=str, default='gpt-3.5-turbo')
parser.add_argument('--start_idx', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--num_epochs', type=int, default=20)
args = parser.parse_args()



def prepare_training_data(depth):
    tokenizer = get_chat_template(
        get_llama_tokenizer(),
        chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    train_file = fetch_formatted_training_file_name(args.iteration, args.training_data_dir, ratio=args.ratio, model_name=args.model_name, start_idx=args.start_idx, num_samples=args.num_samples, depth=depth, backend=args.backend)
    print('Training file:', train_file)
    with open(train_file, 'r') as f:
        formatted_data = json.load(f)
    # shuffle the data
    random.shuffle(formatted_data)
    dataset = Dataset.from_dict({"conversations": formatted_data})
    dataset = dataset.map(formatting_prompts_func, batched = True)
    return dataset

def form_eval_set(dataset, split):
    # Get the total number of examples in the dataset
    num_examples = len(dataset)

    sample_size = max(1, int(num_examples * split))

    # Randomly sample indices
    random_indices = random.sample(range(num_examples), sample_size)

    # Select the examples corresponding to the sampled indices
    eval_dataset = dataset.select(random_indices)
    return eval_dataset
if __name__ == '__main__':
    # print devices
    import torch
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Devices available: ", torch.cuda.device_count())
    device = "cuda:0"
    for depth in tqdm(range(1, args.max_depth + 1)):
        # create dataset for unsloth
        dataset = prepare_training_data(depth)
        LOSS_FOR_SAVE = "loss"

        # form train and eval sets
        train_dataset = dataset
        eval_dataset = form_eval_set(dataset, args.split)
        # print length of train and eval sets
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Eval dataset length: {len(eval_dataset)}")

        
        # load base model
        model, tokenizer = load_llama_model(model_name="unsloth/Meta-Llama-3.1-8B-Instruct", base_model=True, max_seq_length=8192, device=device)
        ratio_str = str(args.ratio).replace('.', '')
        
        # train the model
        base_dir = f"/data/emendes3/lookahead_tuning/LanguageAgentTreeSearch/hotpot_lookahead/value_models"
        checkpoint_dir = os.path.join(base_dir, f"checkpoints/model_{args.model_name}_{args.backend}_{args.iteration}_{ratio_str}_start={args.start_idx}_num_samples={args.num_samples}_depth={depth}_{LOSS_FOR_SAVE}")
        logging_dir = os.path.join(base_dir, f"logs/model_{args.model_name}_{args.backend}_{args.iteration}_{ratio_str}_start={args.start_idx}_num_samples={args.num_samples}_depth={depth}_{LOSS_FOR_SAVE}")
        save_dir = os.path.join(base_dir, f"model_{args.model_name}_{args.backend}_{args.iteration}_{ratio_str}_start={args.start_idx}_num_samples={args.num_samples}_depth={depth}_{LOSS_FOR_SAVE}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        train_llama_model(model, tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset, checkpoint_dir=checkpoint_dir, logging_dir=logging_dir, save_dir=save_dir, loss_for_save=LOSS_FOR_SAVE, num_epochs=args.num_epochs, learning_rate=args.learning_rate, depth=depth)
        