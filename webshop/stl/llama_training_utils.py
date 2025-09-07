import os
import re
import time
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json
from transformers import AutoTokenizer
from typing import Dict, Union, Any

def load_llama_model(model_name="unsloth/Meta-Llama-3.1-8B-Instruct", base_model=True, max_seq_length=8192, device="cuda"):
    print('Loading model:', model_name)
    if base_model:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
            fix_tokenizer=False
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = max_seq_length,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = None,
            # load_in_4bit = True,
        )
    # Move model to GPU
    model = model.to(device)
    return model, tokenizer

def get_llama_tokenizer(model_name="unsloth/Meta-Llama-3.1-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def train_llama_model(model, tokenizer, train_dataset, eval_dataset, checkpoint_dir, logging_dir, save_dir, max_seq_length=8192, loss_for_save="loss", num_epochs=10, learning_rate=2e-4):
    # Set up the trainer
    trainer = SFTTrainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        tokenizer = tokenizer,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = num_epochs,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            output_dir = checkpoint_dir,
            logging_dir = logging_dir,
            optim = "adamw_8bit",
            seed = 3407,
            load_best_model_at_end=True,  # Load best model at the end of training
            evaluation_strategy="epoch",
            save_strategy="epoch",        # Save checkpoints at regular intervals
            save_steps=20,
            metric_for_best_model=loss_for_save,  # Use loss to determine best model
            greater_is_better=False,      # Lower loss is better
            learning_rate=learning_rate,  # Learning rate
            weight_decay = 0.01,
        ),
    )
    
    # Start timing for training
    start_time = time.time()
    print('Checkpoints:', os.listdir(checkpoint_dir))
    # Train the model
    if os.listdir(checkpoint_dir):
        train_output = trainer.train(resume_from_checkpoint=True)
    else:
        train_output = trainer.train()
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Start timing for evaluation
    eval_start_time = time.time()
    
    # Evaluate the model
    eval_output = trainer.evaluate()
    
    # Calculate evaluation time
    eval_time = time.time() - eval_start_time
    
    # Prepare the results to be saved
    results = {
        "train_loss": train_output.training_loss,
        "train_time": train_time,
        "eval_loss": eval_output["eval_loss"],
        "eval_accuracy": eval_output.get("eval_accuracy", None),  # Optional, based on the model
        "eval_time": eval_time,
    }
    
    # Save the results to a JSON file
    with open(os.path.join(logging_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Save the best model and tokenizer (merged)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained_merged(save_dir, tokenizer, save_method = "merged_16bit",)
    

    return model, tokenizer
    



def inference_llama_model(model, tokenizer, text, num_return_sequences=1, max_new_tokens=256, temperature=0.7, top_p=0.95):
    tokenized_inputs = tokenizer(text, return_tensors = "pt").to("cuda")
    # generate
    output = model.generate(**tokenized_inputs, max_length = tokenized_inputs["input_ids"].shape[-1] + max_new_tokens, temperature=temperature, top_p=top_p, num_return_sequences=num_return_sequences)
    
    filtered_output = []
    for i in range(num_return_sequences):
        decoded_output = tokenizer.decode(output[i], skip_special_tokens = True)
        filtered_output.append(decoded_output.replace(text, "").strip())
    return filtered_output

def get_ratio_str(ratio):
    ratio_str = str(ratio).replace(".", "")
    return ratio_str

def get_formatted_model_name(model_name):
    return model_name.replace("/", "_").lower()

def get_backend_contribution(backend):
    formatted_backend = backend.replace("/", "_").lower()
    if formatted_backend == 'gpt-3_5-turbo':
        backend_contribution = ''
    else:
        backend_contribution = f'_{formatted_backend}'
    return backend_contribution

def fetch_raw_training_file_name(idx, training_data_dir, start_idx=0, num_samples=50, ratio=0.75, model_name="unsloth/Meta-Llama-3.1-8B-Instruct", backend="gpt-3.5-turbo", flag=''):
    formatted_model_name = get_formatted_model_name(model_name)
    backend_contribution = get_backend_contribution(backend)
    file_name = os.path.join(training_data_dir, f'action_outcome_raw_{idx}_{formatted_model_name}{backend_contribution}_{start_idx}_{start_idx + num_samples}.json')
    return file_name

def fetch_formatted_training_file_name(idx, training_data_dir, start_idx=0, num_samples=50, ratio=0.75, model_name="unsloth/Meta-Llama-3.1-8B-Instruct", depth=None, backend="gpt-3.5-turbo", flag=''):
    ratio_str = get_ratio_str(ratio)
    formatted_model_name = get_formatted_model_name(model_name)
    backend_contribution = get_backend_contribution(backend)
    train_dir = os.path.join(training_data_dir, f'train_{idx}_{formatted_model_name}_{start_idx}_{start_idx + num_samples}_{ratio_str}')
    os.makedirs(train_dir, exist_ok=True)
    if depth:
        file_name = os.path.join(train_dir, f'action_outcome_formatted_{idx}_{formatted_model_name}{backend_contribution}_{start_idx}_{start_idx + num_samples}_{ratio_str}_depth={depth}.jsonl')
    else:
        file_name = os.path.join(train_dir, f'action_outcome_formatted_{idx}_{formatted_model_name}{backend_contribution}_{start_idx}_{start_idx + num_samples}_{ratio_str}.jsonl')
    if flag:
        file_name = file_name.replace('.jsonl', f'_{flag}.jsonl')
    return file_name

def fetch_formatted_eval_file_name(idx, training_data_dir, start_idx=0, num_samples=50, ratio=0.75, model_name="unsloth/Meta-Llama-3.1-8B-Instruct", depth=None, backend="gpt-3.5-turbo", flag=''):
    ratio_str = get_ratio_str(ratio)
    formatted_model_name = get_formatted_model_name(model_name)
    backend_contribution = get_backend_contribution(backend)
    eval_dir = os.path.join(training_data_dir, f'eval_{idx}_{formatted_model_name}_{start_idx}_{start_idx + num_samples}_{ratio_str}')
    os.makedirs(eval_dir, exist_ok=True)
    if depth:
        file_name = os.path.join(eval_dir, f'action_outcome_formatted_{idx}_{formatted_model_name}{backend_contribution}_{start_idx}_{start_idx + num_samples}_{ratio_str}_depth={depth}_eval.jsonl')
    else:
        file_name = os.path.join(eval_dir, f'action_outcome_formatted_{idx}_{formatted_model_name}{backend_contribution}_{start_idx}_{start_idx + num_samples}_{ratio_str}_eval.jsonl')
    if flag:
        file_name = file_name.replace('.jsonl', f'_{flag}.jsonl')
    return file_name


# llama wrapper class
class LlamaModel:
    def __init__(self, model, tokenizer, max_seq_length=2048, device="cuda"):
        self.model = model
        FastLanguageModel.for_inference(self.model)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device
        
    def train(self, train_dataset, eval_dataset, checkpoint_dir, logging_dir, save_dir):
        self.model, self.tokenizer = train_llama_model(self.model, self.tokenizer, train_dataset, eval_dataset, checkpoint_dir, logging_dir, save_dir, max_seq_length=self.max_seq_length)
    
    def inference(self, text, n=1, max_new_tokens=50, temperature=1, top_p=0.95):
        inference_results = inference_llama_model(self.model, self.tokenizer, text, num_return_sequences=n, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        return inference_results