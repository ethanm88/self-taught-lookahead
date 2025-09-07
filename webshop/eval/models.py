import os
import openai
import random
from openai import OpenAI, AzureOpenAI
import backoff 
from transformers import GPT2Tokenizer
global usage
usage = {}
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

vllm_clients = {}

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(port, **kwargs):
    global usage
    global client
    global vllm_clients
    # fetch model from kwargs
    model = kwargs.get("model")
    if "llama" in model.lower() or "qwen" in model.lower():
        if port is None:
            raise Exception("Port is not set")
        if model not in vllm_clients:
            vllm_clients[model] = OpenAI(api_key=os.environ["VLLM_KEY"], base_url=f"http://localhost:{port}/v1")
        vllm_client = vllm_clients[model]
        res = vllm_client.chat.completions.create(**kwargs)
    else: # use Azure
        res = client.chat.completions.create(**kwargs)
    
    # log usage
    if model not in usage:
        usage[model] = {"completion_tokens": 0, "prompt_tokens": 0}
    usage[model]["completion_tokens"] += res.usage.completion_tokens
    usage[model]["prompt_tokens"] += res.usage.prompt_tokens
    return res

def gpt3(prompt, model="text-davinci-002", temperature=1.0, max_tokens=3192, n=1, stop=None) -> list:
    outputs = []
    for _ in range(n):
        response = client.completions.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop
        )
        outputs.append(response.choices[0].text.strip())
    return outputs

def gpt(prompt, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=3192, n=1, stop=None, system_prompt=None, port=None) -> list:
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop, port=port)

def gpt4(prompt, model="gpt-4", temperature=0.2, max_tokens=3192, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=3192, n=1, stop=None, port=None) -> list:
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop, port=port)
        outputs.extend([choice.message.content for choice in res.choices])
    return outputs

    
def gpt_usage():
    global usage
    def calculate_cost(backend, completion_tokens, prompt_tokens):
        cost = 0
        if backend == "gpt-4o":
            cost = completion_tokens / 1000 * 0.01 + prompt_tokens / 1000 * 0.0025
        elif backend == "gpt-35-turbo":
            cost = completion_tokens / 1000 * 0.0015 + prompt_tokens / 1000 * 0.0005
        return cost

    usage_aggregated = {}
    for backend in usage:
        completion_tokens = usage[backend]["completion_tokens"]
        prompt_tokens = usage[backend]["prompt_tokens"]
        cost = calculate_cost(backend, completion_tokens, prompt_tokens)
        usage_aggregated[backend] = {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

    return usage_aggregated
