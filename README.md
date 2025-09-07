# Language Models can Self-Improve at State-Value Estimation for Better Search
This is the code repository for [Language Models can Self-Improve at State-Value Estimation for Better Search](https://arxiv.org/pdf/2503.02878).


## Setup & Installation
Use the following command to install required packages:
```
conda create -n stl python=3.10
conda activate stl
pip install -r requirements.txt
```

## Reproducing Results Tables
To reproduce the results tables for WebShop and HotpotQA follow the steps below:

### WebShop
```
cd webshop
python process_results_webshop.py # produce the figures in the Table
python bootstrap_test_webshop.py # produce the bootstrap test significance results
```

### HotpotQA
```
cd hotpot
python process_results_hotpot.py # produce the figures in the Table
python bootstrap_test_hotpot.py # produce the bootstrap test significance results
```

## Training STL Value Models
To train STL value models, run the following commands. Note that training only requires a single A40 GPU, you can specify the GPU to use with `CUDA_EXPORT_VISIBLE_DEVICE=n` while training the models:

### WebShop
```
cd webshop/stl
python train_value_model.py
```
This will populate models in `./webshop/stl/stl_value_models`. Specifying the `no_lookahead` and `no_rationalization` flags to `train_value_model.py` will train the models that were used for the ablations.

### HotpotQA
```
cd hotpotqa/stl
python train_value_model.py
```
This will populate models in `./webshop/stl/stl_value_models`.

## Recomputing Results with Trained Models
> NOTE: Recomputing results from the paper with trained models requires access to an AzureOpenAI. We are working on porting this over to OpenAI endpoints.

Follow the below steps to rerun STL on WebShop:

1. Start up a local WebShop development server by following the instructions from the original repository: https://github.com/princeton-nlp/WebShop.
**IMPORTANT:** Ensure that the server is deployed to port `3000` - the evaluation scripts will look for it there. You can optionally change the port in `WEBSHOP_URL` variable in `search.py`.
2. Setup deployments with gpt-3.5-turbo and gpt-4o models name the former `gpt-35-turbo` and the latter `gpt-4o`.
3. Run the following commands to kick off the eval process
```
cd webshop/eval

# vllm settings
export VLLM_KEY="<YOUR_KEY_HERE>"
export VLLM_LOGGING_LEVEL=DEBUG

# azure settings (from your resource)
export AZURE_OPENAI_KEY="<YOUR_API_KEY>"
export AZURE_OPENAI_VERSION="2024-02-15-preview" 
export AZURE_OPENAI_ENDPOINT="<YOUR_API_ENDPOINT>"

# run eval
bash eval_stl.sh gpt-35-turbo # either gpt-4o or gpt-35-turbo
```

### Notes:
* This procedure will generate a file `saved_results/greedy_evaluation_results_stl_unsloth_meta-llama-3.1-8b-instruct_{policy}_0-500.jsonl` where `policy` is either `gpt-35-turbo` or `gpt-4o` based on what is specified. 
* This results file will replace the one already present in the repository. 
* You can calculate the metrics used in the paper by following the steps above about reproducing the results tables
* Expanded nodes are stored in `./webshop/eval/intermediate_eval_results`, to prevent recomputation when the script is rerun



## Citation
If you use this code, please cite our work:
```
@misc{mendes2025languagemodelsselfimprovestatevalue,
      title={Language Models can Self-Improve at State-Value Estimation for Better Search}, 
      author={Ethan Mendes and Alan Ritter},
      year={2025},
      eprint={2503.02878},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.02878}, 
}
```
