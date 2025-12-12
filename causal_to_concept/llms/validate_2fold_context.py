import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import argparse
import logging
from datetime import datetime
from functools import partial
from datasets import load_dataset
import gc  # Garbage collection
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
# from transformers import GPT2LMHeadModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, AutoProcessor
from vae import VAE, vae_loss_function, train_vae, test_vae

# from accelerate import Accelerator
# accelerator = Accelerator()

import sys
sys.path.append('../')
from utils_context import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, get_activations
from utils_context import get_special_directions, get_matrix_directions, train_vae_and_extract_mu, get_top_heads_pns, LocalStore, build_local_interventions_for_train_idx
# import llama

HF_NAMES = {
    'llama_1B': 'meta-llama/Llama-3.2-1B', #meta-llama/Llama-3.2-1B
    'llama_3B': 'meta-llama/Llama-3.2-3B',
    'llama_7B': 'baffo32/decapoda-research-llama-7B-hf', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'vicuna_pns': '/work/hdd/bcxt/yian3/models/vicuna_pns_finetuned',
    'COV_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_finetuned_epoch5',
    'COV_pns_use_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_finetuned_epoch5',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'gemma3_4B': 'google/gemma-3-4b-it',
    'vicuna_13B_toxigen_vicuna_72_0.01_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_72_True_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_72_0.01_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_72_False_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.01_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_0.01_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'vicuna_13B_toxigen_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_72_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_72_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_72_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_72_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_logpns_18_True_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'llama3_8B_hate_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_hate_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'mistral_7B': 'mistralai/Mistral-7B-v0.1',
    'vicuna_13B_hate_vicuna_36_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_36_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_36_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_36_True_0.0001_finetuned_epoch5',  
    'vicuna_13B_hate_vicuna_18_0.0001_acc': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_accuracy_18_False_0.0001_finetuned_epoch5',
    'vicuna_13B_hate_vicuna_18_0.0001_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_hate_vicuna_logpns_18_True_0.0001_finetuned_epoch5',  
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_bce_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_bce_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_False_0.05_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_False_0.05_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_epoch5',
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5': '/work/hdd/bcxt/yian3/toxic/models/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5',
}

class LazyLocalInterventions:
    """
    Acts like a dictionary but computes values on-the-fly to save RAM.
    Only holds one computed item in memory at a time.
    """
    def __init__(self, test_idxs, builder_fn):
        self.test_idxs = list(test_idxs)
        self.builder_fn = builder_fn
        self._cache = {} # Caches the last accessed item
        self._access_count = 0

    def keys(self):
        return self.test_idxs

    def __getitem__(self, key):
        # Return cached if available (handles repeated access for same row)
        if key in self._cache:
            return self._cache[key]
        
        # Compute on demand
        val = self.builder_fn(row_i=key)
        
        # CLEAR CACHE to prevent OOM (The Fix)
        self._cache.clear() 
        self._cache[key] = val
        
        # Optional: Periodic GC to ensure memory is returned to OS
        self._access_count += 1
        if self._access_count % 50 == 0:
            gc.collect()
            
        return val
    
    def __len__(self):
        return len(self.test_idxs)
    
    def items(self):
        # Warning: Iterating items will trigger computation of everything
        for k in self.test_idxs:
            yield k, self[k]

def setup_logging(args):
    """Setup logging to both file and console"""
    # Create logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a unique log filename based on timestamp and key parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{args.model_name}_{args.dataset_name}_seed{args.seed}_fold{args.num_fold}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    logger.info(f"Run parameters: model={args.model_name}, dataset={args.dataset_name}, seed={args.seed}, num_fold={args.num_fold}")
    
    return logger

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--lam', type=float, default=0.0, help='lambda, intervention strength')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--mode', type=str, default='general', help='if get top heads or get activations')
    parser.add_argument('--dataset_name', type=str, default='toxigen_vicuna', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='toxigen_vicuna', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--heads_path', type=str, 
                      default="/work/hdd/bcxt/yian3/toxic/features/heads/False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy",
                      help='Path to selected heads numpy file')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--judge_name', type=str, default="gpt-4o-mini", required=False)
    parser.add_argument('--info_name', type=str, default="gpt-4o-mini", required=False)
    parser.add_argument('--use_special_direction', action='store_true', default=False)
    parser.add_argument('--use_pns', action='store_true', default=False)
    parser.add_argument('--use_mat_direction', action='store_true', default=False)
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of examples to use for testing (for faster testing)')
    parser.add_argument('--use_local_interventions', action='store_true', default=False)
    parser.add_argument('--local_store_dir', type=str, default=None, help='Path to local store directory')
    parser.add_argument('--local_k', type=int, default=20, help='Local builder k')
    parser.add_argument('--local_tau', type=float, default=20.0, help='Local builder tau')
    parser.add_argument('--task', type=str, default='continuation', choices=['continuation', 'rephrase'], help='Task type: continuation or rephrase')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args)
    
    # Extract original model name and head selection method
    orig_model = None
    head_selection = None
    
    if 'llama3_8B' in args.model_name:
        orig_model = 'llama3_8B'
    elif 'vicuna_13B' in args.model_name:
        orig_model = 'vicuna_13B'
    elif 'llama_7B' in args.model_name:
        orig_model = 'llama_7B'
    elif 'llama_1B' in args.model_name:
        orig_model = 'llama_1B'
    elif 'llama_3B' in args.model_name:
        orig_model = 'llama_3B'
    elif 'alpaca_7B' in args.model_name:
        orig_model = 'alpaca_7B'
    elif 'vicuna_7B' in args.model_name:
        orig_model = 'vicuna_7B'
    elif 'llama2_chat_7B' in args.model_name:
        orig_model = 'llama2_chat_7B'
    elif 'llama2_chat_13B' in args.model_name:
        orig_model = 'llama2_chat_13B'
    elif 'llama2_chat_70B' in args.model_name:
        orig_model = 'llama2_chat_70B'
    elif 'llama3_8B_instruct' in args.model_name:
        orig_model = 'llama3_8B_instruct'
    elif 'llama3_70B' in args.model_name:
        orig_model = 'llama3_70B'
    elif 'llama3_70B_instruct' in args.model_name:
        orig_model = 'llama3_70B_instruct'
    elif 'gemma3_4B' in args.model_name:
        orig_model = 'gemma3_4B'
    elif 'mistral_7B' in args.model_name:
        orig_model = 'mistral_7B'
    else:
        orig_model = args.model_name  # fallback
    
    # Determine head selection method
    if 'pns' in args.model_name:
        head_selection = 'pns'
    elif 'acc' in args.model_name:
        head_selection = 'acc'
    else:
        head_selection = 'orig'  # fallback
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.dataset_name == "toxigen":
        dataset = load_dataset("json", data_files="../../dataset/toxiGen.json")["train"]
        df = pd.read_csv(f'./TruthfulQA/{args.dataset_name}.csv')
    elif args.dataset_name == "hate":
        dataset = load_dataset("json", data_files="../../dataset/implicitHate.json")["train"]
        df = pd.read_csv(f'./TruthfulQA/shuffled_{args.dataset_name}.csv')
    elif args.dataset_name == "toxigen_vicuna":
        # df = pd.read_csv(f'./TruthfulQA/{args.dataset_name}.csv')
        toxigen_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
        with open(toxigen_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif args.dataset_name == "hate_vicuna":
        hate_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
        with open(hate_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif args.dataset_name == "paradetox":
        paradetox_path = f'/work/hdd/bcxt/yian3/toxic/features/{args.dataset_name}_texts.json'
        with open(paradetox_path) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)


    # shuffle data to avoid class imbalance
    # np.random.seed(42)
    # indices = np.random.permutation(len(df))
    
    # df = df.iloc[indices].reset_index(drop=True)
    # df.to_csv(f'./TruthfulQA/shuffled_{args.dataset_name}.csv')
    # df.to_json("../../dataset/shuffled_implicitHate.json", orient="records", lines=False)
    # get two folds using numpy
    # fold_idxs = np.array_split(np.arange(len(df))[:100], args.num_fold)
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)


    # create model
    model_name = HF_NAMES[args.model_name]
    MODEL = model_name if not args.model_dir else args.model_dir
    if "gpt2" in args.model_name:
        model = GPT2LMHeadModel.from_pretrained(
            MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto"
        )
    elif "gemma3" in args.model_name:
        model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, device_map="auto").eval()
        tokenizer = AutoProcessor.from_pretrained(MODEL)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # if args.model_name == 'llama_1B' or args.model_name == 'llama_3B' or args.model_name == 'llama3_8B':
        #     tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # else:
        #     tokenizer = LlamaTokenizer.from_pretrained(MODEL)
        model = LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
    
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    try:
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
    except:
        num_layers = 12  # default for tiny_gpt2
        num_heads = 12    
    
    # load activations, derived from get_activations.py 
    if args.dataset_name == "hate_vicuna" or args.dataset_name == "toxigen_vicuna":
        head_wise_activations = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_head_wise.npy") 
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        labels = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_labels.npy") 
        
        activations_dataset = args.dataset_name # if args.activations_dataset is None else args.activations_dataset
        tuning_activations = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_head_wise.npy") 
        tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_labels.npy") 
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")
         
    n,l,h,d = head_wise_activations.shape
    input_dim = l * h * d
    c_path = f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_c_all.pt"


    logger.info(f"No cached c_all found. Training VAE and saving to {c_path}")
    _, _, head_wise_c = train_vae_and_extract_mu(head_wise_activations, labels, input_dim, z_dim=32, h_dim1=128, h_dim2=64,
                            batch_size=128, lr=1e-3, vae_epochs=10, dataset_name=args.dataset_name, model_name=args.model_name, mode='valid', device='cuda')
    logger.info(f"head_wise_c size: {head_wise_c.size()}, model_name: {args.model_name}")
    # separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, categories[:100], args.dataset_name)
    separated_head_wise_activations, separated_labels, separated_head_wise_c, idxs_to_split_at = get_activations(labels, head_wise_activations, head_wise_c, args.dataset_name, args.model_name, args.alpha)
    
    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):
        if i == 1: 
            break

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        logger.info(f"Running fold {i}")

        # pick a val set using numpy
        # train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
        train_set_idxs = train_idxs
        val_set_idxs = None

        # Create necessary directories
        os.makedirs(f'results_dump/answer_dump/{args.dataset_name}/{orig_model}', exist_ok=True)
        os.makedirs(f'results_dump/summary_dump/{args.dataset_name}/{orig_model}', exist_ok=True)
        
        # # save train and test splits
        # df.iloc[train_set_idxs].to_csv(f"splits/{args.model_name}_{args.dataset_name}_fold_{i}_train_seed_{args.seed}.csv", index=False)
        # df.iloc[val_set_idxs].to_csv(f"splits/{args.model_name}_{args.dataset_name}_fold_{i}_val_seed_{args.seed}.csv", index=False)
        # df.iloc[test_idxs].to_csv(f"splits/{args.model_name}_{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get heads first
        args.heads_path = f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_fold_{i}.npy'
        if os.path.exists(args.heads_path):
            selected_heads = np.load(args.heads_path)
            top_heads = selected_heads[:args.num_heads] if len(selected_heads) > args.num_heads else selected_heads
            probes = None
        else:
            if args.use_pns:
                top_heads, pns_scores = get_top_heads_pns(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels,
                            separated_head_wise_c, num_layers, num_heads, num_to_intervene=args.num_heads, lambda_reg=1e-4, sigma_sq=1.0, seed=42, use_random_dir=args.use_random_dir)
                np.save(f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_fold_{i}_pns_scores.npy', pns_scores)
                probes = None
            else:
                top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
                np.save(f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_fold_{i}.npy', top_heads)
        logger.info(f"Heads saved to: {f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_fold_{i}.npy'}")
        # get directions (only for selected heads)
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, selected_heads=top_heads)
        elif args.use_special_direction:
            direct_path = f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_special_direction.npy"
            if os.path.exists(direct_path):
                print(f"Loading direct from {direct_path}")
                com_directions = np.load(direct_path)
            else:
                print(f"No cached direct found. Saving to {direct_path}")
                com_directions = get_special_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, df)
                np.save(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_special_direction.npy", com_directions)
            # com_directions = get_special_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, df, selected_heads=top_heads)
        elif args.use_mat_direction:
            com_directions = get_matrix_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, selected_heads=top_heads)
        else:
            com_directions = None
        pns_suffix = "pns" if args.use_pns else "no_pns"
        store = LocalStore(f"{args.model_name}_{args.dataset_name}_{pns_suffix}", f"/work/hdd/bcxt/yian3/toxic/local_store_toxigen", top_heads, device='cuda', K_default=128)
        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, args.use_mat_direction, args.use_special_direction, com_directions)
        
        # Load precomputed prompt encodings if using special directions
        all_prompt_encodings = None
        if args.use_special_direction:
            prompt_encodings_path = f"/work/hdd/bcxt/yian3/toxic/local_store_toxigen/texts.npy"
            if os.path.exists(prompt_encodings_path):
                logger.info(f"Loading precomputed prompt encodings from {prompt_encodings_path}")
                all_prompt_encodings = np.load(prompt_encodings_path)
            else:
                logger.warning(f"Precomputed prompt encodings not found at {prompt_encodings_path}")
                logger.warning(f"Please run: python build_prompt_encodings.py --dataset_name {args.dataset_name} --model_name {args.model_name}")
                raise FileNotFoundError(f"Precomputed prompt encodings not found. Run build_prompt_encodings.py first.")
        
        logger.info("Initializing LazyLocalInterventions (No pre-computation)")
        # Create a partial function with all fixed arguments
        builder_partial = partial(
            build_local_interventions_for_train_idx,
            df=df, 
            store=store, 
            interventions_global=interventions, 
            lam=args.lam,
            tau=args.local_tau, 
            K=args.local_k, 
            use_contrastive=False, 
            v_global=com_directions, 
            kappa=0.5, 
            num_heads=num_heads, 
            separated_head_wise_activations=separated_head_wise_activations, 
            separated_labels=separated_labels,
            use_special_direction=args.use_special_direction, 
            use_mat_direction=args.use_mat_direction,
            prompt_encodings=all_prompt_encodings
        )
        local_interventions = LazyLocalInterventions(test_idxs, builder_partial)
        logger.info("Finished initializing LazyLocalInterventions")

        def create_intervention_fn(sample_idx):
            """Create an intervention function for a specific sample"""
            def lc_modulated_vector_add(_head_output, layer_name, start_edit_location='lt', prompt_encoding=None):
                head_output = _head_output.detach().type(torch.float32)
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                if "gpt2" in args.model_name:
                    layer = int(layer_name.split('.')[1])  # e.g., 'transform.h.3'
                else:
                    layer = int(layer_name.split('.')[2])

                if prompt_encoding is not None: # use_special_direction
                    assert prompt_encoding.shape == (384,)
                    prompt_encoding = torch.FloatTensor(prompt_encoding).to(head_output.device.index).reshape(-1, 384)
                
                # Use the specific sample_idx to get the right local_interventions
                sample_interventions = local_interventions[sample_idx]
                for head, direction, proj_val_std in sample_interventions[layer_name]:
                    if len(direction.shape) == 2:
                        logger.debug(f"Processing head={head}, direction.shape={direction.shape}")
                        activations = torch.FloatTensor(tuning_activations[:,layer,head,:]).to(head_output.device.index)
                        direction = torch.FloatTensor(direction).to(head_output.device.index) #128 x 384
                        logger.debug(f"activations.shape={activations.shape}, direction.shape={direction.shape}, head_output.shape={head_output.shape}")
                        logger.debug(f"start_edit_location={start_edit_location}")
                        if start_edit_location == 'lt':
                            if prompt_encoding is None:
                                # Since batch_size=1, we use [0] to get the single sample
                                direction_to_add = head_output[0, -1, head, :] @ direction.T # shape: [128]
                                direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add)
                            else: 
                                direction_to_add = prompt_encoding @ direction.T # 1 x 128
                                # Normalize 2D tensor
                                direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=1, keepdim=True)

                            # compute stddev online
                            # Ensure direction_to_add is 2D for matrix multiplication
                            if len(direction_to_add.shape) == 1:
                                direction_to_add = direction_to_add.unsqueeze(0)  # [128] -> [1, 128]
                            proj_vals = activations @ direction_to_add.T # batch_all x 1
                            proj_val_std = torch.std(proj_vals, axis=0).reshape(1, -1) # 1 x 1

                        else:
                            if prompt_encoding is None:
                                direction_to_add = torch.einsum('bij,jk->bik', head_output[0, start_edit_location:, head, :], direction.T)
                                direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=2)[:, :, None] # 1 x location_indices x 128
                            else:
                                direction_to_add = prompt_encoding @ direction.T # 1 x 128
                                direction_to_add = direction_to_add.unsqueeze(1).repeat(1, head_output.shape[1] - start_edit_location, 1) # 1 x location_indices, 128
                                direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=1).reshape(-1, 1) # batch x location_indices x 128

                            # compute stddev online
                            proj_vals = torch.einsum('Bk,bik->Bbi', activations, direction_to_add)
                            proj_val_std = torch.std(proj_vals, axis=0)[:, :, None] 

                        proj_val_std = torch.Tensor(proj_val_std).to(head_output.device.index)
                        if start_edit_location == 'lt': 
                            head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                        else:
                            head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add 

                    else:
                        assert (proj_val_std is not None)
                        direction_to_add = torch.FloatTensor(direction).to(head_output.device.index) # 128 x 1
                        if start_edit_location == 'lt':
                            head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                        else:
                            head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
                
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output.type(torch.float16)
            
            return lc_modulated_vector_add

        # Create a cleaner filename for the output path
        output_filename = f'{args.model_name}_{head_selection}_top{args.num_heads}_alpha{args.alpha}_lam{args.lam}_fold{i}'
        
        if args.use_center_of_mass:
            output_filename += '_com'
        if args.use_random_dir:
            output_filename += '_random'
        if args.use_honest:
            output_filename = 'honest_' + output_filename
        if args.use_special_direction:
            output_filename += '_special'
        if args.use_mat_direction:
            output_filename += '_mat'
        
        # Add task to filename to distinguish between continuation and rephrase
        if args.task == 'rephrase':
            output_filename += '_rephrase'
        # Note: 'continuation' is the default, so we don't add a suffix for it
                    
        prefix = "eval_pns_" if args.use_pns else "eval_"
        if args.dataset_name == 'hate':
            input_path = f'splits/shuffled_{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/{args.dataset_name}/{orig_model}/{output_filename}.csv'
            summary_path = f'results_dump/summary_dump/{args.dataset_name}/{orig_model}/{output_filename}.csv'
        elif args.dataset_name == 'toxigen':
            input_path = f'splits/{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/{args.dataset_name}/{orig_model}/{prefix}_answer_no_inst_{output_filename}.csv'
            summary_path = f'results_dump/summary_dump/{args.dataset_name}/{orig_model}/{output_filename}.csv'
        elif args.dataset_name == 'toxigen_vicuna' or args.dataset_name == 'hate_vicuna':
            input_path = f'splits/{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/{args.dataset_name}/{orig_model}/{prefix}_answer_local_{args.local_k}_{output_filename}.csv'
            summary_path = f'results_dump/summary_dump/{args.dataset_name}/{orig_model}/{prefix}_summary_local_{args.local_k}_{output_filename}.csv'
            
        logger.info(f"input_path: {input_path}")
        logger.info(f"output_path: {output_path}")
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            ['toxic', 'sense'], #, 'mc'
            input_path, 
            output_path, 
            summary_path, 
            device="cuda", 
            interventions=interventions, 
            local_interventions=local_interventions,
            intervention_fn=create_intervention_fn, 
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_special_direction=args.use_special_direction,
            instruction_prompt=True,
            max_examples=args.max_examples,
            task=args.task
        )

        logger.info(f"FOLD {i} Results:")
        logger.info(f"\n{curr_fold_results}")

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    if args.mode != 'get_top_heads':
        results = np.array(results)
        final = results.mean(axis=0)

if __name__ == "__main__":
    main()
