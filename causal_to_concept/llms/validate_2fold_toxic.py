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
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
# from transformers import GPT2LMHeadModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, AutoProcessor
from vae import VAE, vae_loss_function, train_vae, test_vae

# from accelerate import Accelerator
# accelerator = Accelerator()

import sys
sys.path.append('../')
from utils_toxic import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, get_activations
from utils_toxic import get_special_directions, get_matrix_directions, train_vae_and_extract_mu, get_top_heads_pns
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
    'llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_useKL_True_0.05_epoch5': '/work/hdd/bcxt/yian3/toxic/models/tox_par/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_useKL_True_0.05_epoch5',
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
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
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.0)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--use_special_direction', action='store_true', default=False)
    parser.add_argument('--use_pns', action='store_true', default=False)
    parser.add_argument('--use_mat_direction', action='store_true', default=False)
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of examples to use for testing (for faster testing)')
    args = parser.parse_args()
    
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
        with open(paradetox_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    
    # df = df.iloc[indices].reset_index(drop=True)
    # df.to_csv(f'./TruthfulQA/shuffled_{args.dataset_name}.csv')
    # df.to_json("../../dataset/shuffled_implicitHate.json", orient="records", lines=False)
    # get two folds using numpy
    # fold_idxs = np.array_split(np.arange(len(df))[:100], args.num_fold)
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)


    # create model
    model_name = HF_NAMES["honest_" + args.model_name if args.use_honest else args.model_name]
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
    
    # Set pad_token if not already set
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # define number of layers and heads
    try:
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
    except:
        num_layers = 12  # default for tiny_gpt2
        num_heads = 12    
    
    # load activations 
    if args.dataset_name == "hate_vicuna" or args.dataset_name == "toxigen_vicuna":
        head_wise_activations = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_head_wise.npy") # [:200]
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        labels = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_labels.npy") # [:200]
        
        activations_dataset = args.dataset_name # if args.activations_dataset is None else args.activations_dataset
        tuning_activations = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_head_wise.npy") # [:200]
        tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_labels.npy") # [:200]
    elif args.dataset_name == "paradetox":
        head_wise_activations = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_head_wise.npy") # [:200]
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        labels = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_labels.npy") # [:200]
        
        activations_dataset = args.dataset_name # if args.activations_dataset is None else args.activations_dataset
        tuning_activations = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_head_wise.npy") # [:200]
        tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_labels.npy") # [:200]

    n,l,h,d = head_wise_activations.shape
    input_dim = l * h * d
    c_path = f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{args.dataset_name}_c_all.pt"
    print("=================================================================")

    if os.path.exists(c_path):
        print(f"Loading cached c_all from {c_path}")
        head_wise_c = torch.load(c_path)
        print("head_wise_c size", head_wise_c.size(), args.model_name)
    else:
        print(f"No cached c_all found. Training VAE")
        _, _, head_wise_c = train_vae_and_extract_mu(head_wise_activations, labels, input_dim, z_dim=32, h_dim1=128, h_dim2=64,
                                batch_size=128, lr=1e-3, vae_epochs=10, dataset_name=args.dataset_name, model_name=args.model_name, mode='valid', device='cuda')
        print("head_wise_c size", head_wise_c.size(), args.model_name)
    # separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, categories[:100], args.dataset_name)
    separated_head_wise_activations, separated_labels, separated_head_wise_c, idxs_to_split_at = get_activations(labels, head_wise_activations, head_wise_c, args.dataset_name, args.model_name, args.alpha)
    
    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):
        if i == 1: 
            break

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # # pick a val set using numpy
        # train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        # val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
        # train_set_idxs = train_set_idxs.astype(int)
        # val_set_idxs = val_set_idxs.astype(int)
        train_set_idxs = train_idxs
        val_set_idxs = None

        # Create necessary directories
        os.makedirs(f'results_dump/answer_dump/{args.dataset_name}/{orig_model}', exist_ok=True)
        os.makedirs(f'results_dump/summary_dump/{args.dataset_name}/{orig_model}', exist_ok=True)
        
        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        elif args.use_special_direction:
            direct_path = f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_special_direction.npy"
            if os.path.exists(direct_path):
                print(f"Loading direct from {direct_path}")
                com_directions = np.load(direct_path)
            else:
                print(f"No cached direct found. Saving to {direct_path}")
                com_directions = get_special_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, df)
                np.save(f"/work/hdd/bcxt/yian3/toxic/features/{args.model_name}_{activations_dataset}_special_direction.npy", com_directions)
        elif args.use_mat_direction:
            com_directions = get_matrix_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        else:
            com_directions = None
        # breakpoint()
        print("Finished computing com_directions of shape", com_directions.shape)
        if args.mode == 'get_top_heads':
            if args.use_pns:
                top_heads, pns_scores = get_top_heads_pns(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels,
                            separated_head_wise_c, num_layers, num_heads, num_to_intervene=args.num_heads, lambda_reg=1e-4, sigma_sq=1.0, seed=42, use_random_dir=args.use_random_dir)
                np.save(f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_pns_scores.npy', pns_scores)
                probes = None
            else:
                top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
            np.save(f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_top_heads.npy', top_heads)
           
            continue

        args.heads_path = f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_top_heads.npy'
        if os.path.exists(args.heads_path):
            selected_heads = np.load(args.heads_path)
            top_heads = selected_heads[:args.num_heads] if len(selected_heads) > args.num_heads else selected_heads
            probes = None
        else:
            if args.use_pns:
                top_heads, pns_scores = get_top_heads_pns(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels,
                            separated_head_wise_c, num_layers, num_heads, num_to_intervene=args.num_heads, lambda_reg=1e-4, sigma_sq=1.0, seed=42, use_random_dir=args.use_random_dir)
                np.save(f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_pns_scores.npy', pns_scores)
                probes = None
            else:
                top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        np.save(f'/work/hdd/bcxt/yian3/toxic/features/heads/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_top_heads.npy', top_heads)
        print("Heads intervened: ", top_heads)
        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, args.use_mat_direction, args.use_special_direction, com_directions)
        print("Finished computing interventions dict")

        def lt_modulated_vector_add(_head_output, layer_name, start_edit_location='lt', prompt_encoding=None):
            head_output = _head_output.detach().type(torch.float32)
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            if "gpt2" in args.model_name:
                layer = int(layer_name.split('.')[1])  # e.g., 'transform.h.3'
            else:
                layer = int(layer_name.split('.')[2])

            if prompt_encoding is not None: # use_special_direction
                assert prompt_encoding.shape == (384,)
                prompt_encoding = torch.FloatTensor(prompt_encoding).to(head_output.device.index).reshape(-1, 384)
            
            for head, direction, proj_val_std in interventions[layer_name]:
                if len(direction.shape) == 2: # use_mat_direction or use_special_direction
                    activations = torch.FloatTensor(tuning_activations[:,layer,head,:]).to(head_output.device.index) # batch_all x 128
                    assert (proj_val_std is None)
                    direction = torch.FloatTensor(direction).to(head_output.device.index) #128 x 384
                    if start_edit_location == 'lt':
                        if prompt_encoding is None:
                            direction_to_add = head_output[:, -1, head, :] @ direction.T # batch x 128
                        else: # use_special_direction
                            # uses batch size = 1
                            direction_to_add = prompt_encoding @ direction.T # 1 x 128
                        direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=1).reshape(-1, 1)
                        # compute stddev online
                        proj_vals = activations @ direction_to_add.T # batch_all x batch
                        proj_val_std = torch.std(proj_vals, axis=0).reshape(1, -1) # batch x 1
                        # print("proj_val_std has shape", proj_val_std.shape)

                    else:
                        if prompt_encoding is None:
                            direction_to_add = torch.einsum('bij,jk->bik', head_output[:, start_edit_location:, head, :], direction.T)
                            direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=2)[:, :, None] # batch x location_indices x 128
                        else:
                            direction_to_add = prompt_encoding @ direction.T # 1 x 128
                            direction_to_add = direction_to_add.unsqueeze(1).repeat(1, head_output.shape[1] - start_edit_location, 1) # 1 x location_indices, 128
                            direction_to_add = direction_to_add / torch.linalg.norm(direction_to_add, axis=2)[:, :, None] # batch x location_indices x 128

                        # compute stddev online
                        proj_vals = torch.einsum('Bk,bik->Bbi', activations, direction_to_add)
                        proj_val_std = torch.std(proj_vals, axis=0)[:, :, None] # batch x location_indices x 1
                        # print("proj_val_std has shape", proj_val_std.shape)

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

        # Create a cleaner filename for the output path
        output_filename = f'{args.model_name}_{head_selection}_top{args.num_heads}_alpha{args.alpha}_fold{i}'
        
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
            output_path = f'results_dump/answer_dump/{args.dataset_name}/{orig_model}/{prefix}_answer_no_inst_{output_filename}.csv'
            summary_path = f'results_dump/summary_dump/{args.dataset_name}/{orig_model}/{prefix}_summary_no_inst_{output_filename}.csv'
        elif args.dataset_name == 'paradetox':
            input_path = f'splits/{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/{args.dataset_name}/{orig_model}/{prefix}_answer_no_inst_{output_filename}.csv'
            summary_path = f'results_dump/summary_dump/{args.dataset_name}/{orig_model}/{prefix}_summary_no_inst_{output_filename}.csv'
            
        print("input_path", input_path)
        print("output_path", output_path)
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            ['judge', 'info'], #, 'mc'
            input_path, 
            output_path, 
            summary_path, 
            device="cuda", 
            interventions=interventions, 
            intervention_fn=lt_modulated_vector_add, 
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_special_direction=args.use_special_direction,
            instruction_prompt=True,
            max_examples=args.max_examples,
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    if args.mode != 'get_top_heads':
        results = np.array(results)
        final = results.mean(axis=0)

    # print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
