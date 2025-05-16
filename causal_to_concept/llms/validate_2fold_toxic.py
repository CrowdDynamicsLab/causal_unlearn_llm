import torch
from einops import rearrange
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, GPT2LMHeadModel
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
    'honest_llama_7B': 'results_dump/llama_7B_seed_42_top_48_heads_alpha_15', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'honest_alpaca_7B': 'results_dump/alpaca_7B_seed_42_top_48_heads_alpha_15', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'honest_vicuna_7B': 'results_dump/vicuna_7B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'honest_llama2_chat_7B': 'results_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'honest_llama2_chat_13B': 'results_dump/llama2_chat_13B_seed_42_top_48_heads_alpha_15', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'honest_llama2_chat_70B': 'results_dump/llama2_chat_70B_seed_42_top_48_heads_alpha_15', 
    'vicuna_13B': 'lmsys/vicuna-13b-v1.5',
    'vicuna_pns': '/work/hdd/bcxt/yian3/models/vicuna_pns_finetuned',
    'COV_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_finetuned_epoch5',
    'COV_pns_use_pns': '/work/hdd/bcxt/yian3/toxic/models/vicuna_13B_toxigen_vicuna_logpns_18_True_finetuned_epoch5',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--dataset_name', type=str, default='toxigen_vicuna', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='toxigen_vicuna', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--heads_path', type=str, 
                      default="./features/False_vicuna_13B_toxigen_vicuna_seed_2_top_36_heads_alpha_5.0_fold_0_top_heads.npy",
                      help='Path to selected heads numpy file')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--use_special_direction', action='store_true', default=False)
    parser.add_argument('--use_pns', action='store_true', default=False)
    parser.add_argument('--use_mat_direction', action='store_true', default=False)
    args = parser.parse_args()
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
    elif args.dataset_name == "hate_vicuna" or args.dataset_name == "toxigen_vicuna":
        df = pd.read_csv(f'./TruthfulQA/{args.dataset_name}.csv')

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
    model_name = HF_NAMES["honest_" + args.model_name if args.use_honest else args.model_name]
    MODEL = model_name if not args.model_dir else args.model_dir

    if "gpt2" in args.model_name:
        model = GPT2LMHeadModel.from_pretrained(
            MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "/projects/bdeb/chenyuen0103/toxic/models/vicuna_13b_toxigen_vicuna_logpns_finetuned_epoch5_lr0.0001_bs128_lambda0.01",
            torch_dtype=torch.float16,
            local_files_only=True
        )

    

    # define number of layers and heads
    try:
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
    except:
        num_layers = 12  # default for tiny_gpt2
        num_heads = 12    
    
    # load activations 
    if args.dataset_name == "toxigen":

        head_wise_activations = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_head_wise.npy")[:6]
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        labels = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_labels.npy")[:6]
        with open(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_categories.pkl", "rb") as f:
            categories = pickle.load(f)  # List of target groups, 1 per sentence
        # tuning dataset: no labels used, just to get std of activations along the direction
        activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
        tuning_activations = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{activations_dataset}_head_wise.npy")[:6]
        tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{activations_dataset}_labels.npy")[:6]

    elif args.dataset_name == "hate":
        head_wise_activations = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{args.dataset_name}_head_wise.npy")
        # head_wise_activations = head_wise_activations[indices]
        # np.save(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{args.dataset_name}_head_wise.npy", head_wise_activations)
        labels = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{args.dataset_name}_labels.npy")
        # labels = labels[indices]
        # np.save(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{args.dataset_name}_labels.npy", labels)
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        with open(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_categories.pkl", "rb") as f:
            categories = pickle.load(f)  # List of target groups, 1 per sentence


        # tuning dataset: no labels used, just to get std of activations along the direction
        activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
        tuning_activations = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{activations_dataset}_head_wise.npy")
        # tuning_activations = tuning_activations[indices]
        # np.save(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{activations_dataset}_head_wise.npy", tuning_activations)
        tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{activations_dataset}_labels.npy")
        # tuning_labels = tuning_labels[indices]
        # np.save(f"/projects/bdeb/chenyuen0103/toxic/features/shuffled_{args.model_name}_{activations_dataset}_labels.npy", tuning_labels)

    elif args.dataset_name == "hate_vicuna" or args.dataset_name == "toxigen_vicuna":
        head_wise_activations = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_head_wise.npy") # [:200]
        head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
        labels = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_labels.npy") # [:200]
        
        activations_dataset = args.dataset_name if args.activations_dataset is None else args.activations_dataset
        tuning_activations = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{activations_dataset}_head_wise.npy") # [:200]
        tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
        tuning_labels = np.load(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{activations_dataset}_labels.npy") # [:200]
        
    n,l,h,d = head_wise_activations.shape
    input_dim = l * h * d
    c_path = f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{args.dataset_name}_c_all.pt"
    if os.path.exists(c_path):
        print(f"Loading cached c_all from {c_path}")
        head_wise_c = torch.load(c_path)
    else:
        print(f"No cached c_all found. Training VAE and saving to {c_path}")
        _, _, head_wise_c = train_vae_and_extract_mu(head_wise_activations, labels, input_dim, z_dim=32, h_dim1=128, h_dim2=64,
                                batch_size=128, lr=1e-3, vae_epochs=10, dataset_name=args.dataset_name, model_name=args.model_name, device='cuda')
        print("head_wise_c size", head_wise_c.size(), args.model_name)
    # separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, categories[:100], args.dataset_name)
    separated_head_wise_activations, separated_labels, separated_head_wise_c, idxs_to_split_at = get_activations(labels, head_wise_activations, head_wise_c, args.dataset_name, args.model_name)
    
    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"splits/{args.model_name}_{args.dataset_name}_fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"splits/{args.model_name}_{args.dataset_name}_fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_idxs].to_csv(f"splits/{args.model_name}_{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        elif args.use_special_direction:
            direct_path = f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{activations_dataset}_special_direction.npy"
            if os.path.exists(direct_path):
                print(f"Loading direct from {direct_path}")
                com_directions = np.load(direct_path)
            else:
                print(f"No cached direct found. Saving to {direct_path}")
                com_directions = get_special_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, df)
                np.save(f"/projects/bdeb/chenyuen0103/toxic/features/{args.model_name}_{activations_dataset}_special_direction.npy", com_directions)
        elif args.use_mat_direction:
            com_directions = get_matrix_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        else:
            com_directions = None

        # breakpoint()
        print("Finished computing com_directions of shape", com_directions.shape)

        if model == 'none': # model == 'COV_pns' or model == 'vicuna_pns':
            selected_heads = np.load(args.heads_path)
            top_heads = selected_heads[:args.num_heads] if len(selected_heads) > args.num_heads else selected_heads

        else:
            if args.use_pns:
                top_heads, pns_scores = get_top_heads_pns(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels,
                            separated_head_wise_c, num_layers, num_heads, num_to_intervene=args.num_heads, lambda_reg=1e-4, sigma_sq=1.0, seed=42, use_random_dir=args.use_random_dir)
                np.save(f'./features/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_pns_scores.npy', pns_scores)
                probes = None
            else:
                top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
        np.save(f'./features/{args.use_pns}_{args.model_name}_{args.dataset_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}_top_heads.npy', top_heads)
                
        print("Heads intervened: ", sorted(top_heads))
        interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, args.use_mat_direction, args.use_special_direction, com_directions)
        print("Finished computing interventions dict")

        def lt_modulated_vector_add(_head_output, layer_name, start_edit_location='lt', prompt_encoding=None):
            # if torch.isnan(_head_output).any():
            #     print(f"[WARNING] NaNs in {layer_name} head_output!")
            #     print(f"[FATAL] Invalid head_output in {layer_name}!")
            #     print("Max:", _head_output.max(), "Min:", _head_output.min())
            #     return torch.zeros_like(_head_output)
            head_output = _head_output.detach().type(torch.float32)
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            if "gpt2" in args.model_name:
                layer = int(layer_name.split('.')[1])  # e.g., 'transform.h.3'
            else:
                layer = int(layer_name.split('.')[2])

            # print("head output shape", head_output.shape)
            if prompt_encoding is not None: # use_special_direction
                assert prompt_encoding.shape == (384,)
                prompt_encoding = torch.FloatTensor(prompt_encoding).to(head_output.device.index).reshape(-1, 384)
            # print("Prompt encoding", prompt_encoding.shape)
            
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

        filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}'

        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'
        if args.use_honest:
            filename = 'honest_' + filename
        if args.use_special_direction:
            filename += '_special'
        if args.use_mat_direction:
            filename += '_mat'
                    

        prefix = "eval_pns_" if args.use_pns else "eval_"
        if args.dataset_name == 'hate':
            input_path = f'splits/shuffled_{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/shuffled_{args.dataset_name}_{filename}.csv'
            summary_path = f'results_dump/summary_dump/shuffled_{args.dataset_name}_{filename}.csv'
        elif args.dataset_name == 'toxigen':
            input_path = f'splits/{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/{args.dataset_name}_{filename}.csv'
            summary_path = f'results_dump/summary_dump/{args.dataset_name}_{filename}.csv'
        elif args.dataset_name == 'toxigen_vicuna' or args.dataset_name == 'hate_vicuna':
            input_path = f'splits/{args.model_name}_{args.dataset_name}_fold_{i}_test_seed_{args.seed}.csv'
            output_path = f'results_dump/answer_dump/{prefix}_answer_{args.dataset_name}_{filename}.csv'
            summary_path = f'results_dump/summary_dump/{prefix}_summary_{args.dataset_name}_{filename}.csv'
            
        print("input_path", input_path)
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
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    # print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
