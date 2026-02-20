This is the code to run the large language model alignment experiments in the paper ```From Causal to Concept-Based Representation Learning```

# A quick guide

- Follow the instructions to setup the honest-llama [repo](https://github.com/likenneth/honest_llama) and install any additional requirements as per our ```requirements.txt```

- Use their instructions to download the dataset and get the LLaMA activations for ```tqa_mc2```

- The main code changes are in ```utils.py``` and ```validation/validate2fold.py```, the modified versions of which are available here
  
  


08/24/2025
=====================================================================
## get activation
python get_activations.py --model_name vicuna_13B --dataset_name toxigen_vicuna

python get_activations.py --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_useKL_True_0.05_epoch5 --dataset_name toxigen_vicuna


python get_activations.py --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 --dataset_name paradetox
                                  



# get top heads
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name vicuna_13B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_7B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name paradetox --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

---------------------
# finetune align alpha with get top heads
# naming: change name in path, create name in get_activations/utils_toxic/valid2fold_toxic
python finetune_test.py --model_name llama3_8B --dataset_name toxigen_vicuna --head_select logpns --num_heads 18 --use_pns --epochs 5 --lambda_term2 1e-3 --lr 1e-5 --use_l2 --use_kl --lambda_fm 0.001

python finetune_new.py --model_name llama3_8B --dataset_name toxigen_vicuna --head_select logpns --num_heads 36 --use_pns --epochs 5 --lambda_term2 1e-3 --lr 1e-5 --use_kl

python finetune_test.py --model_name vicuna_13B --dataset_name toxigen_vicuna --head_select accuracy --num_heads 36 

python finetune_test.py --model_name vicuna_7B --dataset_name toxigen_vicuna --head_select logpns --num_heads 18 --use_pns --epochs 5 --lambda_term2 1e-3 --lr 1e-5 --use_l2 --use_kl --lambda_fm 0.001 

---------------------
# intervene get activation
python get_activations.py --model_name vicuna_7B_toxigen_vicuna_logpns_18_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 --dataset_name toxigen_vicuna

python get_activations.py --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 --dataset_name paradetox


---------------------
# intervene
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name vicuna_13B --num_heads 36 --alpha 5.0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300 --use_pns 

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_7B --num_heads 1 --alpha 20.0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300 --use_pns 

python validate_2fold_toxic.py --dataset_name paradetox --model_name mistral_7B --num_heads 10 --alpha 2.0 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 18 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300 --use_pns 

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300


python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5 --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_epoch5 --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5 --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_mat_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5 --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_epoch5 --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_epoch5 \
    --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --num_heads 72 --alpha 10 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300 

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_72_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --num_heads 18 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_18_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --num_heads 18 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300



# no intervention baseline
python validate_2fold_toxic.py --model_name llama3_8B --num_heads 10 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction

# local intervention
# first get top heads:
python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --num_heads 72 --alpha 10 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300 --use_pns --mode get_top_heads

python build_local_store.py \
  --dataset_name toxigen_vicuna \
  --model_name llama3_8B \
  --model_path /path/to/your/model \
  --output_dir ./local_store

# consolidtae
python consolidate_diffs.py \
    --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --output_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen \
    --heads_path /work/hdd/bcxt/yian3/toxic/features/heads/True_llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_toxigen_vicuna_seed_2_top_72_heads_fold_0.npy


# one time build_prompt_encodings and neighbors
python build_prompt_encodings.py \
  --dataset_name toxigen_vicuna \
  --out_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen  \
  --st_model all-MiniLM-L6-v2 \
  --batch_embed 512

python build_topk_neighbors.py \
  --dataset toxigen_vicuna \
  --out_dir /work/hdd/bcxt/yian3/toxic/local_store_toxigen  \
  --K 512


  
# evaluate
python read_perplexity.py 
python read_single.py
read_eval.ipynb



# 0913
---------------------
# intervene
python validate_2fold_context.py \
  --model_name llama3_8B \
  --dataset_name toxigen_vicuna \
  --use_local_interventions \
  --local_store_dir /work/hdd/bcxt/yian3/toxic/local_store/llama3_8B_toxigen_vicuna_pns \
  --local_k 256 \
  --lam 0.25 \
  --local_tau 2.0 \
  --max_examples 300 \
  --num_heads 18 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --task continuation

python validate_2fold_context.py \
  --model_name llama3_8B \
  --dataset_name toxigen_vicuna \
  --use_local_interventions \
  --local_store_dir /work/hdd/bcxt/yian3/toxic/local_store/llama3_8B_toxigen_vicuna_pns \
  --local_k 256 \
  --lam 0.5 \
  --local_tau 2.0 \
  --max_examples 300 \
  --num_heads 18 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --task continuation

python validate_2fold_context.py \
  --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
  --dataset_name toxigen_vicuna \
  --use_local_interventions \
  --local_store_dir /work/hdd/bcxt/yian3/toxic/local_store/llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5_toxigen_vicuna_pns \
  --local_k 128 \
  --lam 0.25 \
  --local_tau 2.0 \
  --max_examples 300 \
  --num_heads 18 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --task continuation

 ## timing
 python validate_2fold_time.py \
  --model_name llama3_8B \
  --dataset_name toxigen_vicuna \
  --use_local_interventions \
  --local_store_dir /work/hdd/bcxt/yian3/toxic/local_store/llama3_8B_toxigen_vicuna_pns \
  --local_k 128 \
  --lam 0.25 \
  --local_tau 2.0 \
  --timing \
  --timing_compare_3ways \
  --timing_n_prompts 50 \
  --max_examples 50 \
  --num_heads 18 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --task continuation

# mask
python validate_2fold_mask.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 36 --alpha 0.0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300 --mask 36  --use_pns

# inerventions OOD
python validate_2fold_toxic_ood.py \
  --model_name llama3_8B \
  --dataset_name toxigen_vicuna \
  --dir_dataset_name hate_vicuna \
  --num_heads 36 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --max_examples 300

python validate_2fold_toxic_ood.py \
  --model_name llama3_8B \
  --dataset_name hate_vicuna \
  --dir_dataset_name toxigen_vicuna \
  --num_heads 36 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --max_examples 300

## directions
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 18 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_center_of_mass --max_examples 300 --use_pns 