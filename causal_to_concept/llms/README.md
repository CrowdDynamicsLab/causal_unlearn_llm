This is the code to run the large language model alignment experiments in the paper ```From Causal to Concept-Based Representation Learning```

# A quick guide

- Follow the instructions to setup the honest-llama [repo](https://github.com/likenneth/honest_llama) and install any additional requirements as per our ```requirements.txt```

- Use their instructions to download the dataset and get the LLaMA activations for ```tqa_mc2```

- The main code changes are in ```utils.py``` and ```validation/validate2fold.py```, the modified versions of which are available here

- Call our modified technique using

python validate_2fold.py llama_7B --num_heads 72 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py llama_7B --num_heads 72 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py vicuna_13B --num_heads 18 --alpha 5 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_13B --num_heads 18 --alpha 5 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --model_name llama_3B --num_heads 72 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --model_name vicuna_13B --num_heads 72 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py  --model_name vicuna_pns --num_heads 10 --alpha 5 --seed 0 --device 0 --num_fold 2 --use_special_direction

## 72 heads, alpha = 15, toxigen_vicuna, vicuna_13B
# get top heads
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name vicuna_13B --num_heads 18 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name vicuna_13B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

---------------------
# finetune align alpha with get top heads
# naming: change name in path, create name in get_activations/utils_toxic/valid2fold_toxic
python finetune_test.py --model_name vicuna_13B --dataset_name toxigen_vicuna --head_select logpns --num_heads 36 --alpha 15.0 --use_pns

python finetune_test.py --model_name vicuna_13B --dataset_name toxigen_vicuna --head_select accuracy --num_heads 36 --alpha 15.0

---------------------
# intervene get activation
python get_activations.py --model_name model_name --dataset_name toxigen_vicuna

---------------------
# intervene
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name vicuna_13B_toxigen_vicuna_36_0.0001_acc --num_heads 36 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name vicuna_13B_toxigen_vicuna_36_0.0001_pns --num_heads 36 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns

# no intervention baseline
python validate_2fold_toxic.py --model_name vicuna_13B --num_heads 10 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction
  
=============================================================================================
## 72 heads, alpha = 15, toxigen_vicuna, llama3_8B
# get activations
python get_activations.py --model_name llama3_8B --dataset_name toxigen_vicuna

# get top heads
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

---------------------
# finetune align alpha with get top heads
# naming: change name in path, create name in get_activations/utils_toxic/valid2fold_toxic
python finetune_test.py --model_name llama3_8B --dataset_name toxigen_vicuna --head_select logpns --num_heads 36 --alpha 15.0 --use_pns

python finetune_test.py --model_name llama3_8B --dataset_name toxigen_vicuna --head_select accuracy --num_heads 36 --alpha 15.0

---------------------
# intervene get activation
python get_activations.py --model_name model_name --dataset_name toxigen_vicuna

---------------------
# intervene
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns

# no intervention baseline
python validate_2fold_toxic.py --model_name llama3_8B --num_heads 10 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns
  
=============================================================================================
## 72 heads, alpha = 15, hate_vicuna, llama3_8B
# get activations
python get_activations.py --model_name llama3_8B --dataset_name hate_vicuna

# get top heads
python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

---------------------
# finetune align alpha with get top heads
# naming: change name in path, create name in get_activations/utils_toxic/valid2fold_toxic
python finetune_test.py --model_name llama3_8B --dataset_name hate_vicuna --head_select logpns --num_heads 36 --alpha 15.0 --use_pns

python finetune_test.py --model_name llama3_8B --dataset_name hate_vicuna --head_select accuracy --num_heads 36 --alpha 15.0

---------------------
# intervene get activation
python get_activations.py --model_name model_name --dataset_name hate_vicuna

---------------------
# intervene
python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name llama3_8B --num_heads 18 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name llama3_8B --num_heads 18 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns

# no intervention baseline
python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name llama3_8B --num_heads 10 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns
  

  


=============================================================================================
## 72 heads, alpha = 15, hate_vicuna, vicuna_13B
# get activations
python get_activations.py --model_name vicuna_13B --dataset_name hate_vicuna

# get top heads
python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_13B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_13B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

---------------------
# finetune align alpha with get top heads
# naming: change name in path, create name in get_activations/utils_toxic/valid2fold_toxic
python finetune_test.py --model_name vicuna_13B --dataset_name hate_vicuna --head_select logpns --num_heads 72 --alpha 15.0 --use_pns

python finetune_test.py --model_name vicuna_13B --dataset_name hate_vicuna --head_select accuracy --num_heads 72 --alpha 15.0

---------------------
# intervene get activation
python get_activations.py --model_name model_name --dataset_name hate_vicuna

---------------------
# intervene
python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_13B --num_heads 18 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_13B --num_heads 18 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns

# no intervention baseline
python validate_2fold_toxic.py --dataset_name hate_vicuna --model_name vicuna_13B --num_heads 10 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns
  


08/24/2025
=====================================================================
## 36 heads, alpha = 15, toxigen_vicuna, llama3_8B
# get top heads
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction --use_pns

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 72 --alpha 15 --seed 2 --mode get_top_heads --device 0 --num_fold 2 --use_special_direction

---------------------
# finetune align alpha with get top heads
# naming: change name in path, create name in get_activations/utils_toxic/valid2fold_toxic
python finetune_test.py --model_name llama3_8B --dataset_name toxigen_vicuna --head_select logpns --num_heads 36 --alpha 15.0 --use_pns --epochs 5 --lambda_term2 1e-3 --lr 1e-5 --use_l2 --lambda_fm 0.001

python finetune_test.py --model_name vicuna_13B --dataset_name toxigen_vicuna --head_select accuracy --num_heads 36 --alpha 15.0

python finetune_new.py --model_name llama3_8B --dataset_name toxigen_vicuna --head_select logpns --num_heads 36 --alpha 15.0 --use_pns --epochs 5 --lambda_term2 1e-3 --lr 1e-5 --use_kl
---------------------
# intervene get activation
python get_activations.py --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_False_0.05_epoch5 --dataset_name toxigen_vicuna

python get_activations.py --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_useKL_True_0.05_epoch5 --dataset_name toxigen_vicuna


---------------------
# intervene
python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 36 --alpha 5.0 --seed 2 --device 0 --num_fold 2 --use_special_direction --use_pns --max_examples 300

python validate_2fold_toxic.py --dataset_name paradetox --model_name llama3_8B --num_heads 36 --alpha 5.0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_center_of_mass --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_0.0001_finetuned_epoch5 --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_epoch5 --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5 --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_mat_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_l2_finetuned_epoch5 --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_epoch5 --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_context.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B \
    --num_heads 36 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction \
    --use_pns --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_epoch5 \
    --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_l20.0001_useKL_True_0.001_epoch5 \
    --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300

python validate_2fold_toxic.py --dataset_name toxigen_vicuna \
    --model_name llama3_8B_toxigen_vicuna_logpns_36_True_1e-05_0.001_finetuned_useKL_True_0.05_epoch5 \
    --num_heads 36 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction --max_examples 300


# no intervention baseline
python validate_2fold_toxic.py --model_name llama3_8B --num_heads 10 --alpha 0 --seed 2 --device 0 --num_fold 2 --use_special_direction

# local intervention
python build_local_store.py \
  --dataset_name toxigen_vicuna \
  --model_name llama3_8B \
  --model_path /path/to/your/model \
  --output_dir ./local_store

  
# evaluate
python read_perplexity.py 
python read_single.py


# 0913
---------------------
# intervene
python validate_2fold_context.py \
  --model_name llama3_8B \
  --dataset_name toxigen_vicuna \
  --use_local_interventions \
  --local_store_dir /work/hdd/bcxt/yian3/toxic/local_store/llama3_8B_toxigen_vicuna_pns \
  --local_k 512 \
  --lam 0.25 \
  --local_tau 2.0 \
  --max_examples 300 \
  --num_heads 36 --alpha 5.0 --seed 2 --device 0 --num_fold 2 \
  --use_special_direction --use_pns --task continuation