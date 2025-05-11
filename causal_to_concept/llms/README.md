This is the code to run the large language model alignment experiments in the paper ```From Causal to Concept-Based Representation Learning```

# A quick guide

- Follow the instructions to setup the honest-llama [repo](https://github.com/likenneth/honest_llama) and install any additional requirements as per our ```requirements.txt```

- Use their instructions to download the dataset and get the LLaMA activations for ```tqa_mc2```

- The main code changes are in ```utils.py``` and ```validation/validate2fold.py```, the modified versions of which are available here

- Call our modified technique using

python validate_2fold.py llama_7B --num_heads 72 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold.py llama_7B --num_heads 72 --alpha 0 --seed 0 --device 0 --num_fold 2 --use_special_direction


python validate_2fold_toxic.py llama_7B --num_heads 72 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py vicuna_13B --num_heads 18 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction

python validate_2fold_toxic.py --model_name llama_3B --num_heads 72 --alpha 15 --seed 2 --device 0 --num_fold 2 --use_special_direction

accelerate launch validate_2fold_toxic.py llama3_8B --num_heads 18 --alpha 0 --seed 0 --device 0 --num_fold 2 --use_special_direction