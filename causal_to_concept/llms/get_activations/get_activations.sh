# python get_activations.py --model_name llama3_8B --dataset_name toxigen_vicuna
# python get_activations.py --model_name llama3_8B --dataset_name hate_vicuna

python get_activations.py --model_name vicuna_pns --dataset_name toxigen_vicuna
python get_activations.py --model_name vicuna_pns --dataset_name hate_vicuna

python get_activations.py --model_name llama3_8B --dataset_name toxigen_vicuna
python get_activations.py --model_name llama3_8B --dataset_name paradetox

python get_activations.py --model_name vicuna_13B_toxigen_vicuna_36_0.0001_pns --dataset_name toxigen_vicuna
python get_activations.py --model_name vicuna_13B_toxigen_vicuna_36_0.0001_acc --dataset_name toxigen_vicuna

python get_activations.py --model_name llama3_8B_toxigen_vicuna_36_0.0001_pns --dataset_name toxigen_vicuna
python get_activations.py --model_name llama3_8B_toxigen_vicuna_36_0.0001_acc --dataset_name toxigen_vicuna

python get_activations.py --model_name llama3_8B_toxigen_vicuna_18_0.0001_pns --dataset_name toxigen_vicuna
python get_activations.py --model_name llama3_8B_toxigen_vicuna_18_0.0001_acc --dataset_name toxigen_vicuna


python get_activations.py --model_name llama3_8B_hate_vicuna_36_0.0001_acc --dataset_name hate_vicuna
python get_activations.py --model_name llama3_8B_hate_vicuna_36_0.0001_pns --dataset_name hate_vicuna


python get_activations.py --model_name vicuna_13B_hate_vicuna_36_0.0001_acc --dataset_name hate_vicuna
python get_activations.py --model_name vicuna_13B_hate_vicuna_36_0.0001_pns --dataset_name hate_vicuna


python get_activations.py --model_name vicuna_13B_hate_vicuna_18_0.0001_acc --dataset_name hate_vicuna
python get_activations.py --model_name vicuna_13B_hate_vicuna_18_0.0001_pns --dataset_name hate_vicuna


python get_activations.py --model_name vicuna_13B_hate_vicuna_72_0.0001_acc --dataset_name hate_vicuna
python get_activations.py --model_name vicuna_13B_hate_vicuna_72_0.0001_pns --dataset_name hate_vicuna




