# Get activation from the base model -> Selection top heads -> Finetune these heads -> Get activation from finetuned model -> Intervene on these activations -> Compute toxicity score and perplexity score
import subprocess

# Get activations
def get_activations():
    datasets = ['toxigen_vicuna', 'hate_vicuna']
    models = ['vicuna_13B', 'llama3_8B', 'llama_7B']
    for dataset in datasets:
        for model in models:
            # launch subprocess to get activations
            subprocess.run(['python', 'get_activations.py', '--dataset', dataset, '--model', model], 
                          cwd='causal_to_concept/llms/get_activations')


def get_top_heads():
    selection_methods = ['pns','accuracy']
    datasets = ['toxigen_vicuna', 'hate_vicuna']
    models = ['vicuna_13B', 'llama3_8B', 'llama_7B']
    mode = 'get_top_heads'
    num_heads = [18, 36, 72]
    alpha = [0]
    for dataset in datasets:
        for model in models:
            for method in selection_methods: 
                for num_head in num_heads:
                    for alpha in alpha:
                        #python validate_2fold_toxic.py  vicuna_13b --num_heads 10 --alpha 5 --seed 2 --device 0 --num_fold 2 --use_special_direction
                        subprocess.run(['python', 'validate_2fold_toxic.py', '--dataset', dataset, model, '--num_heads', num_head, '--alpha', alpha, '--seed', '0',
                                         '--device', '0', '--num_fold', '2', '--mode', mode, '--use_special_direction', f"{'--use_pns' if method == 'pns' else ''}"], 
                                    cwd='causal_to_concept/llms')

def finetune_heads():
    datasets = ['toxigen_vicuna', 'hate_vicuna']
    models = ['vicuna_13b', 'llama3-8b', 'llama-7b']
    num_heads = [18, 36, 72]
    for dataset in datasets:
        for model in models:
            # launch subprocess to finetune heads
            subprocess.run(['python', 'finetune_test.py', '--dataset', dataset, '--model', model, '--num_heads', num_heads], 
                          cwd='causal_to_concept/llms')

def steering():
    datasets = ['toxigen_vicuna', 'hate_vicuna']
    models = ['vicuna_13b', 'llama3-8b', 'llama-7b']
    num_heads = [18, 36, 72]
    alpha = [0,5,10]
    selection_methods = ['pns','accuracy']
    for dataset in datasets:
        for model in models:
            for num_head in num_heads:
                for alpha in alpha:
                    for method in selection_methods:
                        # launch subprocess to steering
                        subprocess.run(['python', 'validate_2fold_toxic.py', '--dataset', dataset, model, '--num_heads', num_head, '--alpha', alpha, '--seed', '0',
                                                    '--device', '0', '--num_fold', '2', '--use_special_direction', f"{'--use_pns' if method == 'pns' else ''}"], 
                                                cwd='causal_to_concept/llms')
                        
def perplexity():
    datasets = ['toxigen_vicuna', 'hate_vicuna']
    models = ['vicuna_13b', 'llama3-8b', 'llama-7b']
    num_heads = [18, 36, 72]
    alpha = [0,5,10]
    selection_methods = ['pns','accuracy']
    for dataset in datasets:
        for model in models:
            for num_head in num_heads:
                for alpha in alpha:
                    for method in selection_methods:
                        # launch subprocess to perplexity
                        subprocess.run(['python', 'get_perplexity.py', '--dataset', dataset, model, '--num_heads', num_head, '--alpha', alpha, '--seed', '0',
                                                    '--device', '0', '--num_fold', '2', '--use_special_direction', f"{'--use_pns' if method == 'pns' else ''}"], 
                                                cwd='causal_to_concept/llms')



def main():
    """
    1. get activations from base model (vicuna or llama) from 2 datasets (toxigen_vicuna and hate_vicuna)
        a. hate_vicuna: 
            vicuna_13B
            llama_7B
        b. toxigen_vicuna: 
            vicuna_13B

    2. get top heads from base model using pns (True) or accuracy (False)
    3. finetune heads
        a. num_heads - 18, 36, 72
        b. cov - 0, 0.001, 0.01, 0.1, 1
        c. use_pns - True, False
    4. steering (intervene on top heads)
        a. get activation again
        b. alpha - 0, 5, 10, 15
        c. use_pns - True, False
        d. different finetuned models
    5. perplexity
    6. toxicity score
    """
    get_activations()
    get_top_heads()
    finetune_heads()

if __name__ == "__main__":
    main()