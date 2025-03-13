import yaml
import argparse
import random
from data_generation import DataGen
from models import train_model
import numpy as np
import torch
from evaluation import evaluate

def run_expt(seed, settings):
    with open(settings,'r') as f:
        params = yaml.safe_load(f)

    if seed > 0:
        # Manual seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    data = DataGen(data_params = params['data'])
    params['model']['input_dim'] = params['data']['dim_x']
    params['model']['latent_dim'] = params['data']['envs']['num_concepts']
    params['model']['num_envs'] = params['data']['envs']['n_envs']
    models_and_losses = train_model(data, params)
    evaluate(models_and_losses, data, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111, help='seed')
    parser.add_argument('--settings', type=str, required=True, help='filename for settings')
    args = parser.parse_args()
    run_expt(args.seed, args.settings)