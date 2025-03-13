import yaml
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import TensorDataset as TensorDataset
import matplotlib.pyplot as plt
from models import Identity, Linear_Nonlinearity, EmbeddingNet
from tqdm import tqdm

class DataGen:        
    def __init__(self, data_params, train_fraction=.8, seed=0, **kwargs):
        self.dim_z = data_params['dim_z']
        self.dim_x = data_params['dim_x']
        self.f_type = data_params['f_type']
        self.base_dist = data_params['base_dist']
        self.envs = data_params['envs']
        self.concept_means = self.envs['means']
        self.concept_var = self.envs['var']

        if self.f_type == "identity":
            self.f = Identity()
        elif self.f_type == "linear":
            self.f = Linear_Nonlinearity(self.dim_z, self.dim_x)
        elif self.f_type == "nonlinear":
            self.f = EmbeddingNet(self.dim_z, self.dim_x)
        else:
            assert False, "Unknown function type"

        # concept vectors
        self.concepts = np.random.uniform(-.3, .3, (self.envs['num_concepts'], self.dim_z))
        print("Concept vectors", self.concepts)

        # base distribution
        self.num_samples_per_batch = 1000
        self.cur_z_generator = self.sample_from_base_distribution(self.dim_z, self.base_dist, self.num_samples_per_batch)

        self.num_train_samps = int(train_fraction * data_params['num_samps'])
        self.num_val_samps = data_params['num_samps'] - self.num_train_samps

        print("Generating train data")
        self.obs, self.obs_f, self.int, self.int_f, self.targets, self.env_ids = self.sample(self.num_train_samps)
        print("Generating test data")
        self.obs_val, self.obs_f_val, self.int_val, self.int_f_val, self.targets_val, self.env_ids_val = self.sample(self.num_val_samps)

    def sample_from_base_distribution(self, dim_z, base_dist_params, num_samples_per_batch):
        if base_dist_params["type"] == "gmm":
            # GMM with n_comps distributions, random parameters
            # TODO: scipy.stats.rv_continuous might be more elegant
            n_comps = base_dist_params['n_comps']
            means = np.random.uniform(-1, 1, (n_comps, dim_z))
            print("GMM distribution with means", means)
            # print("og prod", np.einsum('ij,kj->ki', means, self.concepts))
            self.base_distribution_true_means = means
            cov = np.array([np.diag(np.random.uniform(0.01, 0.015, size=dim_z)) for i in range(n_comps)])

            weights = np.random.uniform(0.3, 1, n_comps)
            weights = weights/np.sum(weights)

            while True:
                index = np.random.choice(np.arange(n_comps), p=weights, size=num_samples_per_batch)
                z = np.fromiter((np.random.multivariate_normal(means[i], cov[i]) for i in index), 
                                dtype=np.ndarray,
                                count=num_samples_per_batch)
                z = np.stack(z)
                yield z
        else:
            assert False, "Unknown base distribution"

    def sample(self, num_samps):
        obs_data, obs_data_mapped, int_data, int_data_mapped = None, None, [], []

        print("Generating base env")
        z = []
        num_obs_samples = self.envs['n_envs'] * num_samps
        with tqdm(total = num_obs_samples) as pbar:
            while (len(z) < num_obs_samples):
                z_batch = next(self.cur_z_generator)
                z_batch = z_batch[: num_obs_samples - len(z)]
                z = z_batch if len(z) == 0 else np.vstack((z, z_batch))
                pbar.update(len(z_batch))
        
        x = self.f(torch.tensor(z, dtype=torch.float)).detach().numpy()

        obs_data = z
        obs_data_mapped = x

        target = []
        env_ids = []
        for i in range(self.envs['n_envs']):
            idx = self.envs['signs'][i]
            print("Generating env", i, "with target concept index", idx)

            # rejection sampling
            z = []
            with tqdm(total = num_samps) as pbar:
                while len(z) < num_samps:
                    # Sample z from base dist, then keep it if N(<a, z>; \mu, var) / upper_bound < U(0, 1)
                    z_batch = next(self.cur_z_generator)
                    upper_bound = 1. / (np.sqrt(2 * np.pi * self.envs['var'])) + 1e-5

                    cur_mean = self.envs['means'][i]
                    cur_var = self.envs['var']
                    dot_prod = z_batch @ self.concepts[idx].reshape(-1, 1)
                    cur_pdf = np.fromiter(norm.pdf(dot_prod, cur_mean, np.sqrt(cur_var)), dtype=np.float64, count=self.num_samples_per_batch)

                    threshold = np.random.uniform(0, 1, size = self.num_samples_per_batch)
                    z_batch = z_batch[cur_pdf / upper_bound > threshold]
                    z_batch = z_batch[: num_samps - len(z)]
                    z = z_batch if len(z) == 0 else np.vstack((z, z_batch))
                    pbar.update(len(z_batch))

            x = self.f(torch.tensor(z, dtype=torch.float)).detach().numpy()

            int_data.append(z)
            int_data_mapped.append(x)
            target.append(idx * np.ones(num_samps, dtype=np.int_))
            env_ids.append(i * np.ones(num_samps, dtype=np.int_))
 
        int_data = np.concatenate(int_data, axis=0)
        int_data_mapped = np.concatenate(int_data_mapped, axis=0)
        target = np.concatenate(target, axis=0)
        env_ids = np.concatenate(env_ids, axis=0)

        return obs_data, obs_data_mapped, int_data, int_data_mapped, target, env_ids
    
    def get_dataloaders(self, batch_size, train=True):
        obs_f = self.obs_f if train else self.obs_f_val
        int_f = self.int_f if train else self.int_f_val
        targets = self.targets if train else self.targets_val
        env_ids = self.env_ids if train else self.env_ids_val
        obs_dataloader = DataLoader(TensorDataset(torch.tensor(obs_f, dtype=torch.float)), shuffle=True,
                                    batch_size=batch_size)
        int_dataloader = DataLoader(TensorDataset(torch.tensor(int_f, dtype=torch.float), 
                                                  torch.tensor(targets),
                                                  torch.tensor(env_ids)),
                                    shuffle=True, batch_size=batch_size)
        return obs_dataloader, int_dataloader

if __name__ == "__main__":
    print("Data generation test\n")

    with open(f'settings.yaml','r') as f:
        params = yaml.safe_load(f)
    params['data']['num_samps'] = 100
    datagen = DataGen(data_params=params['data'])
    base_latents = datagen.obs
    base_dataset = datagen.obs_f
    latents = datagen.int
    dataset = datagen.int_f
    target = datagen.targets
    print("base latent shape", base_latents.shape) # (train_frac * n_samps, dim_z)
    print("base dataset shape", base_dataset.shape) # (train_frac * n_samps, dim_x
    print("latent shape", latents.shape) # (n_envs, train_frac * n_samps, dim_z)
    print("dataset shape", dataset.shape) # (n_envs, train_frac * n_samps, dim_x)
    
    print("\nEmpirical Means and variances")
    for i in range(datagen.envs['n_envs']):
        latent_i = latents[i * datagen.num_train_samps: (i + 1) * datagen.num_train_samps]
        assert np.sum(datagen.env_ids == i) == latent_i.shape[0]
        print(f"Latents corresponding to env {i} has shape {latent_i.shape}")
        dot_prod = latent_i @ datagen.concepts[datagen.envs['signs'][i]][:, None]
        print(f"True mean: {datagen.envs['means'][i]: .3f}, generated mean: {dot_prod.mean(): .3f}")
        print(f"True var: {datagen.envs['var']: .3f}, generated var: {dot_prod.var(): .3f}")

    # If base distribution variance is high, this picture doesn't clearly look like a GMM
    # However, if base distribution variance is low, then rejection sampling takes a long time
    print("\nGenerated base distribution stats")
    z = []
    dir = np.random.uniform(0.01, 10., size=datagen.dim_z)
    print("Projecting to random direction", dir)
    print("Projected means", datagen.base_distribution_true_means @ dir)
    print("Plotting histogram of projected base distribution")
    z = base_latents @ dir[:, None]
    plt.hist(z.squeeze(), bins=50)
    plt.savefig("base_dist.png")
    print("Saved histogram of base distribution")