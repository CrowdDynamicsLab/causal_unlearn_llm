import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import numpy as np

from utils import LossCollection

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.ident = torch.nn.Identity()

    def forward(self, x):
        return self.ident(x)

class Linear_Nonlinearity(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear_Nonlinearity, self).__init__()
        self.A = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.A(x)

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim = 16, hidden_layers=1, residual=True):
        super(EmbeddingNet, self).__init__()
        self.residual = residual
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.FC_out = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.FC_input(x))
        for lin_layer in self.hidden_layers:
            if self.residual:
                x = self.LeakyReLU(lin_layer(x)) + x
            else:
                x = self.LeakyReLU(lin_layer(x))
        return self.FC_out(x)


class ParametricPart(nn.Module):
    def __init__(self, d, num_envs, train_scales=False):
        super(ParametricPart, self).__init__()
        self.d = d
        self.num_envs = num_envs

        # The intercept for the logistic regression environment i vs obs
        self.intercepts = torch.nn.Parameter(torch.ones(num_envs, requires_grad=True))
        # The linear coefficient
        self.shifts = torch.nn.Parameter(torch.zeros(num_envs, requires_grad=True))
        # The quadratic coefficient
        self.lambdas = torch.nn.Parameter(torch.ones(num_envs, requires_grad=True))

        # self.scales = torch.nn.Parameter(torch.ones((1, d), requires_grad=True))
        if train_scales:
            self.scales = torch.nn.Parameter(torch.ones((1, d), requires_grad=True))
        else:
            self.register_buffer('scales', torch.ones((1, d)))

    def forward(self, z, t, env_ids):
        z_sel = z[torch.arange(z.size(0)), t]
        intercepts_sel = self.intercepts[env_ids]
        lambdas_sel = self.lambdas[env_ids]
        shifts_sel = self.shifts[env_ids]
        logit = shifts_sel + z_sel * intercepts_sel - (z_sel * lambdas_sel) ** 2
        logit = torch.cat((torch.zeros((z.size(0), 1), device=logit.device), logit.view(-1, 1)), dim=1)
        return logit

class ContrastiveModel(torch.nn.Module):
    def __init__(self, d, num_envs, embedding, train_scales=False):
        super(ContrastiveModel, self).__init__()
        self.embedding = embedding
        self.d = d
        self.num_envs = num_envs
        self.parametric_part = ParametricPart(d, num_envs, train_scales)

    def get_z(self, x):
        return self.embedding(x)

    def forward(self, x, t, env_ids, return_embedding=False):
        z = self.embedding(x)
        logit = self.parametric_part(z, t, env_ids)
        if return_embedding:
            return logit, z
        return logit

def get_contrastive_synthetic(input_dim, latent_dim, hidden_dim, num_envs, hidden_layers=0, residual=True, train_scales=False, **kwargs):
    embedding = EmbeddingNet(input_dim, latent_dim, hidden_dim, hidden_layers=hidden_layers, residual=residual)
    return ContrastiveModel(latent_dim, num_envs, embedding, train_scales)

def get_contrastive_linear_synthetic(input_dim, latent_dim, num_envs, train_scales=False, **kwargs):
    embedding = Linear_Nonlinearity(input_dim=input_dim, output_dim=latent_dim)
    return ContrastiveModel(latent_dim, num_envs, embedding, train_scales)

def build_model_from_kwargs(model_kwargs):
    if model_kwargs["type"] == 'contrastive_linear':
        return get_contrastive_linear_synthetic(**model_kwargs)
    elif model_kwargs["type"] == 'contrastive':
        return get_contrastive_synthetic(**model_kwargs)
    # elif model_kwargs["type"] == 'oracle':
        # return get_oracle_synthetic(**model_kwargs)
    else:
        raise NotImplementedError('Must be in contrastive or oracle')

def train_model(data, params, verbose=True):
    data_params = params['data']
    model_params = params['model']
    train_params = params['train']
    device = train_params.get("device", 'cpu')

    dl_obs, dl_int = data.get_dataloaders(batch_size=train_params['batch_size'], train=True)
    dl_obs_val, dl_int_val = data.get_dataloaders(batch_size=train_params['batch_size'], train=False)

    model = build_model_from_kwargs(params['model'])
    model = model.to(device)
    
    best_model = copy.deepcopy(model)

    mse = torch.nn.MSELoss()
    # mse = torch.nn.HuberLoss(delta=1., reduction='sum')
    ce = torch.nn.CrossEntropyLoss()
    loss_tracker = LossCollection()
    val_loss = np.inf

    epochs = train_params.get("epochs", 10)
    eta = train_params.get('eta', 0.0)
    lr_nonparametric = train_params.get('lr_nonparametric', .1)
    lr_parametric = train_params.get('lr_parametric', lr_nonparametric)
    optimizer_name = train_params.get("optimizer", "sgd").lower()

    non_parametric_params = list(model.embedding.parameters())

    if optimizer_name == 'sgd':
        optim = torch.optim.SGD([
                {'params': model.parametric_part.parameters(), 'lr': lr_parametric},
                {'params': non_parametric_params, 'lr': lr_nonparametric}
            ], weight_decay=train_params.get('weight_decay', 0.0))
    elif optimizer_name == 'adam':
        optim = torch.optim.Adam([
                {'params': model.parametric_part.parameters(), 'lr': lr_parametric},
                {'params': non_parametric_params, 'lr': lr_nonparametric}
            ], weight_decay=train_params.get('weight_decay', 0.0))
    else:
        raise NotImplementedError("Only Adam and SGD supported at the moment")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)

    train_loss_history = []
    val_loss_history = []
    # r2_history = []

    print("epochs", epochs)

    for i in tqdm(range(epochs)):
        model.train()
        for step, data in enumerate(zip(dl_obs, dl_int)):
            x_obs, x_int, t_int, env_ids_int = data[0][0], data[1][0], data[1][1], data[1][2]
            x_obs, x_int, t_int, env_ids_int = x_obs.to(device), x_int.to(device), t_int.to(device), env_ids_int.to(device)
            
            logits_int = model(x_int, t_int, env_ids_int)
            logits_obs, embedding = model(x_obs, t_int, env_ids_int, True)

            classifier_loss = ce(logits_obs, torch.zeros(x_obs.size(0), dtype=torch.long, device=device)) + \
                              ce(logits_int, torch.ones(x_int.size(0), dtype=torch.long, device=device))
            accuracy = (torch.sum(torch.argmax(logits_obs, dim=1) == 0) + torch.sum(
                        torch.argmax(logits_int, dim=1) == 1)) / (2 * x_int.size(0))

            loss = classifier_loss + eta * torch.sum(torch.abs(model.parametric_part.lambdas))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_tracker.add_loss(
                {'CE-loss': classifier_loss.item(), 'accuracy': accuracy.item()}, x_obs.size(0))
        scheduler.step()

        if verbose:
            print("Finished epoch {}".format(i + 1))
            loss_tracker.print_mean_loss()
        train_loss_history.append(loss_tracker.get_mean_loss()['CE-loss'])
        loss_tracker.reset()

        for step, data in enumerate(zip(dl_obs_val, dl_int_val)):
            x_obs, x_int, t_int, env_ids_int = data[0][0], data[1][0], data[1][1], data[1][2]
            x_obs, x_int, t_int, env_ids_int = x_obs.to(device), x_int.to(device), t_int.to(device), env_ids_int.to(device)
            
            logits_int = model(x_int, t_int, env_ids_int)
            logits_obs, embedding = model(x_obs, t_int, env_ids_int, True)
            # method_specific_loss = eta * torch.sum(torch.mean(embedding, dim=0) ** 2)

            classifier_loss = ce(logits_obs, torch.zeros(x_obs.size(0), dtype=torch.long, device=device)) + \
                                  ce(logits_int, torch.ones(x_int.size(0), dtype=torch.long, device=device))
            accuracy = (torch.sum(torch.argmax(logits_obs, dim=1) == 0) + torch.sum(
                        torch.argmax(logits_int, dim=1) == 1)) / (2 * x_int.size(0))
            loss_tracker.add_loss(
                {'CE-loss': classifier_loss.item(), 
                 'accuracy': accuracy.item()}, x_obs.size(0))
        ce_loss = loss_tracker.get_mean_loss()['CE-loss']
        
        if ce_loss < val_loss:
            val_loss = ce_loss
            best_model = copy.deepcopy(model)
            
        if i % 10 == 0 and verbose:
            print("Printing test and validation loss")
            loss_tracker.print_mean_loss()
        val_loss_history.append(ce_loss)

        loss_tracker.reset()
        # if z_gt is not None:
        #     z_gt_tensor = torch.tensor(z_gt, device=device, dtype=torch.float)
        #     z_pred = model.get_z(torch.tensor(x_val, device=device, dtype=torch.float))
        #     r2_history.append(torch.mean(get_R2_values(z_gt_tensor, z_pred)).item())
        # else:
        #     r2_history.append(0)
    # return best_model, model, val_loss, [train_loss_history, val_loss_history, r2_history]
    return best_model, model, val_loss, [train_loss_history, val_loss_history]