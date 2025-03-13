import torch
import numpy as np

from sklearn.cross_decomposition import CCA

import matplotlib.pyplot as plt

from mcc import mean_corr_coef, mean_corr_coef_out_of_sample

def get_R2_values(x_obs, pred_obs):
    pred_obs = pred_obs - torch.mean(pred_obs, dim=0, keepdim=True)
    x_obs = x_obs - torch.mean(x_obs, dim=0, keepdim=True)
    scales = torch.sum(x_obs * pred_obs, dim=0, keepdim=True) / torch.sum(pred_obs * pred_obs, dim=0, keepdim=True)
    return 1 - torch.mean((x_obs - pred_obs * scales) ** 2, dim=0) / torch.mean(x_obs ** 2, dim=0)

def compute_mccs(x, y):
    cutoff = len(x) // 2
    ii, iinot = np.arange(cutoff), np.arange(cutoff, 2 * cutoff)
    mcc_s_in = mean_corr_coef(x=x[ii], y=y[ii])
    mcc_s_out = mean_corr_coef_out_of_sample(x=x[ii], y=y[ii], x_test=x[iinot], y_test=y[iinot])
    return {"mcc_s_in": mcc_s_in, "mcc_s_out": mcc_s_out}


def print_cor_coef(x_obs, pred_obs):
    pred_obs = pred_obs - torch.mean(pred_obs, dim=0, keepdim=True)
    x_obs = x_obs - torch.mean(x_obs, dim=0, keepdim=True)
    pred_obs_copy = pred_obs.detach().clone()
    x_obs_copy = x_obs.detach().clone()
    pred_obs_copy = pred_obs_copy.unsqueeze(1)
    x_obs_copy = x_obs_copy.unsqueeze(2)
    cors = torch.mean(pred_obs_copy * x_obs_copy, dim=0)
    var_x = torch.std(x_obs_copy, dim=0)
    var_pred = torch.std(pred_obs_copy, dim=0)
    print((1 / var_pred).view(-1, 1) * cors * (1/var_x).view(1, -1))