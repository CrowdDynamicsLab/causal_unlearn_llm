import torch
import numpy as np
from metrics import compute_mccs
from sklearn import linear_model

def get_results(model, data, device):
    results = dict()
    samples_for_evaluation = 5000
    model.eval()
    model.to(device)
    x = torch.tensor(data.obs_f, dtype=torch.float, device=device)[:samples_for_evaluation]
    z_pred = model.get_z(x).cpu().detach().numpy()
    z_gt = data.obs[:samples_for_evaluation]
    z_gt = z_gt @ data.concepts.T
    # x_gt = data.obs_f.reshape(data.obs.shape[0], -1)[:samples_for_evaluation]

    regressor = linear_model.LinearRegression().fit(z_pred, z_gt)
    results['R2_Z'] = regressor.score(z_pred, z_gt)

    if not np.isnan(z_pred).any():
        mccs = compute_mccs(z_gt, z_pred)
        for k in mccs:
            results[k] = mccs[k]
    else:
        print('Model predicted NANs, skipping mccs evaluation')

    # print("Learned parametric model parameters\n")
    # print("intercepts", model.parametric_part.intercepts)
    # print("shifts", model.parametric_part.shifts)
    # print("lambdas", model.parametric_part.lambdas)

    model.to(device)
    model.train()
    return results

def evaluate(models_and_losses, data, params):
    train_params = params['train']
    device = train_params.get("device", 'cpu')
    model, _, _, _ = models_and_losses
    results = get_results(model, data, device)
    
    print("Printing results\n", results)