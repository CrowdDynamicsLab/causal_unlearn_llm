import importlib
import copy
import io, time
# from io import BytesIO
import os
import collections
import argparse
from itertools import combinations, cycle, product
import math
import numpy as np
import pandas as pd
import pickle
import tarfile
import random
import re
import requests
from nltk.corpus import stopwords
from scipy.sparse import hstack, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
import torch.nn.functional as F
from vae import *

from collections import Counter, defaultdict
import sys

import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

import torch
from torchvision import datasets, transforms
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.autograd import Variable
from transformers import GPT2Tokenizer, GPT2LMHeadModel


from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import *

def compute_prob(logits, mode="logistic"):
    if mode == "linear":
        probs = torch.max(torch.stack([logits,torch.zeros_like(logits)],dim=2),dim=2)[0]
        probs = torch.min(torch.stack([probs,torch.ones_like(probs)],dim=2),dim=2)[0]
    elif mode == "logistic":
        probs = nn.Sigmoid()(logits)
    return probs

def mean_nll(probs, y, mode="logistic"):
    if mode == "linear":
        mean_nll = nn.MSELoss()(probs, y)
    elif mode == "logistic":
        mean_nll = nn.BCELoss()(probs, y)
    return mean_nll

def mean_accuracy(probs, y):
    preds = (probs > 0.5).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def mean_rep_by_label(representations, labels):
    unique_labels = np.unique(labels)
    means = {label: representations[labels == label].mean(axis=0)[:10] for label in unique_labels}
    return means

class CausalRepDecoder(nn.Module):
    def __init__(self, causal_dim, gpt2_model_name='gpt2'):
        super().__init__()
        self.causal_dim = causal_dim
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Project causal representation to GPT2 hidden size
        self.projector = nn.Linear(causal_dim, self.gpt2.config.n_embd)

    def forward(self, causal_rep, max_length=50):
        """
        causal_rep: Tensor of shape [batch_size, causal_dim]
        """
        batch_size = causal_rep.size(0)
        prefix_embedding = self.projector(causal_rep).unsqueeze(1)  # [B, 1, hidden_dim]

        # Generate dummy token inputs to start generation
        input_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id).to(causal_rep.device)

        # Forward with prefix hidden state manually injected
        outputs = self.gpt2.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=None
        )
        return outputs

    def decode_output(self, token_ids):
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.input_dim = args.input_dim
#         self.z_dim = args.z_dim
#         self.num_features = args.num_features
#         lin1 = nn.Linear(self.input_dim, self.num_features)
#         lin4 = nn.Linear(self.z_dim, 1)
#         for lin in [lin1, lin4]:
#             nn.init.xavier_uniform_(lin.weight)
#             nn.init.zeros_(lin.bias)
#         self._main = nn.Sequential(lin1)
#         self._tvaez = nn.Sequential(lin4) 
#         self.finallayer = nn.Linear(self.num_features + 1, 1)
#     def forward(self, inputbow, vaez):
#         features = torch.matmul(inputbow, F.softmax(self._main[0].weight,dim=1).T)
#         logits = self.finallayer(torch.cat([features, self._tvaez(vaez)],dim=1))
#         probs = compute_prob(logits, mode=args.mode)
#         features_ctr = features - features.mean(dim=0)
#         beta_hat = 0.
#         feature_hats = 0.
#         logit_hats = logits
#         prob_hats = probs
#         return features, logits, probs, beta_hat, logit_hats, prob_hats
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.num_features = args.num_features
        lin1 = nn.Linear(args.input_dim, args.hidden_dim)
        lin2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        lin3 = nn.Linear(args.hidden_dim, self.num_features)
        lin4 = nn.Linear(args.z_dim, args.hidden_dim)
        lin5 = nn.Linear(args.hidden_dim, 1)
        for i,lin in enumerate([lin1, lin2, lin3, lin4, lin5]):
            nn.init.xavier_uniform_(lin.weight, 1.)
            nn.init.zeros_(lin.bias)
            #print("CAUSAL REP layer", i, lin.weight.abs().mean())
            # initialization to be too larg values will create optimization problems
            while lin.weight.abs().mean() > 0.1:
                nn.init.xavier_uniform_(lin.weight, 1.)
                print("layer", i, lin.weight.abs().mean())
        self._main = nn.Sequential(lin1, nn.ReLU(True), 
                                   lin2, nn.ReLU(True), 
                                   lin3, nn.ReLU(True), 
                                   nn.BatchNorm1d(args.num_features, affine=False))
        self._tvaez = nn.Sequential(lin4, nn.ReLU(True), lin5, nn.ReLU(True))
        self.finallayer = nn.Linear(args.num_features+1, 1)
    def forward(self, input, vaez):
        features = self._main(input)
        
        logits = self.finallayer(torch.cat([features, self._tvaez(vaez)],dim=1))
        probs = compute_prob(logits, mode=args.mode)
        features_ctr = features - features.mean(dim=0)
        beta_hat = 0.
        feature_hats = 0.
        logit_hats = logits
        prob_hats = probs
        return features, logits, probs, beta_hat, logit_hats, prob_hats


# the Net component is not used
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(args.num_features, 1)
    def forward(self, x):
        x = self.fc(x)
        return x

def initNet(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)

def evaluate(model, dataloader):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.cuda(), yb.cuda()
            _, _, probs = model(xb)
            acc = mean_accuracy(probs, yb)
            total_acc += acc.item() * xb.size(0)
    return total_acc / len(dataloader.dataset)

def train(model, dataloader, optimizer, args):
    model.train()
    out_features = []
    for xb_text, xb_pcaz, xb_vaez, yb in dataloader:
        xb_text, xb_pcaz, xb_vaez, yb = xb_text.cuda(), xb_pcaz.cuda(), xb_vaez.cuda(), yb.cuda()

        features, logits, probs, beta_hat, logit_hats, prob_hats = model(xb_text, xb_vaez)
        out_features.append(features.detach().cpu())
        nll = mean_nll(probs, yb, mode=args.mode) 
        nll_hat = mean_nll(prob_hats, yb, mode=args.mode) 
        acc = mean_accuracy(probs, yb)
        acc_hat = mean_accuracy(prob_hats, yb)
        
        y = yb - yb.mean()
        y = y.view(-1, 1) if y.dim() == 1 else y
        # Feature + latent
        X = torch.cat([features, xb_vaez], dim=1)
        X = X - X.mean(dim=0)
        X = torch.cat([torch.ones(X.shape[0], 1).cuda(), X], dim=1)
        beta = [torch.matmul(
            torch.matmul(
                torch.inverse(args.l2_reg * torch.eye(X.shape[1]).cuda() + X.T @ X), X.T), y[:, j]) 
                for j in range(y.shape[1])]

        covs = cov(torch.cat([beta[0][1:args.num_features+1] * features, 
                              torch.unsqueeze((beta[0][-args.z_dim:] * xb_vaez).sum(dim=1),1)], dim=1))[-1][:-1] # extract the last row to have cov(Features, C)
        causalrep = ((features.std(dim=0) * beta[0][1:args.num_features + 1]) ** 2).sum()

        weight_norm = torch.tensor(0.).cuda()
        for w in model.finallayer.parameters():
            weight_norm += w.norm().pow(2)

        l2penalty = args.l2_reg * weight_norm

        l1_penalty = F.softmax(model._main[0].weight,dim=1).abs().sum()
        train_causalrep_loss = -causalrep.clone() # + 1e-3 * l1_penalty - 1e-2 * torch.log(1 - train_featureZr2)

        optimizer.zero_grad()
        train_causalrep_loss.backward(retain_graph=True)
        optimizer.step()
    out_features = torch.cat(out_features, dim=0)
    return out_features
        
def evaluate(model, dataloader, args):
    model.eval()
    total_acc = 0
    total_nll = 0
    total = 0
    with torch.no_grad():
        for xb_text, xb_pcaz, xb_vaez, yb in dataloader:
            xb_text, xb_pcaz, xb_vaez, yb = xb_text.cuda(), xb_pcaz.cuda(), xb_vaez.cuda(), yb.cuda()
            features, logits, probs, _, _, _ = model(xb_text, xb_vaez)
            total_acc += mean_accuracy(probs, yb).item() * len(yb)
            total_nll += mean_nll(probs, yb, mode=args.mode).item() * len(yb)
            total += len(yb)
    return total_acc / total, total_nll / total

def causalrep_diagnostics(model, envs, args):
    model.eval()
    with torch.no_grad():
        train_features, train_y = model(envs[0][args.mode_train_data], envs[0][args.mode_latent])[0].cpu().numpy(), envs[0]['labels'].cpu().numpy()
        testct_features, testct_y = model(envs[1][args.mode_train_data], envs[1][args.mode_latent])[0].cpu().numpy(), envs[1]['labels'].cpu().numpy()
        testobs_features, testobs_y = model(envs[2][args.mode_train_data], envs[2][args.mode_latent])[0].cpu().numpy(), envs[2]['labels'].cpu().numpy()

    C_vals = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    for C in C_vals:
        print('\ncausal-pred-w-features', 'C', C)
        clf = LogisticRegression(C=C, class_weight='balanced', solver='lbfgs')
        clf.fit(train_features, train_y)

        resulttrain = classification_report((train_y > 0), (clf.predict(train_features) > 0), output_dict=True)
        resultct = classification_report((testct_y > 0), (clf.predict(testct_features) > 0), output_dict=True)
        resultobs = classification_report((testobs_y > 0), (clf.predict(testobs_features) > 0), output_dict=True)

        print('train', resulttrain['accuracy'])
        print('testct', resultct['accuracy'])
        print('testobs', resultobs['accuracy'])

    # print("\n\n##### causal rep top words")
    # feature_weights = torch.topk(F.softmax(model._main[0].weight, dim=1), 20, axis=1)
    # top_causal_words = feature_weights[1].cpu().numpy()
    # top_causal_weights = feature_weights[0].cpu().detach().numpy()

    # beta_feat = model._main[0].weight.detach().cpu().numpy()
    # for j in np.argsort(-np.abs(beta_feat[0][:args.num_features])):
    #     print("feature", j)
    #     print("coefficient", beta_feat[0][j])
    #     sort_causal_words = np.argsort(-top_causal_weights[j])[:20]
    #     print("top causal words", [id2term[i] for i in top_causal_words[j][sort_causal_words]], top_causal_weights[j][sort_causal_words])
    #     print("top causal words", top_causal_weights[j][sort_causal_words])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_1B', help='model name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--dataset', type=str, default='toxigen', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='toxigen', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--use_special_direction', action='store_true', default=False)
    parser.add_argument('--use_mat_direction', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--l2_reg', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--split_ratio', type=float, default=0.6)
    parser.add_argument('--method', type=str, default="baseline", choices=["baseline", "causalrep"])
    parser.add_argument('--mode', type=str, default="linear", choices=["linear", "logistic"])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_features', type=int, default=384)
    parser.add_argument('--input_dim', type=int, default=0)
    parser.add_argument('--vae_input_dim', type=int, default=384)
    parser.add_argument('--vae_epochs', type=int, default=51)
    parser.add_argument('--spurious_corr', type=float, default=0.9)
    parser.add_argument('--alter_freq', type=int, default=50)
    parser.add_argument('--mode_latent', type=str, default="pcaz", choices=["vaez", "bertz", "bertz_cl", "pcaz"])
    parser.add_argument('--mode_train_data', type=str, default="text", choices=["text", "bertz"])
    args = parser.parse_args()
    
    randseed = int(time.time()*1e7%1e8)
    print("random seed: ", randseed)
    print("device: ", args.device)
    sys.stdout.flush()
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    res = pd.DataFrame(vars(args), index=[0])
    res['randseed'] = randseed
    
    sys.stdout.flush()

    moniker = args.dataset

    out_dir = moniker + '_out'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dat_file = '../../causal_to_concept/llms/TruthfulQA/' + moniker + '.csv'


    df = pd.read_csv(dat_file).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    texts = df['text'].tolist()
    labels = df['label'].astype(float).values

    split1 = int(args.split_ratio * len(texts))
    split2 = int(split1 + 0.2 * len(texts))
    train_text, train_label = texts[:split1], labels[:split1]
    testobs_text, testobs_label = texts[split1:split2], labels[split1:split2]
    testct_text, testct_label = texts[split2:], labels[split2:]

    # st_model = SentenceTransformer('all-MiniLM-L6-v2')
    # train_emb = np.array(st_model.encode(train_text, show_progress_bar=True))
    # testobs_emb = np.array(st_model.encode(testobs_text, show_progress_bar=True))
    # testct_emb = np.array(st_model.encode(testct_text, show_progress_bar=True))
    
    # train_emb = torch.tensor(train_emb, dtype=torch.float32)
    # testobs_emb = torch.tensor(testobs_emb, dtype=torch.float32)
    # testct_emb = torch.tensor(testct_emb, dtype=torch.float32)

    # train_label = torch.tensor(train_label, dtype=torch.float32)
    # testobs_label = torch.tensor(testobs_label, dtype=torch.float32)
    # testct_label = torch.tensor(testct_label, dtype=torch.float32)

    # train_dataset = TensorDataset(train_emb, train_label)
    # testobs_dataset = TensorDataset(testobs_emb, testobs_label)
    # testct_dataset = TensorDataset(testct_emb, testct_label)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # testobs_loader = DataLoader(testobs_dataset, batch_size=args.batch_size, shuffle=False)
    # testct_loader = DataLoader(testct_dataset, batch_size=args.batch_size, shuffle=False)
    
    # top_feature_idx, placebo_feature_idx, coef = get_top_terms(train_emb, train_label, coef_thresh=0.0, placebo_thresh=0.1) # use coef_threshold=0.0 to take all features, no thresholding happening here.
    # fea_corrcoef = np.corrcoef(train_emb[:,top_feature_idx].T) - np.eye(train_emb[:,top_feature_idx].shape[1])
    # colinear_fea = np.where(fea_corrcoef>0.96)[0]
    # feature_idx = np.array(list(set(top_feature_idx) - set(colinear_fea)))

    # X_train, train_label = train_emb[:,feature_idx], train_label
    # X_testobs, testobs_label = testobs_emb[:,feature_idx], testobs_label
    # X_testct, testct_label = testct_emb[:,feature_idx], testct_label

    # vocabsize = X_train.shape[1]
    # args.input_dim = vocabsize
    
    # # calculate pca embedding
    # pca = PCA(n_components=args.z_dim)
    # # pca.fit(np.row_stack([X_train_np, X_testobs_np, X_testct_np]))
    # pca.fit(np.row_stack([train_emb[:,feature_idx]]))
    # train_pca_embedding = torch.from_numpy(pca.transform(train_emb[:,feature_idx])).float().cuda()
    # testobs_pca_embedding = torch.from_numpy(pca.transform(testobs_emb[:,feature_idx])).float().cuda()
    # testct_pca_embedding = torch.from_numpy(pca.transform(testct_emb[:,feature_idx])).float().cuda()


    # envs = [
    #     {'text': train_emb.cuda(), 'pcaz': train_pca_embedding, 'labels': train_label.cuda()}, \
    #     {'text': testct_emb.cuda(), 'pcaz': testct_pca_embedding, 'labels': testct_label.cuda()}, \
    #     {'text': testobs_emb.cuda(), 'pcaz': testobs_pca_embedding, 'labels': testobs_label.cuda()}]
    
    if args.method == 'baseline':
        vec = TfidfVectorizer(min_df=10, binary=True, max_df=0.8, ngram_range=(1,3))

        X_full = vec.fit_transform(train_text)
        X_train_full = vec.transform(train_text)
        X_testobs_full = vec.transform(testobs_text)
        X_testct_full = vec.transform(testct_text)

        feats = np.array(vec.get_feature_names_out())

        top_feature_idx, placebo_feature_idx, coef = get_top_terms(vec.transform(train_text), train_label, coef_thresh=0.0, placebo_thresh=0.1) 

        X_train_np = vec.transform(train_text).toarray()
        X_testobs_np = vec.transform(testobs_text).toarray()
        X_testct_np = vec.transform(testct_text).toarray()

        fea_corrcoef = np.corrcoef(X_train_np[:,top_feature_idx].T) - np.eye(X_train_np[:,top_feature_idx].shape[1])
        colinear_fea = np.where(fea_corrcoef>0.96)[0]
        feature_idx = np.array(list(set(top_feature_idx) - set(colinear_fea)))
        print(len(feature_idx), X_train_np.shape)

        id2term = collections.OrderedDict({i:v for i,v in enumerate(feats[feature_idx])})
        term2id = collections.OrderedDict({v:i for i,v in enumerate(feats[feature_idx])})


        spurious_words = np.array([term2id['as'], term2id['also'], term2id['am'], term2id['an']])


        final_train_accs = []
        final_test_accs = []
        final_train_baselineaccs = []
        final_test_baselineaccs = []
        final_train_baselinevaeaccs = []
        final_test_baselinevaeaccs = []


        def make_environment(texts, labels, e):
            def torch_bernoulli(p, size):
                return (torch.rand(size) < p)
            def torch_xor(a, b):
                a = torch.tensor(a)
                b = torch.tensor(b)
                return torch.abs(a.float() - b.float()) # Assumes both inputs are either 0 or 1
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (torch.tensor(labels) == 1).float()
            labels = torch_xor(labels, torch_bernoulli(0.35, len(labels)))
            # Assign a color based on the label; flip the color with probability e
            spurious_counts = torch.stack([torch_xor(labels, torch_bernoulli(e, len(labels))) for i in range(len(spurious_words))], axis=1)
            # Apply the color to the image by zeroing out the other color channel
            texts[:,spurious_words] = spurious_counts.cpu().numpy()

            return {
                'texts': torch.from_numpy(texts).float(),
                'labels': labels[:, None],
                'colors': spurious_counts
            }

        train_data = make_environment(X_train_np[:,feature_idx], train_label, 0.9)
        X_train, train_label = train_data['texts'], train_data['labels']

        testobs_data = make_environment(X_testobs_np[:,feature_idx], testobs_label, 0.9)
        X_testobs, testobs_label = testobs_data['texts'], testobs_data['labels'] 

        testct_data = make_environment(X_testct_np[:,feature_idx], testct_label, 0.9)
        X_testct, testct_label = testct_data['texts'], testct_data['labels']

        envs = [
            {'text': X_train, 'labels': train_label}, \
            {'text': X_testct, 'labels': testct_label}, \
            {'text': X_testobs, 'labels': testobs_label}]

        for C in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
            alpha = 1./C
            print('\nnaive-pred', 'C', C)

            clf = LogisticRegression(C=C, class_weight='balanced', solver='lbfgs')
            clf.fit(envs[0][args.mode_train_data].cpu().detach().numpy(), train_label.cpu().detach().numpy())
            resulttrain = classification_report((train_label.cpu().detach().numpy() > 0), (clf.predict(envs[0][args.mode_train_data].cpu().detach().numpy()) > 0), output_dict=True)
            resultct = classification_report((testct_label.cpu().detach().numpy() > 0), (clf.predict(envs[1][args.mode_train_data].cpu().detach().numpy()) > 0), output_dict=True)
            resultobs = classification_report((testobs_label.cpu().detach().numpy() > 0), (clf.predict(envs[2][args.mode_train_data].cpu().detach().numpy())> 0), output_dict=True)
            print('train',resulttrain['accuracy'])
            print('testobs',resultobs['accuracy'])
            print('testct',resultct['accuracy'])
            sys.stdout.flush()

            naive_weights = clf.coef_
            top_naive_words = np.argsort(-np.abs(naive_weights))[0,:20]
            top_coef = naive_weights[0,top_naive_words]

            print("top naive words", [id2term[i] for i in top_naive_words], top_coef)


    elif args.method == 'causalrep':
        # get VAE cause
        vae = VAE(x_dim=args.vae_input_dim, h_dim1=args.hidden_dim, h_dim2=args.hidden_dim, z_dim=args.z_dim)
        vae = vae.to(args.device)
        optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr, weight_decay=1e-8)
        for epoch in range(args.vae_epochs):
            train_vae(vae, train_loader, optimizer_vae, epoch)
            test_vae(vae, testobs_loader)

        train_vae_recon, train_vae_mu, train_vae_logvar = vae(envs[0]['text'].view(-1, args.vae_input_dim))
        testct_vae_recon, testct_vae_mu, testct_vae_logvar = vae(envs[1]['text'].view(-1, args.vae_input_dim))
        testobs_vae_recon, testobs_vae_mu, testobs_vae_logvar = vae(envs[2]['text'].view(-1, args.vae_input_dim))
        print("shapes", train_vae_recon.shape, train_vae_mu.shape)
        mean_reps = mean_rep_by_label(train_vae_recon, train_label)
        print("vaez by labels", mean_reps)
        mean_reps_mu = mean_rep_by_label(train_vae_mu, train_label)
        print("vae mu by labels", mean_reps_mu)
        
        envs[0]['vaez'] = train_vae_mu.detach()
        envs[1]['vaez'] = testct_vae_mu.detach()
        envs[2]['vaez'] = testobs_vae_mu.detach()

        dataloaders = []

        for env in envs:
            # Ensure everything is a float tensor
            text = torch.tensor(env['text']).float()
            pcaz = torch.tensor(env['pcaz']).float()
            vaez = env['vaez'].float() if isinstance(env['vaez'], torch.Tensor) else torch.tensor(env['vaez']).float()
            labels = torch.tensor(env['labels']).float()

            dataset = TensorDataset(text, pcaz, vaez, labels)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle='train' in env['text'].shape)  # shuffle only for training
            dataloaders.append(dataloader)

        train_loader = dataloaders[0]
        testct_loader = dataloaders[1]
        testobs_loader = dataloaders[2]
        
        
        mlp = MLP().cuda()
        optimizer_causalrep = optim.Adam(mlp._main.parameters(), lr=args.lr, weight_decay=1e-8)
        for epoch in range(args.epochs):
            test_output = train(mlp, train_loader, optimizer_causalrep, args)

            if epoch % 10 == 0:
                train_acc, train_nll = evaluate(mlp, train_loader, args)
                val_acc, val_nll = evaluate(mlp, testct_loader, args)
                print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, NLL={train_nll:.4f} | Val Acc={val_acc:.4f}, NLL={val_nll:.4f}")

            if epoch % 20 == 0:
                causalrep_diagnostics(mlp, envs, args)
        # Final test on testobs
        test_acc, test_nll = evaluate(mlp, testobs_loader, args)
        print(f"\nFinal Test (OBS) Accuracy: {test_acc:.4f}, NLL: {test_nll:.4f}")

    model = CausalRepDecoder(causal_dim=384).cuda()
    output_ids = model(test_output.cuda(), max_length=30)
    texts = model.decode_output(output_ids)

    print(train_text[:10])
    for i, txt in enumerate(texts[:10]):
        print(f"Sample {i+1}: {txt}")

