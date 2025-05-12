#optimization.py
GNN_PATH = './GNN/'
import os
if not os.path.exists(GNN_PATH):
    os.makedirs(GNN_PATH)

# %%
# Install packages
import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torch-geometric", "scikit-learn", "pandas", "shapely", "seaborn", "pygmtools", "numpy", "moviepy<2.0.0", "matplotlib", "tensorboard"])

# Check if pygmtools is installed
try:
    import pygmtools
except ImportError:#pygmtools library
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Thinklab-SJTU/pygmtools.git"])

# Check pytorch version and make sure you use a GPU Kernel
import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)

# Check nvcc version
try:
    subprocess.run(["nvcc", "--version"], check=True)
except FileNotFoundError:
    print("nvcc is not installed or not in PATH.")

# Check GPU
try:
    subprocess.run(["nvidia-smi"], check=True)
except FileNotFoundError:
    print("nvidia-smi is not installed or not in PATH.")

#set device as cuda if available to load model and data on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# ─── Standard library ──────────────────────────────────────────────────────────
import copy
import os
import sys
import pickle
import random
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

# ─── Third-party libraries ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.affinity import translate
from shapely.geometry import Polygon
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GCNConv

from moviepy.editor import ImageSequenceClip
import optuna
import json

# ─── Local application/library imports ────────────────────────────────────────
import pygmtools
pygmtools.BACKEND = 'pytorch'

destination_dir = os.path.join('AFAT')

# Ensure the destination directory is in sys.path
if destination_dir not in sys.path:
    sys.path.append(destination_dir)

# AFA-U inlier predictor and Top-K matching from AFAT
from k_pred_net import Encoder as AFAUEncoder
from sinkhorn_topk import soft_topk

# %%
# Set Seed for reproducibility
seed = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed once at the beginning
set_seed(seed)

# For reproducible DataLoader shuffle
g = torch.Generator()
g.manual_seed(seed)

#----------------------------------------
#            PARAMETERS OPTIMIZATION
#----------------------------------------

def objective_gm(trial, train_dataset, val_dataset, path):
    # iperparametri da esplorare
    lr           = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout      = trial.suggest_uniform("dropout", 0.0, 0.6)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    out_dim      = trial.suggest_categorical("out_dim",    [16, 32, 64])
    batch_size   = trial.suggest_categorical("batch_size", [2, 4, 8])
    heads        = trial.suggest_int("heads",           1,   4)
    attn_dropout = trial.suggest_uniform("attn_dropout", 0.0, 0.6)
    num_layers   = trial.suggest_int("num_layers",       1,   3)
    sinkhorn_tau = trial.suggest_loguniform("tau",      1e-3, 1e-1)
    max_iter     = trial.suggest_int("max_iter",        10, 100)
    in_dim = train_dataset[0][0].x.size(1)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

    # modello
    class MatchingModel(nn.Module):
        def __init__(self,dropout, hidden_dim, out_dim, heads, attn_dropout, num_layers, sinkhorn_tau, max_iter):
            super().__init__()
            self.gnn = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(num_layers):
                # new: always average the heads so the feature‐dim stays dims[i+1]
                self.gnn.append(
                GATv2Conv(dims[i], dims[i+1],
                            heads=heads, concat=False,
                            dropout=attn_dropout))
            self.dropout = dropout
            # # bilinear weight matrix A per affinity
            # std = 1.0 / math.sqrt(out_dim)
            # self.A = nn.Parameter(torch.randn(out_dim, out_dim) * std)
            # self.temperature = temperature
            # InstanceNorm per-sample
            self.inst_norm = nn.InstanceNorm2d(1, affine=True)
            self.tau = sinkhorn_tau
            self.max_iter = max_iter

        def encode(self, x, edge_index):
            for i, conv in enumerate(self.gnn):
                x = conv(x, edge_index)
                if i < len(self.gnn)-1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def forward(self, batch1, batch2, perm_list, batch_idx1=None, batch_idx2=None, inference=False):
            device = next(self.parameters()).device
            x1, e1 = batch1.x.to(device), batch1.edge_index.to(device)
            x2, e2 = batch2.x.to(device), batch2.edge_index.to(device)
            perm_list = [p.to(device) for p in perm_list]
            
            if batch_idx1 is None:
                batch_idx1 = batch1.batch.to(device)
                batch_idx2 = batch2.batch.to(device)
            h1 = self.encode(x1, e1)
            h2 = self.encode(x2, e2)
            B = batch_idx1.max().item()+1
            loss = 0
            for b in range(B):
                h1_b = h1[batch_idx1==b]
                h2_b = h2[batch_idx2==b]

                # affinity matrix + normalization + sinkhorn
                sim = torch.matmul(h1_b, h2_b.T) # [n1, n2]
                sim_batched = sim.unsqueeze(0).unsqueeze(1) # [1,1,n1,n2]
                sim_normed = self.inst_norm(sim_batched).squeeze(1) # [1,n1,n2] -> [n1,n2]

                S = pygmtools.sinkhorn(sim_normed, tau=self.tau, max_iter=self.max_iter)[0]
                loss = loss + bce_permutation_loss(S, perm_list[b])
            return loss / B

    model = MatchingModel(
        dropout=dropout,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        heads=heads,
        attn_dropout=attn_dropout,
        num_layers=num_layers,
        sinkhorn_tau=sinkhorn_tau,
        max_iter=max_iter
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training rapido con early stopping
    best_val = float('inf')
    counter = 0
    for epoch in range(30):
        # train
        model.train()
        for b1, b2, perm in train_loader:
            optimizer.zero_grad()
            loss = model(b1, b2, perm)
            loss.backward()
            optimizer.step()
        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b1, b2, perm in val_loader:
                val_loss += model(b1, b2, perm).item()
        val_loss /= len(val_loader)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= 5:
                break
    
    # save best trial info
    if trial.number == 0 or best_val <= trial.study.best_value:
        result = {
            "val_loss": best_val,
            "params": trial.params
        }
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "best_trial_results.json"), "w") as f:
            json.dump(result, f, indent=2)

    return best_val

def objective_pgm(trial, train_dataset, val_dataset, path):
    lr           = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout_emb  = trial.suggest_uniform("dropout", 0.0, 0.6)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    out_dim      = trial.suggest_categorical("out_dim",    [16, 32, 64])
    batch_size   = trial.suggest_categorical("batch_size", [2, 4, 8])
    attn_dropout = trial.suggest_uniform("attn_dropout", 0.0, 0.6)
    max_iter     = trial.suggest_int("max_iter",        10, 100)
    tau          = trial.suggest_loguniform("tau",      1e-3, 1e-1)
    num_layers   = trial.suggest_int("num_layers",       1, 3)
    heads        = trial.suggest_int("heads",           1,   4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

    # Flexible model for partial matching
    class MatchingModel_GATv2SinkhornTopK_OPT(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, sinkhorn_max_iter, sinkhorn_tau,
                    attention_dropout, dropout_emb, num_layers, heads):
            super().__init__()
            self.gnn = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            for i in range(num_layers):
                # new: always average the heads so the feature‐dim stays dims[i+1]
                self.gnn.append(
                GATv2Conv(dims[i], dims[i+1],
                            heads=heads, concat=False,
                            dropout=attention_dropout))
            self.dropout = nn.Dropout(p=dropout_emb)
            # # bilinear weight matrix A per affinity
            # std = 1.0 / math.sqrt(out_dim)
            # self.A = nn.Parameter(torch.randn(out_dim, out_dim) * std)
            # self.temperature = temperature
            # InstanceNorm per-sample
            self.inst_norm = nn.InstanceNorm2d(1, affine=True)
            self.sinkhorn_max_iter = sinkhorn_max_iter
            self.sinkhorn_tau = sinkhorn_tau

        def encode(self, x, edge_index):
            for i, conv in enumerate(self.gnn):
                x = conv(x, edge_index)
                if i < len(self.gnn) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
            return x

        def forward(self, batch1, batch2, perm_list, batch_idx1=None, batch_idx2=None, inference=False):
            device = next(self.parameters()).device
            x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
            x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
            perm_list = [p.to(device) for p in perm_list]

            batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
            batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)

            h1 = self.encode(x1, edge1)
            h2 = self.encode(x2, edge2)

            B = batch_idx1.max().item() + 1
            loss = 0.0
            for b in range(B):
                h1_b = h1[batch_idx1 == b]
                h2_b = h2[batch_idx2 == b]

                # affinity matrix + normalization + sinkhorn
                sim = torch.matmul(h1_b, h2_b.T) # [n1, n2]
                sim_batched = sim.unsqueeze(0).unsqueeze(1) # [1,1,n1,n2]
                sim_normed = self.inst_norm(sim_batched).squeeze(1) # [1,n1,n2] -> [n1,n2]

                n1 = torch.tensor([h1_b.size(0)], dtype=torch.int32, device=device)
                n2 = torch.tensor([h2_b.size(0)], dtype=torch.int32, device=device)
                S = pygmtools.sinkhorn(sim_normed, n1=n1, n2=n2, max_iter=self.sinkhorn_max_iter, tau=self.sinkhorn_tau)

                ks_gt = torch.tensor([h2_b.size(0)], dtype=torch.long, device=device)

                _, soft_S = soft_topk(S, ks_gt, max_iter=self.sinkhorn_max_iter,
                                    tau=self.sinkhorn_tau, nrows=n1, ncols=n2,
                                    return_prob=True)

                loss += bce_permutation_loss(soft_S[0], perm_list[b])
            return loss / B

    model = MatchingModel_GATv2SinkhornTopK_OPT(
        in_dim=train_dataset[0][0].x.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        sinkhorn_max_iter=max_iter,
        sinkhorn_tau=tau,
        attention_dropout=attn_dropout,
        dropout_emb=dropout_emb,
        num_layers=num_layers,
        heads=heads
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float('inf')
    counter = 0
    for epoch in range(30):
        model.train()
        for b1, b2, perm in train_loader:
            optimizer.zero_grad()
            loss = model(b1, b2, perm)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b1, b2, perm in val_loader:
                val_loss += model(b1, b2, perm).item()
        val_loss /= len(val_loader)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= 5:
                break

    # save best trial info
    if trial.number == 0 or best_val <= trial.study.best_value:
        result = {
            "val_loss": best_val,
            "params": trial.params
        }
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "best_trial_results.json"), "w") as f:
            json.dump(result, f, indent=2)

    return best_val

### FUNCTIONS WITH BCE
def bce_permutation_loss(P, P_gt, eps: float = 1e-9):
    """Element-wise Binary Cross Entropy loss between prediction and ground truth."""
    return - (P_gt * torch.log(P + eps) + (1 - P_gt) * torch.log(1 - P + eps)).mean()

def deserialize_graph_matching_dataset(path: str, filename: str = "train_dataset.pkl") -> List[Tuple[Data, Data, torch.Tensor]]:
    """
    Deserialize a dataset of (Data1, Data2, PermutationMatrix) tuples from a file.
    """
    full_path = os.path.join(path, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    with open(full_path, 'rb') as f:
        pairs = pickle.load(f)

    print(f"Loaded {len(pairs)} pairs from {full_path}")
    return pairs

class GraphMatchingDataset(Dataset):
    def __init__(self, pairs):  # lista di (Data, Data, P)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # data1, data2, P
    

def collate_pyg_matching(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data1_list, data2_list, perm_list = zip(*batch)
    
    # Sposta ogni grafo sul device corretto
    data1_list = [d.to(device) for d in data1_list]
    data2_list = [d.to(device) for d in data2_list]
    
    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    
    return batch1, batch2, perm_list

print("Loading dataset...")

#load preprocessed dataset
gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "room_dropout_noise")
models_path = os.path.join(GNN_PATH, 'models', "partial_graph_matching", "room_dropout_noise")

original_graphs = deserialize_graph_matching_dataset(
    gm_equal_preprocessed_path,
    "original.pkl"
)
noise_graphs = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "noise.pkl"
)

train_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "train_dataset.pkl"
)
val_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "valid_dataset.pkl"
)
test_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "test_dataset.pkl"
)

train_dataset = GraphMatchingDataset(train_list)
val_dataset = GraphMatchingDataset(val_list)
test_dataset = GraphMatchingDataset(test_list)

print("Starting hyperparameter optimization...")

#param opt 
study = optuna.create_study(direction="minimize")
study.optimize(lambda trial: objective_pgm(trial, train_dataset, val_dataset, models_path), n_trials=30)
print("Best hyperparameters: ", study.best_params)
print("Best trial: ", study.best_trial)
print("Best value: ", study.best_value)
# Save the study
with open(os.path.join(models_path, "study.pkl"), "wb") as f:
    pickle.dump(study, f)
# # Load the study
# with open(os.path.join(models_path, "study.pkl"), "rb") as f:
#     study = pickle.load(f)
# Plot the study
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(os.path.join(models_path, "opt_history.png"))