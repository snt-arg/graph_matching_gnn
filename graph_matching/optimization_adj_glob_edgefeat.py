#optimization.py
GNN_PATH = './GNN/'
# Fallback root for the real validation scenes -- see pgm_training_adj_glob_edgefeat.py.
import os
# Fallback root for the real validation scenes. Repo-relative by default (override with
# GM_DATASET_ROOT); resolves to the same absolute path this was previously hardcoded to.
DATASET_DUMP_REAL_ROOT = os.path.join(
    os.environ.get("GM_DATASET_ROOT",
                   os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..", "datasets", "Graph-matching"))),
    "real")
import os
if not os.path.exists(GNN_PATH):
    os.makedirs(GNN_PATH)

# %%
# Install packages
import subprocess
import sys

# Install required packages
# Skipped when the environment already satisfies them (or GM_SKIP_PIP_INSTALL=1), so the
# script can run inside a prepared venv / offline box without `uv` reaching the network.
# On a fresh training machine nothing changes: the imports fail and the install still runs.
_REQUIRED_MODULES = ["torch", "torch_geometric", "sklearn", "pandas", "shapely", "seaborn",
                     "pygmtools", "matplotlib", "tensorboard", "optuna", "botorch", "wandb",
                     "plotly"]


def _environment_already_satisfied():
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in _REQUIRED_MODULES)


if os.environ.get("GM_SKIP_PIP_INSTALL") == "1" or _environment_already_satisfied():
    print("[setup] dependencies already present -> skipping `uv pip install`")
else:
    subprocess.check_call(["uv", "pip", "install", "torch", "torch-geometric", "scikit-learn", "pandas", "shapely", "seaborn", "pygmtools", "numpy<2", "moviepy<2.0.0", "matplotlib", "tensorboard", "optuna", "optuna-integration", "botorch", "wandb", "plotly", "kaleido"])
# Check if pygmtools is installed
try:
    import pygmtools
except ImportError:#pygmtools library
    subprocess.check_call(["uv", "pip", "install", "git+https://github.com/Thinklab-SJTU/pygmtools.git"])

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
from optuna.integration import BoTorchSampler
import wandb

from edge_features import (EDGE_ATTR_DIM, build_edge_index_and_attr, node_features,
                           normalize_edge_attr)


class EdgeAwareGATLayer(nn.Module):
    """
    One message-passing hop implementing arXiv:2409.11972 Eq. 4 (node update) + Eq. 5 (edge
    update) -- see pgm_training_adj_glob_edgefeat.py for the full rationale (identical here,
    duplicated per this codebase's existing convention of one self-contained script per stage).
    """
    def __init__(self, node_in_dim, node_out_dim, edge_in_dim, edge_out_dim, heads=1, attn_dropout=0.0):
        super().__init__()
        self.gath = GATv2Conv(node_in_dim, node_out_dim, heads=heads, concat=False,
                               edge_dim=edge_in_dim, dropout=attn_dropout, aggr='max')
        self.g_v = nn.Sequential(nn.Linear(node_in_dim + node_out_dim, node_out_dim), nn.ReLU())
        self.g_e = nn.Sequential(nn.Linear(node_in_dim + edge_in_dim + node_in_dim, edge_out_dim), nn.ReLU())

    def forward(self, x, edge_index, edge_attr, update_edge=True):
        agg = self.gath(x, edge_index, edge_attr=edge_attr)
        x_new = self.g_v(torch.cat([x, agg], dim=-1))
        if not update_edge:
            return x_new, None
        src, dst = edge_index[0], edge_index[1]  # dst == i (target/self), matches edge_features.py's swap
        edge_new = self.g_e(torch.cat([x[dst], edge_attr, x[src]], dim=-1))
        return x_new, edge_new
import json

# ─── Local application/library imports ────────────────────────────────────────
import pygmtools
pygmtools.BACKEND = 'pytorch'

# AFAT lives next to this file. Resolve it from the script's own location rather than the
# working directory: GNN_PATH ('./GNN/') is relative to the repo root, so the script is meant
# to run from graph_matching_gnn/ -- at which point a bare 'AFAT' would not resolve and the
# k_pred_net import dies before anything else happens.
destination_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AFAT')

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

# ─── CLI arguments ─────────────────────────────────────────────────────────
# One parameterized script for every experiment. `experiment` selects the
# dataset subfolder, the models/output subfolder AND the W&B group/name/tags,
# so two experiments run with a different value stay fully separated.
#   python optimization.py fully_glob_95
#   python optimization.py adj_no_glob_65 --wandb-mode offline
import argparse

parser = argparse.ArgumentParser(
    description="GP+BoTorch+LogNEI hyperparameter optimization for partial graph matching."
)
parser.add_argument(
    "experiment",
    help="Experiment id. Used as the dataset/models subfolder name and the "
         "W&B group/name/tag (e.g. fully_glob_95, adj_no_glob_65).",
)
parser.add_argument("--wandb-project", default="graph-matching",
                    help="W&B project, shared across experiments (default: graph-matching).")
parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"),
                    choices=["online", "offline", "disabled"],
                    help="W&B mode; use 'offline' on nodes without internet, then `wandb sync`.")
parser.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging entirely.")
cli_args = parser.parse_args()

# ─── Weights & Biases logging ──────────────────────────────────────────────
# One W&B run per trial, grouped under the study. Logs per-epoch val_loss /
# best / patience plus the trial's hyperparameters.
# EXPERIMENT keeps THIS run's logs separate from the other study (used as the
# W&B group, the run-name prefix and a tag).
EXPERIMENT    = cli_args.experiment
USE_WANDB     = not cli_args.no_wandb
WANDB_PROJECT = cli_args.wandb_project
WANDB_MODE    = cli_args.wandb_mode

def objective_pgm(trial, train_dataset, val_dataset, path, real_dataset=None):
    # ─── Noise reduction ───────────────────────────────────────────────────
    # Re-seed every RNG at the start of each trial so that evaluating the SAME
    # hyperparameters always yields the SAME val_loss. This removes the noise
    # coming from (a) random weight init, (b) dropout masks and (c) mini-batch
    # shuffling, turning the Optuna objective into a (near-)deterministic
    # function of x — which is exactly what a GP surrogate assumes.
    set_seed(seed)
    trial_generator = torch.Generator()
    trial_generator.manual_seed(seed)

    # suggest_float(..., log=True) replaces the deprecated suggest_loguniform/uniform.
    # Ranges widened to give the sampler a larger space to explore.
    lr           = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128, 256])
    out_dim      = trial.suggest_categorical("out_dim", [8, 16, 32, 64, 128])
    edge_hidden_dim = trial.suggest_categorical("edge_hidden_dim", [8, 16, 32])
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    dropout_emb  = trial.suggest_float("dropout_emb", 0.0, 0.8)
    attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.8)
    sinkhorn_max_iter = trial.suggest_int("sinkhorn_max_iter", 5, 200)
    sinkhorn_tau = trial.suggest_float("sinkhorn_tau", 5e-3, 5.0, log=True)
    num_layers   = trial.suggest_int("num_layers",       1, 5)
    heads        = trial.suggest_int("heads",           1,   8)

    # generator=trial_generator makes the shuffle order reproducible per trial
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_pyg_matching, generator=trial_generator)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_pyg_matching)
    real_loader = None
    if real_dataset is not None and len(real_dataset) > 0:
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_pyg_matching)

    # One W&B run per trial (params are all suggested at this point)
    run = None
    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity="arg-graph-matching",
            name=f"{EXPERIMENT}_trial_{trial.number}",
            group=EXPERIMENT,
            tags=[EXPERIMENT, "botorch_lognei"],
            config=trial.params,
            mode=WANDB_MODE,
            reinit=True,
        )


    # Flexible model for partial matching
    class MatchingModel_MLPGATv2Sinkhorn_OPT(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, sinkhorn_max_iter, sinkhorn_tau,
                    attention_dropout, dropout_emb, num_layers, heads,
                    edge_dim=EDGE_ATTR_DIM, edge_hidden_dim=16):
            super().__init__()
            # MLP for initial node feature transformation
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_emb),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_emb)
            )
            self.edge_proj = nn.Linear(edge_dim, edge_hidden_dim)
            self.gnn = nn.ModuleList()
            dims = [hidden_dim] * num_layers + [out_dim]
            for i in range(num_layers):
                # Always average the heads so the feature-dim stays dims[i+1]
                self.gnn.append(
                    EdgeAwareGATLayer(dims[i], dims[i+1], edge_hidden_dim, edge_hidden_dim,
                                      heads=heads, attn_dropout=attention_dropout)
                )
            self.dropout = nn.Dropout(p=dropout_emb)
            # # bilinear weight matrix A per affinity
            # std = 1.0 / math.sqrt(out_dim)
            # self.A = nn.Parameter(torch.randn(out_dim, out_dim) * std)
            # self.temperature = temperature
            # InstanceNorm per-sample
            self.inst_norm = nn.InstanceNorm2d(1, affine=True)
            self.sinkhorn_max_iter = sinkhorn_max_iter
            self.sinkhorn_tau = sinkhorn_tau

        def encode(self, x, edge_index, edge_attr):
            edge_attr = self.edge_proj(edge_attr)
            for i, layer in enumerate(self.gnn):
                is_last = i == len(self.gnn) - 1
                x, edge_attr = layer(x, edge_index, edge_attr, update_edge=not is_last)
                if not is_last:
                    x = F.relu(x)
                    x = self.dropout(x)
                    edge_attr = F.relu(edge_attr)
                    edge_attr = self.dropout(edge_attr)
            return x

        def forward(self, batch1, batch2, perm_list, batch_idx1=None, batch_idx2=None, inference=False):
            device = next(self.parameters()).device
            x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
            x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
            edge_attr1, edge_attr2 = batch1.edge_attr.to(device), batch2.edge_attr.to(device)
            perm_list = [p.to(device) for p in perm_list]

            batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
            batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)

            # Apply MLP before GNN
            h1 = self.mlp(x1)
            h2 = self.mlp(x2)
            h1 = self.encode(h1, edge1, edge_attr1)
            h2 = self.encode(h2, edge2, edge_attr2)

            B = batch_idx1.max().item() + 1
            perm_pred_list = []
            for b in range(B):
                h1_b = h1[batch_idx1 == b]
                h2_b = h2[batch_idx2 == b]

                # affinity matrix + normalization + sinkhorn
                sim = torch.matmul(h1_b, h2_b.T) # [n1, n2]
                sim_batched = sim.unsqueeze(0).unsqueeze(1) # [1,1,n1,n2]
                sim_normed = self.inst_norm(sim_batched).squeeze(1) # [1,n1,n2]

                # g1 -> A-graph
                # g2 -> S-graph (partial)
                n1_val = h1_b.size(0)
                n2_val = h2_b.size(0)

                transposed = n1_val > n2_val

                if transposed:
                    # traspose to use dummy_row
                    sim_input = sim_normed.transpose(-2, -1)   # [1, n2, n1]
                    nr = torch.tensor([n2_val], dtype=torch.long, device=device)
                    nc = torch.tensor([n1_val], dtype=torch.long, device=device)
                else:
                    sim_input = sim_normed                     # [1, n1, n2]
                    nr = torch.tensor([n1_val], dtype=torch.long, device=device)
                    nc = torch.tensor([n2_val], dtype=torch.long, device=device)

                S = pygmtools.sinkhorn(
                    sim_input,
                    n1=nr, n2=nc,
                    dummy_row=(n1_val != n2_val),
                    max_iter=self.sinkhorn_max_iter,
                    tau=self.sinkhorn_tau
                )

                if transposed:
                    S = S.transpose(-2, -1)   # rollback to [1, n1, n2]

                perm_pred_list.append(S.squeeze(0))  # [n1, n2]

            # Inference: return the per-graph soft assignments (for F1/metrics).
            if inference:
                return perm_pred_list

            # Training/eval loss: mean WBCE over the graphs in the batch (unchanged).
            loss = 0.0
            for S, P_gt in zip(perm_pred_list, perm_list):
                loss += weighted_bce_loss(S, P_gt)
            return loss / B

    model = MatchingModel_MLPGATv2Sinkhorn_OPT(
        in_dim=train_dataset[0][0].x.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        edge_dim=train_dataset[0][0].edge_attr.size(1),
        edge_hidden_dim=edge_hidden_dim,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_tau=sinkhorn_tau,
        attention_dropout=attn_dropout,
        dropout_emb=dropout_emb,
        num_layers=num_layers,
        heads=heads
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epochs = 30
    patience   = 5
    best_val = float('inf')
    best_f1 = 0.0          # validation F1 at the best-val epoch (auxiliary metric)
    best_real_f1 = 0.0     # real-set F1 at the best-val epoch (auxiliary metric)
    counter = 0
    stopped_early = False
    try:
        for epoch in range(max_epochs):
            model.train()
            for b1, b2, perm in train_loader:
                optimizer.zero_grad()
                loss = model(b1, b2, perm)
                loss.backward()
                optimizer.step()

            # ─── Validation: loss (objective, unchanged) + F1 on the SAME val set ──
            model.eval()
            val_loss = 0.0
            tp = fp = fn = 0
            with torch.no_grad():
                for b1, b2, perm in val_loader:
                    pred_list = model(b1, b2, perm, inference=True)  # per-graph soft S
                    # same per-batch mean WBCE as the training forward (loss / n_graphs)
                    batch_loss = sum(weighted_bce_loss(S, P.to(S.device))
                                     for S, P in zip(pred_list, perm)) / len(pred_list)
                    val_loss += batch_loss.item()
                    for S, P in zip(pred_list, perm):
                        P_hard = hard_perm_from_scores(S)
                        tpi, fpi, fni, _ = permutation_confusion_counts(P_hard, P.to(S.device))
                        tp += tpi; fp += fpi; fn += fni
            val_loss /= len(val_loader)
            _, _, val_f1 = permutation_precision_recall_f1(tp, fp, fn)

            # F1 on the connectivity-matched real set (logging only, does not affect the objective)
            real_f1 = eval_f1_on_loader(model, real_loader) if real_loader is not None else None

            trial.report(val_loss, epoch)

            if val_loss < best_val:
                best_val = val_loss
                best_f1 = val_f1          # keep the F1 measured at the best-loss epoch
                if real_f1 is not None:
                    best_real_f1 = real_f1
                counter = 0
            else:
                counter += 1

            # Early-stopping visibility (flush=True so it shows live in SLURM logs)
            _rf1 = f" | real_f1={real_f1:.4f}" if real_f1 is not None else ""
            print(f"[trial {trial.number}] epoch {epoch + 1:02d}/{max_epochs} | "
                  f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f}{_rf1} | "
                  f"best_loss={best_val:.4f} | best_f1={best_f1:.4f} | "
                  f"patience={counter}/{patience}", flush=True)

            if run is not None:
                _log = {"epoch": epoch + 1, "val_loss": val_loss, "val_f1": val_f1,
                        "best_val": best_val, "best_f1": best_f1, "patience": counter}
                if real_f1 is not None:
                    _log["real_f1"] = real_f1
                    _log["best_real_f1"] = best_real_f1
                run.log(_log)

            if trial.should_prune():
                raise optuna.TrialPruned()
            if counter >= patience:
                print(f"[trial {trial.number}] EARLY STOP at epoch {epoch + 1} "
                      f"(best val_loss={best_val:.4f}, val_f1={best_f1:.4f})", flush=True)
                stopped_early = True
                break
    except torch.cuda.OutOfMemoryError:
        # A too-large sampled config (widened search space) did not fit in GPU
        # memory. Skip this trial instead of letting the exception crash the whole
        # study: prune it so the BO sampler just moves on to the next config.
        torch.cuda.empty_cache()
        print(f"[trial {trial.number}] CUDA OOM -> pruning this trial", flush=True)
        if run is not None:
            run.finish()
        raise optuna.TrialPruned()
    finally:
        # Free this trial's GPU memory so fragmentation does not build up across a
        # long, uncapped study (many trials with different model/batch sizes).
        del model, optimizer
        torch.cuda.empty_cache()

    if not stopped_early:
        print(f"[trial {trial.number}] reached the {max_epochs}-epoch cap WITHOUT "
              f"early stopping (best val_loss={best_val:.4f}, val_f1={best_f1:.4f}) -> "
              f"30 epochs may be too few, consider raising max_epochs.", flush=True)

    # Persist the trial's validation loss and F1 (val + real) as Optuna user attrs (in study.pkl)
    trial.set_user_attr("val_f1", best_f1)
    trial.set_user_attr("val_loss", best_val)
    trial.set_user_attr("real_f1", best_real_f1)

    # save best trial info (loss + F1 val/real alongside the hyperparameters)
    if trial.number == 0 or best_val <= trial.study.best_value:
        result = {
            "val_loss": best_val,
            "val_f1": best_f1,
            "real_f1": best_real_f1,
            "params": trial.params
        }
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "best_trial_results.json"), "w") as f:
            json.dump(result, f, indent=2)

    if run is not None:
        run.summary["best_val"] = best_val
        run.summary["best_f1"] = best_f1
        run.summary["best_real_f1"] = best_real_f1
        run.finish()

    return best_val

### FUNCTIONS WITH BCE
def bce_permutation_loss(P, P_gt, eps: float = 1e-9):
    """Element-wise Binary Cross Entropy loss between prediction and ground truth."""
    assert P.shape == P_gt.shape, f"Shape mismatch: P={P.shape}, P_gt={P_gt.shape}"
    return - (P_gt * torch.log(P + eps) + (1 - P_gt) * torch.log(1 - P + eps)).mean()

def weighted_bce_loss(S_pred, S_gt):
    """
    Computes the Weighted Binary Cross-Entropy for the Permutation Loss.
    
    Args:
        S_pred (torch.Tensor): Sinkhorn output [B, N, N], probabilities between 0 and 1.
        S_gt (torch.Tensor): Ground truth matrix [B, N, N], binary values (0 or 1).

    Returns:
        torch.Tensor: The scalar value of the average loss.
    """
    # 1. Compute the positive class weight (pos_weight)
    # In graph matching, matches (1) are very rare compared to non-matches (0).
    # pos_weight = (total number of 0s) / (total number of 1s)
    num_pos = S_gt.sum() 
    num_neg = (1.0 - S_gt).sum()
    
    # Prevents division by zero if the batch has no matches (unlikely but safe)
    if num_pos > 0:
        pos_weight = num_neg / num_pos
    else:
        pos_weight = torch.tensor(1.0, device=S_pred.device)
        
    # 2. Compute BCE without reduction
    # We use reduction='none' to obtain a loss map of dimension [B, N, N]
    # S_pred is clipped (or eps is added internally by PyTorch) 
    # to avoid probabilities exactly equal to 0 or 1 causing NaN in logarithms.
    bce_loss_map = F.binary_cross_entropy(S_pred, S_gt.float(), reduction='none')
    
    # 3. Apply weight to ONLY positive matches
    # We create a weight matrix with the same dimension as the loss
    # For each cell: if ground truth is 1, the weight is 'pos_weight', if 0 the weight is 1.
    weight_matrix = S_gt * pos_weight + (1.0 - S_gt) * 1.0
    
    # Multiply the original loss by the weight matrix
    weighted_bce_loss_map = bce_loss_map * weight_matrix
    
    # 4. Final reduction
    # Compute the mean over the entire tensor to return the scalar
    return weighted_bce_loss_map.mean()


def hard_perm_from_scores(P: torch.Tensor) -> torch.Tensor:
    """Convert a soft permutation matrix into a hard assignment (one per column)."""
    hard = torch.zeros_like(P)
    hard[P.argmax(dim=0), torch.arange(P.shape[1], device=P.device)] = 1
    return hard


def permutation_confusion_counts(P_pred_hard: torch.Tensor, P_gt: torch.Tensor):
    """Return (tp, fp, fn, tn) counts comparing hard predictions to ground truth."""
    pred = (P_pred_hard > 0.5).to(P_gt.dtype)
    tp = (pred * P_gt).sum().item()
    fp = (pred * (1 - P_gt)).sum().item()
    fn = ((1 - pred) * P_gt).sum().item()
    tn = ((1 - pred) * (1 - P_gt)).sum().item()
    return tp, fp, fn, tn


def permutation_precision_recall_f1(tp, fp, fn, eps: float = 1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


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


# ─── Real (out-of-distribution) validation set ─────────────────────────────────
# Same loading/normalization as pgm_training_ws_room_inc_WBCE_scratch.py, so we can
# also log the F1 on the connectivity-matched real set (real / real_adj / real_fully).
node_type_mapping = {"room": [1, 0], "ws": [0, 1]}


class _GraphWrapper:
    """Stub to unpickle situational_graphs_wrapper.GraphWrapper (we only need `.graph`)."""
    pass


class _WrapperUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'situational_graphs_wrapper' in module or name == 'GraphWrapper':
            return _GraphWrapper
        return super().find_class(module, name)


def _load_situational_graph(path: str) -> nx.DiGraph:
    with open(path, 'rb') as f:
        obj = _WrapperUnpickler(f).load()
    if isinstance(obj, nx.Graph):
        return obj
    g = getattr(obj, 'graph', None)
    if isinstance(g, nx.Graph):
        return g
    raise TypeError(f"Unexpected object in {path}: {type(obj)}")


def _strip_scene_prefix(node_id) -> str:
    s = str(node_id)
    for pre in ('s_', 'a_'):
        if s.startswith(pre):
            return s[len(pre):]
    return s


def _real_nx_to_pyg(graph: nx.DiGraph) -> Data:
    """Same reduced [type(2), length_or_-1(1)] node schema + edge_attr as
    nx_to_pyg_data_preserve_order in pgm_training_adj_glob_edgefeat.py (edge_features.py)."""
    node_ids = [str(n) for n in graph.nodes()]
    raw_id_map = {n: i for i, n in enumerate(graph.nodes())}
    x = node_features(graph, node_type_mapping)
    edge_index, edge_attr = build_edge_index_and_attr(graph, raw_id_map)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data


def deserialize_norm_stats(path: str, filename: str = "norm_stats.pt"):
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        return None
    stats = torch.load(full_path, map_location="cpu")
    # edge_mean/edge_std absent in pre-edge-normalization datasets -> None, caller falls back.
    return stats["mean"], stats["std"], stats.get("edge_mean"), stats.get("edge_std")


def build_real_validation_pairs(real_dir: str, verbose: bool = True,
                                mean: torch.Tensor = None, std: torch.Tensor = None,
                                edge_mean: torch.Tensor = None, edge_std: torch.Tensor = None):
    """(Data_A, Data_S, P, scene) tuples from the real folders; normalized with the
    training (mean, std) when given. Mirrors scratch.py's loader."""
    pairs = []
    if not os.path.isdir(real_dir):
        if verbose:
            print(f"  [real] Directory not found: {real_dir}")
        return pairs
    for folder in sorted(os.listdir(real_dir)):
        d = os.path.join(real_dir, folder)
        if not os.path.isdir(d):
            continue
        prior_p = os.path.join(d, 'Prior.pkl')
        online_p = os.path.join(d, 'Online.pkl')
        gt_p = os.path.join(d, 'ground_truth.json')
        if not (os.path.exists(prior_p) and os.path.exists(online_p) and os.path.exists(gt_p)):
            if verbose:
                print(f"  [real] Skipping '{folder}': missing Prior/Online/ground_truth.")
            continue
        g_a = _load_situational_graph(prior_p)
        g_s = _load_situational_graph(online_p)
        with open(gt_p) as f:
            gt = json.load(f)
        data_a = _real_nx_to_pyg(g_a)
        data_s = _real_nx_to_pyg(g_s)
        if mean is not None and std is not None:
            data_a.x = (data_a.x - mean) / (std + 1e-8)
            data_s.x = (data_s.x - mean) / (std + 1e-8)
        if edge_mean is not None and edge_std is not None:
            data_a.edge_attr = normalize_edge_attr(data_a.edge_attr, edge_mean, edge_std)
            data_s.edge_attr = normalize_edge_attr(data_s.edge_attr, edge_mean, edge_std)
        a_index = {}
        for i, nid in enumerate(data_a.node_names):
            a_index.setdefault(nid, i)
            a_index.setdefault(_strip_scene_prefix(nid), i)
        s_index = {}
        for j, nid in enumerate(data_s.node_names):
            s_index.setdefault(nid, j)
            s_index.setdefault(_strip_scene_prefix(nid), j)
        gt_entries = list(gt.get('rooms', {}).items()) + [tuple(p) for p in gt.get('ws', [])]
        P = torch.zeros((data_a.num_nodes, data_s.num_nodes), dtype=torch.float32)
        matched = 0
        for s_id, a_id in gt_entries:
            j = s_index.get(str(s_id), s_index.get(_strip_scene_prefix(s_id)))
            i = a_index.get(str(a_id), a_index.get(_strip_scene_prefix(a_id)))
            if i is not None and j is not None:
                P[i, j] = 1.0
                matched += 1
        if verbose:
            warn = "  <-- LOW COVERAGE" if matched < 0.5 * max(len(gt_entries), 1) else ""
            print(f"  [real] {folder:24} | A(g1)={data_a.num_nodes:3} S(g2)={data_s.num_nodes:3} "
                  f"| GT matched {matched}/{len(gt_entries)}{warn}")
        pairs.append((data_a, data_s, P, folder))
    if verbose:
        print(f"  [real] Loaded {len(pairs)} real validation scene(s).")
    return pairs


def eval_f1_on_loader(model, loader) -> float:
    """Micro-averaged F1 of the hardened Sinkhorn assignment over a loader."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for b1, b2, perm in loader:
            pred_list = model(b1, b2, perm, inference=True)
            for S, P in zip(pred_list, perm):
                P_hard = hard_perm_from_scores(S)
                tpi, fpi, fni, _ = permutation_confusion_counts(P_hard, P.to(S.device))
                tp += tpi; fp += fpi; fn += fni
    _, _, f1 = permutation_precision_recall_f1(tp, fp, fn)
    return f1


print("Loading dataset...")

#load preprocessed dataset
gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", EXPERIMENT)
models_path = os.path.join(GNN_PATH, 'models', "partial_graph_matching", EXPERIMENT)

train_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "train_dataset.pkl"
)
val_list = deserialize_graph_matching_dataset(
    gm_local_preprocessed_path,
    "valid_dataset.pkl"
)

train_dataset = GraphMatchingDataset(train_list)
val_dataset = GraphMatchingDataset(val_list)

# ─── Real validation set (connectivity-matched, normalized with train stats) ───
# adj_* -> real_adj, fully_* -> real_fully, otherwise real. Used ONLY for logging
# the F1 (the optimization objective stays the validation loss).
_reference = ("adj" if EXPERIMENT.startswith("adj")
              else "fully" if EXPERIMENT.startswith("fully") else "equal")
_real_variant = {"adj": "real_adj", "fully": "real_fully"}.get(_reference, "real")
real_val_dir = os.path.join(GNN_PATH, _real_variant)
if not os.path.isdir(real_val_dir):
    _fallback_real_dir = os.path.join(DATASET_DUMP_REAL_ROOT, _real_variant)
    if os.path.isdir(_fallback_real_dir):
        print(f"[real] {real_val_dir} not found -> using {_fallback_real_dir}")
        real_val_dir = _fallback_real_dir
_norm = deserialize_norm_stats(gm_local_preprocessed_path)
if _norm is None:
    _rmean = _rstd = _redge_mean = _redge_std = None
    print(f"[WARNING] norm_stats.pt not found in {gm_local_preprocessed_path}; the real F1 "
          f"will be computed on UNNORMALIZED features (not comparable to training).")
else:
    _rmean, _rstd, _redge_mean, _redge_std = _norm
_in_dim = train_dataset[0][0].x.size(1)
print(f"Loading real validation set from {real_val_dir} (normalized={_norm is not None})...")
_real_pairs = build_real_validation_pairs(real_val_dir, mean=_rmean, std=_rstd,
                                          edge_mean=_redge_mean, edge_std=_redge_std)
real_dataset = GraphMatchingDataset([(a, s, P) for a, s, P, _ in _real_pairs]) if _real_pairs else None

print("Continuing hyperparameter optimization WBCE...")


def logNEI_candidates_func(train_x, train_obj, train_con, bounds, pending_x=None):
    """GP surrogate + q-Log Noisy Expected Improvement (LogNEI) acquisition.

    Optuna's BoTorchSampler calls this to pick the next configuration:
      train_x   : (n, d) past params, already normalized to the unit cube
      train_obj : (n, 1) past objective values, already sign-flipped so that
                  HIGHER is better (BoTorch always maximizes)
      bounds    : (2, d) = [[0...], [1...]]
    Returns the next candidate (1, d) in the same normalized space; Optuna
    un-normalizes it back to real hyperparameter values.
    """
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood

    # BoTorch works best in double precision
    train_x = train_x.to(torch.float64)
    train_obj = train_obj.to(torch.float64)
    bounds = bounds.to(torch.float64)

    # ── GP surrogate: learns loss(x) + calibrated uncertainty ──────────────
    model = SingleTaskGP(
        train_x, train_obj,
        outcome_transform=Standardize(m=train_obj.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # ── LogNEI acquisition: noise-robust, numerically stable EI ────────────
    acqf = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        prune_baseline=True,
    )

    # ── Optimize the acquisition to get the next point to evaluate ─────────
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )
    return candidates


study_path = os.path.join(models_path, "study.pkl")

# GP + BoTorch + LogNEI sampler
sampler = BoTorchSampler(
    candidates_func=logNEI_candidates_func,
    n_startup_trials=20,   # random warm-up before the GP is trustworthy (~2x dim)
    seed=seed,
)

# No pruner: BoTorch fits the GP only on COMPLETE trials, so pruned trials
# would give the surrogate no observation. With a small budget we'd rather keep
# every data point; the in-trial early stopping (patience 5) already caps cost.
pruner = optuna.pruners.NopPruner()

# Load the previous study
if os.path.exists(study_path):
    with open(study_path, "rb") as f:
        study = pickle.load(f)
    # IMPORTANT: a pickled study restores its OLD sampler/pruner (default TPE +
    # MedianPruner). Override both, otherwise the settings above are ignored.
    study.sampler = sampler
    study.pruner = pruner
    print(f"Loaded existing study with {len(study.trials)} trials.")
    print("NOTE: the search-space ranges were widened. Old trials sampled by "
          "TPE under the previous (narrower) distributions may bias the GP — "
          "consider deleting study.pkl to start a clean BoTorch study.")
else:
    print("No existing study found, starting a new one.")
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

# ── Continuous checkpointing + time budget ─────────────────────────────────
# The job runs on SLURM with a 2-day wall-time limit. Instead of capping the
# number of trials, we run until a self-imposed timeout (kept below the SLURM
# limit so the final save + plot still execute) and pickle the study after
# EVERY trial. If the job is killed anyway, study.pkl already holds every
# completed trial and the next job resumes from it.
TIME_BUDGET_SEC = 46 * 3600   # ~46h, ~2h margin under a 48h SLURM wall-time

def save_study_callback(study, trial):
    with open(study_path, "wb") as f:
        pickle.dump(study, f)

study.optimize(
    lambda trial: objective_pgm(trial, train_dataset, val_dataset, models_path, real_dataset),
    n_trials=None,               # no cap: run as many trials as fit in the budget
    timeout=TIME_BUDGET_SEC,     # stop gracefully before SLURM kills the job
    callbacks=[save_study_callback],
)

# Final save (also covered by the per-trial callback above)
with open(study_path, "wb") as f:
    pickle.dump(study, f)

# Plot the study results
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(os.path.join(models_path, "opt_history.html"))
# fig.write_image(os.path.join(models_path, "opt_history.png"))
