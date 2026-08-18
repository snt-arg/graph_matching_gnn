#graph_matching.py
GNN_PATH = './GNN/'
import os
if not os.path.exists(GNN_PATH):
    os.makedirs(GNN_PATH)

# %%
# Install packages
import subprocess
import sys

# Install required packages
subprocess.check_call(["uv", "pip", "install", "torch", "torch-geometric", "scikit-learn", "pandas", "shapely", "seaborn", "pygmtools", "numpy<2", "moviepy<2.0.0", "matplotlib", "tensorboard", "optuna", "plotly", "kaleido", "wandb"])
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
import wandb


from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GCNConv

from moviepy.editor import ImageSequenceClip
from typing import Optional, Literal
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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

# %% [markdown]
# # Utilities

# %% [markdown]
# ## Dataset

# %%
#----------------------------------------
#            DATASET UTILS
#----------------------------------------

def deserialize_MSD_dataset(data_path, original_path=None, noise_path=None, dimensions_path=None):
    dataset_dir = Path(data_path)

    dimensions = []
    if dimensions_path is not None:
        # Load dimensions
        dimensions_file = dataset_dir / f"{dimensions_path}.pickle"
        if not dimensions_file.exists():
            raise FileNotFoundError(f"Dimensions file not found at {dimensions_file}")
        with open(dimensions_file, 'rb') as f:
            dimensions = pickle.load(f)

    # Clear existing graphs
    original = []
    noise = []

    if original_path is not None:
        original_dir = dataset_dir / original_path
        original_files = sorted(original_dir.glob("*.pt"), key=lambda f: int(f.stem))
        print(f"Loading {len(original_files)} original graphs...")
        for file in tqdm(original_files, desc="Original graphs"):
            with open(str(file), "rb") as f:
                graph = pickle.load(f)
                graph.graph['name'] = file.stem
            original.append(graph)

    if noise_path is not None:
        def extract_numeric_key(file):
            """Extracts (X, Y) from filenames like 'X_Y.pt' for proper numeric sorting."""
            name_parts = file.stem.split("_")
            return int(name_parts[0]), int(name_parts[1])

        noise_dir = dataset_dir / noise_path
        noise_files = sorted(noise_dir.glob("*.pt"), key=extract_numeric_key)
        print(f"Loading {len(noise_files)} noise graphs...")
        for file in tqdm(noise_files, desc="Noise graphs"):
            with open(str(file), "rb") as f:
                graph = pickle.load(f)
                graph.graph['name'] = file.stem
            noise.append(graph)

    return original, noise, dimensions

def serialize_graph_matching_dataset(pairs: List[Tuple[Data, Data, torch.Tensor]], path: str, filename: str = "train_dataset.pkl"):
    """
    Serialize a list of (Data1, Data2, PermutationMatrix) tuples to a file.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)

    with open(full_path, 'wb') as f:
        pickle.dump(pairs, f)

    print(f"Serialized {len(pairs)} pairs to {full_path}")

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

def plot_a_graph(graphs_list, path=None, viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=False, viz_walls=True):
    """
    Visualizes geometries, wall segments, and graph edges for multiple apartments in 2D.

    Parameters:
    graphs_list (list of networkx.Graph): List of graphs with nodes ('type', 'center', 'normal') and edges for the apartments.
    viz_normals (bool): If True, plots wall segment normals.
    viz_rooms (bool): If True, displays room polygons.
    viz_ws (bool): If True, displays wall segments.
    viz_openings (bool): If True, displays openings (doors and windows).
    viz_wall_edges (bool): If True, displays edges between wall segments.
    viz_connection_edges (bool): If True, displays edges connecting rooms via openings.
    viz_walls (bool): If True, displays wall nodes and their edges.
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    legend_added = False  # Flag to ensure the legend is added only once
    normal_added = False  # Flag to ensure the "Normal" label is added only once

    for graphs in graphs_list:
        # Visualize room polygons
        if viz_rooms:
            room_nodes = [n for n, d in graphs.nodes(data=True) if d['type'] == 'room']
            for idx, room_node in enumerate(room_nodes):
                room_data = graphs.nodes[room_node]
                # Plot the polygon
                room_polygon = Polygon(room_data['polygon'])
                x, y = room_polygon.exterior.xy
                ax.plot(x, y, color='black', alpha=0.2, label='Room polygon' if not legend_added and idx == 0 else "")
                # Draw room centroids
                ax.scatter(room_data['center'][0], room_data['center'][1], color='blue', s=100, label='Room centroid' if not legend_added and idx == 0 else "")

        # Visualize wall nodes and edges
        if viz_walls:
            wall_nodes = [n for n, d in graphs.nodes(data=True) if d['type'] == 'wall']
            for idx, wn in enumerate(wall_nodes):
                wall_data = graphs.nodes[wn]
                # Plot the polygon of the wall
                wall_polygon = Polygon(wall_data['polygon'])
                x, y = wall_polygon.exterior.xy
                ax.plot(x, y, color='purple', linestyle='-', label='Wall polygon' if not legend_added and idx == 0 else "")
                ax.scatter(wall_data['center'][0], wall_data['center'][1], color='purple', s=50, label='Wall centroid' if not legend_added and idx == 0 else "")

            if viz_normals:
                wall_ws = [n for n, d in graphs.nodes(data=True) if d['type'] == 'wall_ws']
                for idx, wn in enumerate(wall_ws):
                    ws_data = graphs.nodes[wn]
                    ax.scatter(ws_data['center'][0], ws_data['center'][1], color='purple', s=20, label='Wall ws' if not legend_added and idx == 0 else "")
                    ax.arrow(ws_data['center'][0], ws_data['center'][1],
                             ws_data['normal'][0], ws_data['normal'][1],
                             head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal' if not normal_added else "")
                    normal_added = True

            wall_edges = [(u, v) for u, v, d in graphs.edges(data=True) if 'wall' in u or 'wall' in v]
            for idx, edge in enumerate(wall_edges):
                start_node = graphs.nodes[edge[0]]
                end_node = graphs.nodes[edge[1]]
                ax.plot([start_node['center'][0], end_node['center'][0]],
                        [start_node['center'][1], end_node['center'][1]],
                        color='purple', linestyle='--', label='Wall edge' if not legend_added and idx == 0 else "")

        # Visualize openings
        if viz_openings:
            opening_nodes = [n for n, d in graphs.nodes(data=True) if 'door' in d['type'] or 'window' in d['type']]
            for idx, on in enumerate(opening_nodes):
                opening_data = graphs.nodes[on]
                opening_polygon = Polygon(opening_data['polygon'])
                x, y = opening_polygon.exterior.xy
                ax.plot(x, y, color='orange', label='Opening polygon' if not legend_added and idx == 0 else "")
                # Draw opening centroids
                ax.scatter(opening_data['center'][0], opening_data['center'][1], color='orange', s=10, label='Opening centroid' if not legend_added and idx == 0 else "")

            if viz_normals:
                opening_ws = [n for n, d in graphs.nodes(data=True) if d['type'] == 'door_ws' or d['type'] == 'window_ws']
                for idx, wn in enumerate(opening_ws):
                    ws_data = graphs.nodes[wn]
                    ax.scatter(ws_data['center'][0], ws_data['center'][1], color='orange', s=10, label='Opening ws' if not legend_added and idx == 0 else "")
                    ax.arrow(ws_data['center'][0], ws_data['center'][1],
                             ws_data['normal'][0], ws_data['normal'][1],
                             head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal' if not normal_added else "")
                    normal_added = True

            # Draw opening edges
            open_edges = [(u, v) for u, v, d in graphs.edges(data=True) if 'door' in u or 'window' in v or 'door' in v or 'window' in u]
            for idx, edge in enumerate(open_edges):
                start_node = graphs.nodes[edge[0]]
                end_node = graphs.nodes[edge[1]]
                ax.plot([start_node['center'][0], end_node['center'][0]],
                        [start_node['center'][1], end_node['center'][1]],
                        color='orange', linestyle='--', label='Opening edge' if not legend_added and idx == 0 else "")

        # Visualize ws room
        if viz_ws:
            ws_nodes = [n for n, d in graphs.nodes(data=True) if d['type'] == 'ws']
            for idx, wn in enumerate(ws_nodes):
                ws_data = graphs.nodes[wn]
                ax.scatter(ws_data['center'][0], ws_data['center'][1], color='red', s=20, label='Ws segment' if not legend_added and idx == 0 else "")
                if viz_room_normals:
                    ax.arrow(ws_data['center'][0], ws_data['center'][1],
                             ws_data['normal'][0], ws_data['normal'][1],
                             head_width=0.1, head_length=0.1, fc='green', ec='green', label='Normal' if not normal_added else "")
                    normal_added = True
                if 'limits' in ws_data:
                    limit_1, limit_2 = ws_data['limits']
                    ax.plot([limit_1[0], limit_2[0]],
                            [limit_1[1], limit_2[1]],
                            color='black', linewidth=1.0,
                            label='Ws limits' if idx == 0 else "")
            ws_edges = [(u, v) for u, v, d in graphs.edges(data=True) if 'ws_same_room' in d['type'] or 'ws_belongs_room' in d['type']]
            for idx, edge in enumerate(ws_edges):
                start_node = graphs.nodes[edge[0]]
                end_node = graphs.nodes[edge[1]]
                ax.plot([start_node['center'][0], end_node['center'][0]],
                    [start_node['center'][1], end_node['center'][1]],
                    color='gray', linestyle='--', label='Ws edge' if not legend_added and idx == 0 else "")

        # Visualize connection edges
        if viz_room_connection:
            connection_edges = [(u, v) for u, v, d in graphs.edges(data=True) if 'connected' in d['type']]
            for idx, edge in enumerate(connection_edges):
                start_node = graphs.nodes[edge[0]]
                end_node = graphs.nodes[edge[1]]
                ax.plot([start_node['center'][0], end_node['center'][0]],
                        [start_node['center'][1], end_node['center'][1]],
                        color='blue', linestyle='-', label='Connection edge' if not legend_added and idx == 0 else "")

        legend_added = True  # Set the flag to True after processing the first graph

    plt.title("Apartment Graph Visualization")
    plt.legend()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

node_type_mapping = {"room": [1, 0], "ws": [0, 1]}

def pyg_data_to_nx_digraph(data: Data, graph_list: List[nx.DiGraph]) -> nx.DiGraph:
    """
    Convert a PyTorch Geometric Data object back to a NetworkX DiGraph,
    restoring original node IDs using data.node_names and data.permutation,
    matching with the graph in graph_list that has the same name.
    """
    assert hasattr(data, 'node_names'), \
        "Data object must contain 'node_names' to restore original node IDs."
    assert hasattr(data, 'permutation'), \
        "Data object must contain 'permutation' to reorder nodes."
    assert hasattr(data, 'name'), \
        "Data object must contain 'name' to match with graph_list."

    matching_graph = next((g for g in graph_list if g.graph.get('name') == data.name), None)
    if matching_graph is None:
        raise ValueError(f"No graph with name {data.name} found in graph_list.")

    orig_names = data.node_names
    perm = data.permutation.tolist()
    node_ids = [orig_names[idx] for idx in perm]

    G = nx.DiGraph()
    for node_id in node_ids:
        if node_id in matching_graph.nodes:
            G.add_node(node_id, **matching_graph.nodes[node_id])

    for u_idx, v_idx in data.edge_index.t().tolist():
        u = node_ids[u_idx]
        v = node_ids[v_idx]
        if matching_graph.has_edge(u, v):
            G.add_edge(u, v, **matching_graph.edges[u, v])

    G.graph['name'] = data.name
    return G


def nx_to_pyg_data_preserve_order(graph: nx.DiGraph) -> Data:
    """
    Convert a NetworkX DiGraph to a PyTorch Geometric Data object,
    preserving node insertion order, storing 'node_names' and an identity 'permutation'.
    """
    node_ids = list(graph.nodes())
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    x = torch.stack([
        torch.tensor(
            node_type_mapping[graph.nodes[n]['type']] +
            list(graph.nodes[n]['center']) +
            list(graph.nodes[n]['normal']) +
            [graph.nodes[n].get('length', -1)],
            dtype=torch.float32
        )
        for n in node_ids
    ])

    edge_index = torch.tensor(
        [[id_map[u], id_map[v]] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous() if graph.edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.name = graph.graph.get('name')
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data

def split_graphs_stratified(
    pairs: List[Tuple[Data, Data, torch.Tensor]],
    train_frac: float = 0.7,
    val_frac: float   = 0.15,
    test_frac: float  = 0.15,
    n_bins: int       = 5,
    seed: int         = seed,
    stratify_on: str  = "g2"     # "g1" oppure "g2"
) -> Tuple[
    List[Tuple[Data,Data,torch.Tensor]],
    List[Tuple[Data,Data,torch.Tensor]],
    List[Tuple[Data,Data,torch.Tensor]]
]:
    """
    Stratified split of graph-matching pairs into train/val/test.
    Puoi stratificare sulla dimensione di g1 o di g2.
    """
    assert stratify_on in ("g1","g2"), "stratify_on must be 'g1' or 'g2'"
    total = train_frac + val_frac + test_frac
    assert abs(total - 1.0) < 1e-6, "train+val+test fractions must sum to 1.0"

    # scegli la dimensione su cui stratificare
    if stratify_on == "g1":
        sizes = np.array([g1.num_nodes for g1, g2, P in pairs])
    else:
        sizes = np.array([g2.num_nodes for g1, g2, P in pairs])

    # quantile‑binning per equal‑frequency
    while n_bins > 1:
        try:
            size_bins = pd.qcut(sizes, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            n_bins -= 1
            continue
        counts = np.bincount(size_bins, minlength=n_bins)
        if np.all(counts >= 2):
            break
        n_bins -= 1

    idx = np.arange(len(pairs))
    if n_bins <= 1:
        # fallback random split
        train_idx, temp_idx = train_test_split(idx, test_size=1-train_frac, random_state=seed)
        rel_val = val_frac/(val_frac+test_frac)
        val_idx, test_idx = train_test_split(temp_idx, test_size=1-rel_val, random_state=seed)
    else:
        train_idx, temp_idx = train_test_split(
            idx, test_size=(1.0-train_frac),
            random_state=seed, stratify=size_bins
        )
        rel_val = val_frac/(val_frac+test_frac)
        temp_bins = size_bins[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1.0-rel_val),
            random_state=seed, stratify=temp_bins
        )

    train = [pairs[i] for i in train_idx]
    val   = [pairs[i] for i in val_idx]
    test  = [pairs[i] for i in test_idx]
    return train, val, test

# Controllo rapido delle distribuzioni
def describe(split, name):
    sz = [g1.num_nodes for g1, _, _ in split]
    print(f"{name}: count={len(split)}, nodes min={min(sz)}, max={max(sz)}, mean={np.mean(sz):.1f}")

def generate_matching_pair_as_data(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    pairs_list: List[Tuple[Data, Data, torch.Tensor]],
    noise_level: int = None
) -> None:
    """
    Generate a matching pair for partial graph matching:
    - g1: complete graph (reference)
    - g2: partial graph to be permuted
    Stores (Data_g1, Data_g2_permuted, P) in pairs_list, where P is ground truth of shape [|g1|, |g2|].
    """
    # Convert reference graph
    pyg_g1 = nx_to_pyg_data_preserve_order(g1)

    # Prepare original names and permutation for g2
    orig_names = list(g2.nodes())
    num_g1 = g1.number_of_nodes()
    num_g2 = len(orig_names)
    perm_indices = torch.randperm(num_g2)

    # Build permuted g2
    g2_perm = nx.DiGraph()
    g2_perm.graph['name'] = g2.graph.get('name', '')
    for new_idx, orig_idx in enumerate(perm_indices.tolist()):
        orig_id = orig_names[orig_idx]
        g2_perm.add_node(new_idx, **g2.nodes[orig_id])
    # Remap edges
    orig_to_new = {orig_names[idx]: new for new, idx in enumerate(perm_indices.tolist())}
    for u, v, data_edge in g2.edges(data=True):
        if u in orig_to_new and v in orig_to_new:
            g2_perm.add_edge(orig_to_new[u], orig_to_new[v], **data_edge)

    # Convert permuted graph and attach metadata
    pyg_g2 = nx_to_pyg_data_preserve_order(g2_perm)
    pyg_g2.permutation = perm_indices
    pyg_g2.node_names = orig_names
    if noise_level is not None:
        pyg_g2.noise_level = noise_level

    # Build partial assignment ground truth P [|g1| x |g2|]
    P = torch.zeros((num_g1, num_g2), dtype=torch.float32)
    g1_ids = list(g1.nodes())
    # For each permuted node in g2, find matching index in g1
    for j, orig_idx in enumerate(perm_indices.tolist()):
        orig_id = orig_names[orig_idx]
        if orig_id in g1_ids:
            i = g1_ids.index(orig_id)
            P[i, j] = 1.0

    # Append without transpose to keep shape [|g1|, |g2|]
    pairs_list.append((pyg_g1, pyg_g2, P))


def plot_two_graphs_with_matching(graphs_list, gt_perm, original_graphs, path=None, noise_graphs=None, pred_perm=None,
                                  viz_rooms=True, viz_ws=True,
                                  viz_room_connection=True,
                                  viz_normals=False, viz_room_normals=False,
                                  match_display="all"):
    assert match_display in {"all", "correct", "wrong"}, "match_display must be one of: 'all', 'correct', 'wrong'"
    assert len(graphs_list) == 2, "graphs_list must contain exactly two graphs."
    if noise_graphs is None:
        noise_graphs = original_graphs

    # Extract tensors and original node order
    g1tensor, g2tensor = copy.deepcopy(graphs_list[0]), copy.deepcopy(graphs_list[1])
    # Node names for g1 in original order
    node_names1 = list(g1tensor.node_names)
    # Reconstruct node names for g2 according to its permutation
    orig_names2 = list(g2tensor.node_names)
    perm = g2tensor.permutation.tolist()
    node_names2 = [orig_names2[p] for p in perm]

    # Convert to NetworkX
    g1 = copy.deepcopy(pyg_data_to_nx_digraph(g1tensor, original_graphs))
    g2_original = copy.deepcopy(pyg_data_to_nx_digraph(g2tensor, noise_graphs))
    g2 = g2_original.copy()

    # Translate g2 for side-by-side plot
    max_x_g1 = max(data['center'][0] for _, data in g1.nodes(data=True))
    min_x_g2 = min(data['center'][0] for _, data in g2.nodes(data=True))
    translation_x = (max_x_g1 - min_x_g2) + 10.0
    for _, data in g2.nodes(data=True):
        data['center'][0] += translation_x
        if 'polygon' in data:
            poly = data['polygon']
            if isinstance(poly, Polygon):
                data['polygon'] = translate(poly, xoff=translation_x)
            else:
                data['polygon'] = Polygon([(x + translation_x, y) for x, y in poly])
        if 'limits' in data:
            data['limits'] = [[x + translation_x, y] for x, y in data['limits']]

    fig, ax = plt.subplots(figsize=(16, 10))
    legend_added = set()

    def plot_graph(g, is_g1):
        color_room = 'lightblue' if is_g1 else 'navajowhite'
        color_ws = 'red' if is_g1 else 'purple'
        prefix = "(G1)" if is_g1 else "(G2)"

        if viz_rooms:
            for n, d in g.nodes(data=True):
                if d['type'] == 'room' and 'polygon' in d:
                    poly = Polygon(d['polygon']) if not isinstance(d['polygon'], Polygon) else d['polygon']
                    x, y = poly.exterior.xy
                    ax.fill(x, y, color=color_room, alpha=0.3,
                            label=f"Room polygon {prefix}" if f"room-poly-{prefix}" not in legend_added else "")
                    ax.scatter(d['center'][0], d['center'][1], color='blue', s=80,
                               label=f"Centroid {prefix}" if f"room-pt-{prefix}" not in legend_added else "")
                    legend_added.update({f"room-poly-{prefix}", f"room-pt-{prefix}"})

        if viz_ws:
            for n, d in g.nodes(data=True):
                if d['type'] == 'ws':
                    ax.scatter(d['center'][0], d['center'][1], color=color_ws, s=20,
                               label=f"WS {prefix}" if f"ws-{prefix}" not in legend_added else "")
                    legend_added.add(f"ws-{prefix}")
                    if 'limits' in d:
                        limit1, limit2 = d['limits']
                        ax.plot([limit1[0], limit2[0]], [limit1[1], limit2[1]],
                                color='black', linewidth=1.0,
                                label=f"WS limits {prefix}" if f"limits-{prefix}" not in legend_added else "")
                        legend_added.add(f"limits-{prefix}")

    plot_graph(g1, is_g1=True)
    plot_graph(g2, is_g1=False)

    # Plot matching lines with partial-match and ID presence checks
    if pred_perm is not None:
        for i in range(pred_perm.shape[0]):  # for each row
            # skip if ground truth has no assignment for this node
            if gt_perm[i].sum().item() == 0:
                continue
            row = pred_perm[i]
            # determine if prediction exists
            if row.sum().item() == 0:
                # missing prediction: draw based on ground truth
                j_gt = gt_perm[i].argmax().item()
                # map indices to node IDs
                id1 = node_names1[i]
                if id1 not in g1.nodes:
                    continue
                if j_gt < len(node_names2):
                    id2 = node_names2[j_gt]
                else:
                    continue
                if id2 not in g2.nodes:
                    continue
                pt1 = g1.nodes[id1]['center']
                pt2 = g2.nodes[id2]['center']
                # skip if match_display filters out missing
                if match_display in {"correct", "wrong"}:
                    continue
                color = 'yellow'
                label = None
                if 'missing' not in legend_added:
                    label = 'Missing match'
                    legend_added.add('missing')
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                        color=color, linestyle='--', alpha=0.6, linewidth=1, label=label)
                continue
            # has prediction: handle correct/wrong
            j = row.argmax().item()
            # map indices to node IDs
            id1 = node_names1[i]
            if id1 not in g1.nodes:
                continue
            if j < len(node_names2):
                id2 = node_names2[j]
            else:
                continue
            if id2 not in g2.nodes:
                continue
            pt1 = g1.nodes[id1]['center']
            pt2 = g2.nodes[id2]['center']
            is_correct = (j < gt_perm.shape[1] and gt_perm[i, j] == 1)
            if match_display == "correct" and not is_correct:
                continue
            if match_display == "wrong" and is_correct:
                continue
            color = 'green' if is_correct else 'red'
            label = None
            if color == 'green' and 'correct' not in legend_added:
                label = 'Correct match'
                legend_added.add('correct')
            elif color == 'red' and 'wrong' not in legend_added:
                label = 'Wrong match'
                legend_added.add('wrong')
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                    color=color, linestyle='-', alpha=0.6, linewidth=1, label=label)

    ax.set_title("Graph Matching: Green = Correct, Red = Wrong")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

def normalize_data_pairs(
    pairs: List[Tuple[Data, Data, torch.Tensor]],
    mean: torch.Tensor,
    std: torch.Tensor
) -> List[Tuple[Data, Data, torch.Tensor]]:
    """
    Normalizza per-feature i tensori x in ciascun Data object all'interno delle tuple.

    Args:
        pairs: Lista di tuple (Data1, Data2, P)
        mean: Tensor di media per-feature (shape: [num_features])
        std: Tensor di deviazione standard per-feature (shape: [num_features])

    Returns:
        Lista di tuple con i Data normalizzati.
    """
    normalized_pairs = []
    for data1, data2, P in pairs:
        data1.x = (data1.x - mean) / (std + 1e-8)
        data2.x = (data2.x - mean) / (std + 1e-8)
        normalized_pairs.append((data1, data2, P))
    return normalized_pairs

def compute_mean_std(pairs: List[Tuple[Data, Data, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcola la media e la deviazione standard per-feature dai Data objects nel training set.

    Args:
        pairs: Lista di tuple (Data1, Data2, P) del training set

    Returns:
        Tuple contenente (mean, std) per-feature
    """
    x_list = []
    for data1, data2, _ in pairs:
        x_list.append(data1.x)
        x_list.append(data2.x)
    x_all = torch.cat(x_list, dim=0)
    mean = x_all.mean(dim=0)
    std = x_all.std(dim=0)
    return mean, std


# %% [markdown]
# ## Logging

# %%
#----------------------------------------
#            LOGGING
#----------------------------------------

# Generic logger setup for any model/dataset
def setup_tb_logger(
    base_dir: str = "runs",
    model_name: str = None,
    dataset_name: str = None,
    experiment_name: str = None
) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter with a structured log directory.

    Args:
        base_dir: root directory for all runs.
        model_name: identifier for the model (e.g. "GATv2", "MyModel").
        dataset_name: identifier for the dataset (e.g. "CIFAR10").
        experiment_name: optional extra tag (e.g. "dropout0.3").

    Returns:
        writer: a SummaryWriter instance logging to runs/... directory.
    """
    parts = []
    if model_name:
        parts.append(model_name)
    if dataset_name:
        parts.append(dataset_name)
    if experiment_name:
        parts.append(experiment_name)
    # timestamp for uniqueness
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir = os.path.join(base_dir, "__".join(parts))
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def log_gradients(
    writer: SummaryWriter,
    model: torch.nn.Module,
    epoch: int,
    prefix: str = "grad_norms"
) -> None:
    """
    Log the L2 norm of gradients of all parameters in the model.

    Args:
        writer: SummaryWriter returned by setup_tb_logger.
        model: the neural network model whose gradients to log.
        epoch: current epoch or step index.
        prefix: prefix for the TensorBoard tags.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            writer.add_scalar(f"{prefix}/{name}", grad_norm, epoch)
            if wandb.run is not None:
                wandb.log({f"{prefix}/{name}": grad_norm}, step=epoch)


def log_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    epoch: int,
    prefix: str = ""
) -> None:
    """
    Log arbitrary metrics (e.g. losses, accuracies) to TensorBoard.

    Args:
        writer: SummaryWriter returned by setup_tb_logger.
        metrics: dict of metric_name -> value.
        epoch: current epoch or step index.
        prefix: optional prefix for tags (e.g. "train", "val").
    """
    log_dict = {}
    for key, value in metrics.items():
        tag = f"{prefix}/{key}" if prefix else key
        writer.add_scalar(tag, value, epoch)
        log_dict[tag] = value
    if wandb.run is not None:
        wandb.log(log_dict, step=epoch)

# %% [markdown]
# ## Training

# %%
#----------------------------------------
#            TRAINING UTILS
#----------------------------------------

# Create the plot
def plot_losses(train_losses, val_losses, output_path=None):
    epochs = list(range(len(train_losses)))
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=epochs, y=train_losses, label="Training Loss")
    sns.lineplot(x=epochs, y=val_losses, label="Validation Loss")

    # Add labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        # Save the plot to the specified path
        plt.savefig(output_path)
        plt.close()

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

### FUNCTIONS WITH BCE
def bce_permutation_loss(P, P_gt, eps: float = 1e-9):
    """Element-wise Binary Cross Entropy loss between prediction and ground truth."""
    assert P.shape == P_gt.shape, f"Shape mismatch: P={P.shape}, P_gt={P_gt.shape}"
    return - (P_gt * torch.log(P + eps) + (1 - P_gt) * torch.log(1 - P + eps)).mean()

def hard_perm_from_scores(P: torch.Tensor) -> torch.Tensor:
    """Convert a soft permutation matrix into a hard assignment (one per column)."""
    hard = torch.zeros_like(P)
    hard[P.argmax(dim=0), torch.arange(P.shape[1], device=P.device)] = 1
    return hard

def permutation_confusion_counts(P_pred_hard: torch.Tensor, P_gt: torch.Tensor) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) counts comparing hard predictions to ground truth."""
    pred = (P_pred_hard > 0.5).to(P_gt.dtype)
    tp = (pred * P_gt).sum().item()
    fp = (pred * (1 - P_gt)).sum().item()
    fn = ((1 - pred) * P_gt).sum().item()
    tn = ((1 - pred) * (1 - P_gt)).sum().item()
    return tp, fp, fn, tn

def permutation_precision_recall_f1(tp: int, fp: int, fn: int, eps: float = 1e-9) -> Tuple[float, float, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

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

def train_epoch_sinkhorn(model, loader, optimizer, writer, epoch, eps: float = 1e-9):
    """
    Trains one epoch of a Sinkhorn-based graph matching model using Binary Cross Entropy (BCE) loss.
    Returns:
        avg_loss (float): average BCE loss per graph.
        avg_acc (float): permutation accuracy over all columns.
        avg_f1 (float): F1 score on hard assignments.
        all_embeddings (list): collected embeddings from the model.
        per_level (dict): per noise-level metric accumulators.
    """
    model.train()
    total_loss = 0.0
    num_graphs = 0
    total_entries = 0
    tp = fp = fn = tn = 0
    all_embeddings = []
    per_level: Dict[int, Dict] = {}
    device = next(model.parameters()).device

    num_batches = len(loader)
    for batch_i, (batch1, batch2, perm_list) in enumerate(loader):
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        perm_list = [p.to(device) for p in perm_list]

        optimizer.zero_grad()
        batch_idx1 = batch1.batch
        batch_idx2 = batch2.batch
        pred_perm_list, batch_embeddings = model(batch1, batch2, perm_list, batch_idx1, batch_idx2)

        noise_levels_batch = [int(x) for x in batch2.noise_level] if hasattr(batch2, 'noise_level') else [None] * len(perm_list)

        # accumulo loss per grafo
        batch_loss = 0.0
        for P, P_gt, level in zip(pred_perm_list, perm_list, noise_levels_batch):
            loss = weighted_bce_loss(P, P_gt)  # assume reduction='mean'
            batch_loss += loss
            total_loss += loss.item()
            num_graphs += 1

            P_hard = hard_perm_from_scores(P)
            tpi, fpi, fni, tni = permutation_confusion_counts(P_hard, P_gt)
            tp += tpi
            fp += fpi
            fn += fni
            tn += tni
            total_entries += (tpi + fpi + fni + tni)

            if level is not None:
                if level not in per_level:
                    per_level[level] = {'acc_list': [], 'prec_list': [], 'rec_list': [], 'f1_list': [], 'loss_list': [], 'num_graphs': 0}
                d = per_level[level]
                n_tot = tpi + fpi + fni + tni
                graph_acc = (tpi + tni) / n_tot if n_tot > 0 else 0.0
                graph_prec, graph_rec, graph_f1 = permutation_precision_recall_f1(tpi, fpi, fni)
                d['acc_list'].append(graph_acc)
                d['prec_list'].append(graph_prec)
                d['rec_list'].append(graph_rec)
                d['f1_list'].append(graph_f1)
                d['loss_list'].append(loss.item())
                d['num_graphs'] += 1

        batch_loss = batch_loss / len(pred_perm_list)  # per logging/grad
        batch_loss.backward()
        # Log gradient norms once per epoch (on the last batch) to avoid same-step
        # collisions and per-batch I/O in the hot loop.
        if batch_i == num_batches - 1:
            log_gradients(writer, model, epoch)
        optimizer.step()

        # Do NOT collect batch_embeddings during training: in train mode they are
        # attached to the autograd graph, so keeping references would retain the
        # full graph of every batch for the whole epoch (large GPU memory leak).
        # The caller discards the train embeddings anyway, so we drop them here.

    avg_loss = total_loss / num_graphs if num_graphs > 0 else 0.0
    avg_acc = (tp + tn) / total_entries if total_entries > 0 else 0.0
    _, _, avg_f1 = permutation_precision_recall_f1(tp, fp, fn, eps=eps)
    return avg_loss, avg_acc, avg_f1, all_embeddings, per_level


def evaluate_sinkhorn(model, loader, eps: float = 1e-9):
    """
    Evaluates a Sinkhorn-based graph matching model using Binary Cross Entropy
    and permutation accuracy.
    Returns:
        avg_acc (float): permutation accuracy over all columns.
        avg_prec (float): precision on hard assignments.
        avg_rec (float): recall on hard assignments.
        avg_f1 (float): F1 score on hard assignments.
        avg_loss (float): average BCE loss per graph.
        all_embeddings (list): collected embeddings from the model.
        per_level (dict): per noise-level metric accumulators.
    """
    model.eval()
    total_entries = 0
    tp = fp = fn = tn = 0
    total_loss = 0.0
    num_graphs = 0
    all_embeddings = []
    per_level: Dict[int, Dict] = {}

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch1, batch2, perm_list in loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            perm_list = [p.to(device) for p in perm_list]

            batch_idx1 = batch1.batch
            batch_idx2 = batch2.batch
            pred_perm_list, batch_embeddings = model(batch1, batch2, perm_list, batch_idx1, batch_idx2)

            noise_levels_batch = [int(x) for x in batch2.noise_level] if hasattr(batch2, 'noise_level') else [None] * len(perm_list)

            for P, P_gt, level in zip(pred_perm_list, perm_list, noise_levels_batch):
                P_hard = hard_perm_from_scores(P)
                tpi, fpi, fni, tni = permutation_confusion_counts(P_hard, P_gt)
                tp += tpi
                fp += fpi
                fn += fni
                tn += tni
                total_entries += (tpi + fpi + fni + tni)

                # loss per grafo
                loss = weighted_bce_loss(P, P_gt)
                total_loss += loss.item()
                num_graphs += 1

                if level is not None:
                    if level not in per_level:
                        per_level[level] = {'acc_list': [], 'prec_list': [], 'rec_list': [], 'f1_list': [], 'loss_list': [], 'num_graphs': 0}
                    d = per_level[level]
                    n_tot = tpi + fpi + fni + tni
                    graph_acc = (tpi + tni) / n_tot if n_tot > 0 else 0.0
                    graph_prec, graph_rec, graph_f1 = permutation_precision_recall_f1(tpi, fpi, fni)
                    d['acc_list'].append(graph_acc)
                    d['prec_list'].append(graph_prec)
                    d['rec_list'].append(graph_rec)
                    d['f1_list'].append(graph_f1)
                    d['loss_list'].append(loss.item())
                    d['num_graphs'] += 1

            # Move embeddings off the GPU as they are collected: batch_embeddings
            # is a list of (h1_b, h2_b) tensors on CUDA. Detaching + copying to CPU
            # frees the GPU copies immediately and keeps eval memory flat even on
            # large validation/test sets.
            all_embeddings.extend(
                (h1_b.detach().cpu(), h2_b.detach().cpu()) for h1_b, h2_b in batch_embeddings
            )

    avg_acc = (tp + tn) / total_entries if total_entries > 0 else 0.0
    avg_loss = total_loss / num_graphs if num_graphs > 0 else 0.0
    avg_prec, avg_rec, avg_f1 = permutation_precision_recall_f1(tp, fp, fn, eps=eps)
    return avg_acc, avg_prec, avg_rec, avg_f1, avg_loss, all_embeddings, per_level


#----------------------------------------
#       REAL VALIDATION SET (Prior=A-graph / Online=S-graph)
#----------------------------------------

class _GraphWrapper:
    """Placeholder used only to unpickle `situational_graphs_wrapper.GraphWrapper`
    objects. We just need the wrapped networkx graph stored in its `.graph` attribute,
    so the original class definition is not required."""
    pass


class _WrapperUnpickler(pickle.Unpickler):
    """Unpickler that maps the (possibly missing) GraphWrapper class to a stub."""
    def find_class(self, module, name):
        if 'situational_graphs_wrapper' in module or name == 'GraphWrapper':
            return _GraphWrapper
        return super().find_class(module, name)


def _load_situational_graph(path: str) -> nx.DiGraph:
    """Load a Prior/Online .pkl and return the underlying networkx DiGraph,
    whether it is stored raw or wrapped inside a GraphWrapper (`.graph`)."""
    with open(path, 'rb') as f:
        obj = _WrapperUnpickler(f).load()
    if isinstance(obj, nx.Graph):
        return obj
    g = getattr(obj, 'graph', None)
    if isinstance(g, nx.Graph):
        return g
    raise TypeError(f"Unexpected object in {path}: {type(obj)}")


def _strip_scene_prefix(node_id) -> str:
    """Strip the 's_'/'a_' scene prefix some ground-truth files use for node ids."""
    s = str(node_id)
    for pre in ('s_', 'a_'):
        if s.startswith(pre):
            return s[len(pre):]
    return s


def _real_nx_to_pyg(graph: nx.DiGraph, center_dim: int) -> Data:
    """
    Convert a real (Prior/Online) DiGraph to a PyG Data object with the same
    feature layout as the training graphs: [type(2), center(center_dim), normal(2), length(1)].
    Original (string) node ids are kept in `node_names` so the ground truth can be applied.
    Rooms may lack 'normal'/'length' -> defaulted to [0,0] / -1 (training convention).
    Center is trimmed/padded to `center_dim` so it matches the model's `in_dim`.
    """
    node_ids = [str(n) for n in graph.nodes()]
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    feats = []
    for n in graph.nodes():
        d = graph.nodes[n]
        center = list(np.asarray(d['center'], dtype=float).ravel())[:center_dim]
        if len(center) < center_dim:
            center = center + [0.0] * (center_dim - len(center))
        normal = list(np.asarray(d.get('normal', [0.0, 0.0]), dtype=float).ravel())[:2]
        if len(normal) < 2:
            normal = normal + [0.0] * (2 - len(normal))
        length = float(d.get('length', -1))
        feats.append(node_type_mapping[d['type']] + center + normal + [length])
    x = torch.tensor(feats, dtype=torch.float32)

    if graph.number_of_edges() > 0:
        edge_index = torch.tensor(
            [[id_map[str(u)], id_map[str(v)]] for u, v in graph.edges()],
            dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.node_names = node_ids
    data.permutation = torch.arange(len(node_ids), dtype=torch.long)
    return data


def deserialize_norm_stats(path: str, filename: str = "norm_stats.pt"):
    """
    Load the per-feature (mean, std) saved next to a dataset by dataset_gen.py.
    Returns None if the file is missing (older datasets) so the caller can warn.
    """
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        return None
    stats = torch.load(full_path, map_location="cpu")
    return stats["mean"], stats["std"]


def build_real_validation_pairs(real_dir: str, in_dim: int, verbose: bool = True,
                                mean: torch.Tensor = None, std: torch.Tensor = None
                                ) -> List[Tuple[Data, Data, torch.Tensor, str]]:
    """
    Build (Data_A, Data_S, P, scene_name) tuples from the real validation folders.

    Each folder must contain:
      - Prior.pkl   -> A-graph  (g1, reference/complete)
      - Online.pkl  -> S-graph  (g2, partial)
      - ground_truth.json with {"rooms": {S_id: A_id, ...}, "ws": [[S_id, A_id], ...]}
        (keys/values may carry an 's_'/'a_' prefix that is stripped when matching).

    P has shape [|A|, |S|], consistent with the training ground truth.
    Ground-truth entries whose ids are not found in the graphs are skipped and reported.

    If `mean`/`std` are given (the training normalization stats), node features are
    normalized the same way as the training data so the model sees them at the same
    scale; otherwise raw features are used (metrics not comparable to training).
    """
    center_dim = in_dim - 5  # type(2) + normal(2) + length(1) are fixed
    pairs: List[Tuple[Data, Data, torch.Tensor, str]] = []

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

        g_a = _load_situational_graph(prior_p)    # A-graph -> g1
        g_s = _load_situational_graph(online_p)   # S-graph -> g2
        with open(gt_p) as f:
            gt = json.load(f)

        data_a = _real_nx_to_pyg(g_a, center_dim)
        data_s = _real_nx_to_pyg(g_s, center_dim)

        # normalize with the training stats so the model sees the same feature scale
        if mean is not None and std is not None:
            data_a.x = (data_a.x - mean) / (std + 1e-8)
            data_s.x = (data_s.x - mean) / (std + 1e-8)

        # id -> index lookups, tolerant to the 's_'/'a_' prefixes
        a_index: Dict[str, int] = {}
        for i, nid in enumerate(data_a.node_names):
            a_index.setdefault(nid, i)
            a_index.setdefault(_strip_scene_prefix(nid), i)
        s_index: Dict[str, int] = {}
        for j, nid in enumerate(data_s.node_names):
            s_index.setdefault(nid, j)
            s_index.setdefault(_strip_scene_prefix(nid), j)

        # ground truth maps  S-node (key) -> A-node (value)  for both rooms and ws
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


def evaluate_real_set(model, real_pairs, writer=None, epoch: int = 0,
                      use_hungarian: bool = True, eps: float = 1e-9, verbose: bool = True):
    """
    Evaluate the model on the real validation scenes (one graph pair per scene),
    computing the same metrics used for the noise levels: WBCE loss, permutation
    accuracy, precision/recall/F1 on the hardened Sinkhorn output, and (optionally)
    the same metrics after the Hungarian assignment.

    Metrics are aggregated over all scenes and also reported per scene; everything
    is logged to TensorBoard / W&B under the 'real' prefix. Returns the aggregate dict.
    """
    if not real_pairs:
        return {}

    model.eval()
    device = next(model.parameters()).device

    tp = fp = fn = tn = 0
    total_entries = 0
    total_loss = 0.0
    h_tp = h_fp = h_fn = h_tn = 0
    per_scene: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for data_a, data_s, P_gt, name in real_pairs:
            d1 = data_a.to(device)
            d2 = data_s.to(device)
            P_gt = P_gt.to(device)
            bidx1 = torch.zeros(d1.num_nodes, dtype=torch.long, device=device)
            bidx2 = torch.zeros(d2.num_nodes, dtype=torch.long, device=device)

            pred_list, _ = model(d1, d2, perm_list=None,
                                  batch_idx1=bidx1, batch_idx2=bidx2, inference=True)
            S = pred_list[0]  # soft Sinkhorn assignment [n1, n2]

            loss = weighted_bce_loss(S, P_gt)
            total_loss += loss.item()

            P_hard = hard_perm_from_scores(S)
            tpi, fpi, fni, tni = permutation_confusion_counts(P_hard, P_gt)
            tp += tpi; fp += fpi; fn += fni; tn += tni
            total_entries += (tpi + fpi + fni + tni)

            n_tot = tpi + fpi + fni + tni
            scene = {'loss': loss.item()}
            scene['acc'] = (tpi + tni) / n_tot if n_tot > 0 else 0.0
            _, _, scene['f1'] = permutation_precision_recall_f1(tpi, fpi, fni)

            if use_hungarian:
                sim = S.unsqueeze(0)
                n1 = torch.tensor([S.shape[0]], dtype=torch.int32, device=device)
                n2 = torch.tensor([S.shape[1]], dtype=torch.int32, device=device)
                H = pygmtools.hungarian(sim, n1=n1, n2=n2).squeeze(0).to(P_gt.device)
                htpi, hfpi, hfni, htni = permutation_confusion_counts(H, P_gt)
                h_tp += htpi; h_fp += hfpi; h_fn += hfni; h_tn += htni
                _, _, scene['hungarian_f1'] = permutation_precision_recall_f1(htpi, hfpi, hfni)

            per_scene[name] = scene

    n_scenes = len(real_pairs)
    avg_loss = total_loss / n_scenes
    avg_acc = (tp + tn) / total_entries if total_entries > 0 else 0.0
    avg_prec, avg_rec, avg_f1 = permutation_precision_recall_f1(tp, fp, fn, eps=eps)

    metrics = {'loss': avg_loss, 'acc': avg_acc,
               'precision': avg_prec, 'recall': avg_rec, 'f1': avg_f1}
    if use_hungarian:
        h_acc = (h_tp + h_tn) / total_entries if total_entries > 0 else 0.0
        h_prec, h_rec, h_f1 = permutation_precision_recall_f1(h_tp, h_fp, h_fn, eps=eps)
        metrics.update({'hungarian_acc': h_acc, 'hungarian_precision': h_prec,
                        'hungarian_recall': h_rec, 'hungarian_f1': h_f1})

    if writer is not None:
        log_metrics(writer, metrics, epoch, prefix="real")
        for name, scene in per_scene.items():
            log_metrics(writer, scene, epoch, prefix=f"real/{name}")

    if verbose:
        line = f"  [Epoch {epoch:03}] REAL val | Loss {avg_loss:.4f} | Acc {avg_acc:.4f} | F1 {avg_f1:.4f}"
        if use_hungarian:
            line += f" | Hungarian F1 {metrics['hungarian_f1']:.4f}"
        print(line)
        for name in sorted(per_scene.keys()):
            s = per_scene[name]
            extra = f" | Hung F1 {s['hungarian_f1']:.4f}" if 'hungarian_f1' in s else ""
            print(f"      {name:24} Loss {s['loss']:.4f} | Acc {s['acc']:.4f} | F1 {s['f1']:.4f}{extra}")

    return metrics


def predict_matching_matrix(model, data1, data2, use_hungarian: bool = True):
    """
    Produces a matching matrix between data1 and data2.
    If use_hungarian=True, applies the Hungarian algorithm to the similarity scores.
    Otherwise returns the raw similarity matrix.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        data1 = data1.to(device)
        data2 = data2.to(device)
        batch_idx1 = torch.zeros(data1.num_nodes, dtype=torch.long, device=device)
        batch_idx2 = torch.zeros(data2.num_nodes, dtype=torch.long, device=device)

        sim_matrix_list, _ = model(data1, data2, batch_idx1=batch_idx1, batch_idx2=batch_idx2, inference=True)
        sim = sim_matrix_list[0].unsqueeze(0)  # [1, N1, N2]

        n1 = torch.tensor([sim.shape[1]], dtype=torch.int32, device=device)
        n2 = torch.tensor([sim.shape[2]], dtype=torch.int32, device=device)

        if use_hungarian:
            # returns a hard assignment matrix [N1, N2]
            return pygmtools.hungarian(sim, n1=n1, n2=n2).squeeze(0)
        else:
            # return soft scores [N1, N2]
            return sim.squeeze(0)


def train_loop(model, optimizer, train_loader, val_loader, num_epochs, writer,
               best_model_path='checkpoint.pt', final_model_path='final_model.pt',
               patience=10, resume=False, unfreeze_epoch=None, log_level_every=100,
               real_pairs=None, eval_real_every=10):
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    epoch = start_epoch = 0

    train_losses = []
    val_losses = []
    val_embeddings_history = []

    # Resume from checkpoint if requested
    if resume and os.path.exists(best_model_path):
        print(f"Loading checkpoint from {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        print(f"Resumed from epoch {start_epoch}")
        epoch = start_epoch

    print("Starting training...")

    try:
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_loss, train_acc, train_f1, _, train_per_level = train_epoch_sinkhorn(model, train_loader, optimizer, writer, epoch)
            # Evaluate
            val_acc, val_prec, val_rec, val_f1, val_loss, val_embeddings, val_per_level = evaluate_sinkhorn(model, val_loader)

            log_metrics(writer, {"loss": train_loss, "acc": train_acc, "f1": train_f1}, epoch, prefix="train")
            log_metrics(writer, {"loss": val_loss, "acc": val_acc, "precision": val_prec, "recall": val_rec, "f1": val_f1}, epoch, prefix="val")

            for level, d in val_per_level.items():
                if d['num_graphs'] > 0:
                    lvl_acc  = float(np.mean(d['acc_list']))
                    lvl_f1   = float(np.mean(d['f1_list']))
                    lvl_loss = float(np.mean(d['loss_list']))
                    log_metrics(writer, {"acc": lvl_acc, "f1": lvl_f1, "loss": lvl_loss}, epoch, prefix=f"val/level_{level}")

            for level, d in train_per_level.items():
                if d['num_graphs'] > 0:
                    lvl_acc  = float(np.mean(d['acc_list']))
                    lvl_f1   = float(np.mean(d['f1_list']))
                    lvl_loss = float(np.mean(d['loss_list']))
                    log_metrics(writer, {"acc": lvl_acc, "f1": lvl_f1, "loss": lvl_loss}, epoch, prefix=f"train/level_{level}")

            # Evaluate on the separate REAL validation set every `eval_real_every` epochs
            if real_pairs and eval_real_every > 0 and epoch % eval_real_every == 0:
                evaluate_real_set(model, real_pairs, writer=writer, epoch=epoch, use_hungarian=True)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Do NOT accumulate val_embeddings across epochs: these are GPU
            # tensors for the WHOLE validation set and were never used after
            # training, so they piled up on the GPU every epoch -> CUDA OOM
            # around epoch 35. If you need them, recompute for a single epoch
            # with evaluate_sinkhorn and move them to CPU (.detach().cpu()).

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch
                }, best_model_path)
                print(f"[Epoch {epoch}] Saved new best model.")
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:03} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

            if log_level_every > 0 and epoch % log_level_every == 0 and val_per_level:
                print(f"  [Epoch {epoch:03}] Val metrics per noise level (mean over graphs):")
                for level in sorted(val_per_level.keys()):
                    d = val_per_level[level]
                    if d['num_graphs'] > 0:
                        lvl_acc  = float(np.mean(d['acc_list']))
                        lvl_f1   = float(np.mean(d['f1_list']))
                        lvl_loss = float(np.mean(d['loss_list']))
                        print(f"    Level {level:>3}%: Loss {lvl_loss:.4f} | Acc {lvl_acc:.4f} | F1 {lvl_f1:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best was {best_epoch}.")
                break

            # if epoch == unfreeze_epoch:
            #     print(f"Unlocking GNN for progressive fine-tuning at epoch {epoch}")
            #     for param in model.gnn.parameters():
            #         param.requires_grad = True

            #     # Rebuild optimizer with lower LR for GNN
            #     optimizer = torch.optim.Adam([
            #         {"params": model.mlp.parameters(), "lr": 5e-5},
            #         {"params": model.gnn.parameters(), "lr": 5e-5},  # lower learning rate
            #         {"params": model.inst_norm.parameters(), "lr": 1e-4},
            #     ], weight_decay=1e-4)

    except KeyboardInterrupt:
        print("Training interrupted manually (Ctrl+C).")

    log_metrics(writer, {"best_val_loss": best_val_loss, "best_epoch": best_epoch}, epoch, prefix="best")
    writer.close()

    # Save final model
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'best_epoch': best_epoch
    }, final_model_path)
    print("Final model saved.")

    return train_losses, val_losses, val_embeddings_history

# %% [markdown]
# ## Models

# %%
#----------------------------------------
#            MODELS
#----------------------------------------

###     PARTIAL GRAPH MATCHING MODEL with MLP
class MatchingModel_MLPGATv2SinkhornWBCE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, sinkhorn_max_iter: int = 10, sinkhorn_tau: float = 1.0,
                 attention_dropout: float = 0.1, dropout_emb: float = 0.1, num_layers: int = 2, heads: int = 1):
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
        self.gnn = nn.ModuleList()
        dims = [hidden_dim] * num_layers + [out_dim]
        for i in range(num_layers):
            # Always average the heads so the feature-dim stays dims[i+1]
            self.gnn.append(
                GATv2Conv(dims[i], dims[i+1],
                            heads=heads, concat=False,
                            dropout=attention_dropout)
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

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.gnn):
            x = conv(x, edge_index)
            if i < len(self.gnn) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def forward(self, batch1, batch2, perm_list=None, batch_idx1=None, batch_idx2=None, inference=False):
        device = next(self.parameters()).device
        x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
        x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
        perm_list = [p.to(device) for p in perm_list] if perm_list is not None else None

        batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
        batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)
        
        # Apply MLP before GNN
        h1 = self.mlp(x1)
        h2 = self.mlp(x2)
        h1 = self.encode(h1, edge1)
        h2 = self.encode(h2, edge2)

        B = batch_idx1.max().item() + 1
        perm_pred_list = []
        all_embeddings = []

        for b in range(B):
            h1_b = h1[batch_idx1 == b]   # [n1, d]
            h2_b = h2[batch_idx2 == b]   # [n2, d]
            N1, N2 = h1_b.size(0), h2_b.size(0)

            # affinity matrix + normalization + sinkhorn
            sim = torch.matmul(h1_b, h2_b.T) # [n1, n2]
            sim_batched = sim.unsqueeze(0).unsqueeze(1) # [1,1,n1,n2]
            sim_normed = self.inst_norm(sim_batched).squeeze(1) # [1,n1,n2]

            # g1 -> A-graph 
            # g2 -> S-graph (partial)
            transposed = N1 > N2

            if transposed:
                # traspose to use dummy_row
                sim_input = sim_normed.transpose(-2, -1)   # [1, n2, n1]
                nr = torch.tensor([N2], dtype=torch.long, device=device)
                nc = torch.tensor([N1], dtype=torch.long, device=device)
            else:
                sim_input = sim_normed                     # [1, n1, n2]
                nr = torch.tensor([N1], dtype=torch.long, device=device)
                nc = torch.tensor([N2], dtype=torch.long, device=device)

            S = pygmtools.sinkhorn(
                sim_input,
                n1=nr, n2=nc,
                dummy_row=(N1 != N2),
                max_iter=self.sinkhorn_max_iter,
                tau=self.sinkhorn_tau
            )

            if transposed:
                S = S.transpose(-2, -1)   # rollback to [1, n1, n2]

            perm_pred_list.append(S.squeeze(0))  # [n1, n2]
            all_embeddings.append((h1_b, h2_b))

        return perm_pred_list, all_embeddings

def objective_pgm(trial, train_dataset, val_dataset, path):
    lr           = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    out_dim      = trial.suggest_categorical("out_dim", [16, 32, 64])
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout_emb  = trial.suggest_uniform("dropout_emb", 0.0, 0.6)
    attn_dropout = trial.suggest_uniform("attn_dropout", 0.0, 0.6)
    sinkhorn_max_iter = trial.suggest_int("sinkhorn_max_iter", 10, 100)
    sinkhorn_tau = trial.suggest_uniform("sinkhorn_tau", 0.01, 1.0)
    num_layers   = trial.suggest_int("num_layers",       1, 3)
    heads        = trial.suggest_int("heads",           1,   4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)


    # Flexible model for partial matching
    class MatchingModel_MLPGATv2Sinkhorn_OPT(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, sinkhorn_max_iter, sinkhorn_tau,
                    attention_dropout, dropout_emb, num_layers, heads):
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
            self.gnn = nn.ModuleList()
            dims = [hidden_dim] * num_layers + [out_dim]
            for i in range(num_layers):
                # Always average the heads so the feature-dim stays dims[i+1]
                self.gnn.append(
                    GATv2Conv(dims[i], dims[i+1],
                              heads=heads, concat=False,
                              dropout=attention_dropout)
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
            
            # Apply MLP before GNN
            h1 = self.mlp(x1)
            h2 = self.mlp(x2)
            h1 = self.encode(h1, edge1)
            h2 = self.encode(h2, edge2)

            B = batch_idx1.max().item() + 1
            loss = 0.0
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

                loss += weighted_bce_loss(S.squeeze(0), perm_list[b])

            return loss / B

    model = MatchingModel_MLPGATv2Sinkhorn_OPT(
        in_dim=train_dataset[0][0].x.size(1),
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        sinkhorn_max_iter=sinkhorn_max_iter,
        sinkhorn_tau=sinkhorn_tau,
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

# %% [markdown]
# ## XAI

# %%
#----------------------------------------
#            XAI UTILS
#----------------------------------------
def plt2arr(fig):
    """
    Converts a matplotlib figure to a NumPy RGB array.
    Ensures the canvas is drawn before reading pixel data.
    """
    # Attach a canvas if not already present
    if fig.canvas is None or not isinstance(fig.canvas, FigureCanvas):
        FigureCanvas(fig)

    # Force the figure to render
    fig.canvas.draw()

    # Get the image from the buffer
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)

    # Remove the alpha channel (RGBA -> RGB)
    img_rgb = img[..., :3].copy()

    return img_rgb

def get_node_type_labels(h):
    if h.shape[1] < 2:
        raise ValueError("The embedding must have at least two dimensions to distinguish types.")
    return torch.argmax(h[:, :2], dim=1)  # 0 = room, 1 = ws

def visualize(h, node_type_labels, graph_labels, epoch, node_type_filter: Optional[Literal["room", "ws", "all"]] = "all"):
    fig, ax = plt.subplots(figsize=(10, 10), frameon=False)
    fig.suptitle(f'Epoch index = {epoch}')
    z = TSNE(2, random_state=42, init='pca').fit_transform(h.cpu().numpy())
    ax.set_xticks([]); ax.set_yticks([])

    appearance = {
        (0, 0): ('tab:blue', 'o'),    # G1 - room
        (0, 1): ('tab:green', 's'),   # G1 - ws
        (1, 0): ('tab:orange', '^'),  # G2 - room
        (1, 1): ('tab:red', 'D'),     # G2 - ws
    }

    for g in [0, 1]:
        for t in [0, 1]:
            if node_type_filter == "room" and t != 0:
                continue
            if node_type_filter == "ws" and t != 1:
                continue
            mask = ((graph_labels == g) & (node_type_labels == t)).cpu().numpy()
            if np.any(mask):
                z_sub = z[mask]
                color, marker = appearance[(g, t)]
                label = f"{'G1' if g==0 else 'G2'} - {'room' if t==0 else 'ws'}"
                ax.scatter(z_sub[:, 0], z_sub[:, 1],
                           s=50, c=color, marker=marker,
                           edgecolors='k', linewidths=0.5,
                           alpha=0.4, label=label)

    ax.legend(loc='best', frameon=True)
    fig.canvas.draw()
    arr = plt2arr(fig)
    plt.close(fig)
    return arr

def create_embedding_gif_stride(history, output_path, embedding_type, pair=0, fps=1, step=5, node_type_filter: Optional[Literal["room", "ws", "all"]] = "all"):
    plt.close('all')
    images = []

    for epoch in range(0, len(history), step):
        h1, h2 = history[epoch][pair]
        h = torch.cat([h1, h2], dim=0)

        node_types = embedding_type
        graph_labels = torch.cat([
            torch.zeros(h1.size(0), dtype=torch.long),
            torch.ones(h2.size(0), dtype=torch.long)
        ], dim=0)

        graph_labels = graph_labels.to(device)
        node_types = node_types.to(device)
        
        images.append(visualize(h, node_types, graph_labels, epoch, node_type_filter=node_type_filter))

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_gif(output_path, fps=fps)
    print(f"GIF saved at: {output_path}")

def visualize_initial_embeddings(h1, h2, output_path, node_type_filter: Optional[Literal["room", "ws", "all"]] = "all"):
    h = torch.cat([h1, h2], dim=0)
    node_types = get_node_type_labels(h)
    graph_labels = torch.cat([
        torch.zeros(h1.size(0), dtype=torch.long),
        torch.ones(h2.size(0), dtype=torch.long)
    ], dim=0)

    graph_labels = graph_labels.to(device)
    node_types = node_types.to(device)

    fig, ax = plt.subplots(figsize=(10, 10), frameon=False)
    z = TSNE(2, random_state=42, init='pca').fit_transform(h.cpu().numpy())
    ax.set_xticks([]); ax.set_yticks([])

    appearance = {
        (0, 0): ('tab:blue', 'o'),
        (0, 1): ('tab:green', 's'),
        (1, 0): ('tab:orange', '^'),
        (1, 1): ('tab:red', 'D'),
    }

    for g in [0, 1]:
        for t in [0, 1]:
            if node_type_filter == "room" and t != 0:
                continue
            if node_type_filter == "ws" and t != 1:
                continue
            mask = ((graph_labels == g) & (node_types == t)).cpu().numpy()
            if np.any(mask):
                z_sub = z[mask]
                color, marker = appearance[(g, t)]
                label = f"{'G1' if g==0 else 'G2'} - {'room' if t==0 else 'ws'}"
                ax.scatter(z_sub[:, 0], z_sub[:, 1],
                           s=50, c=color, marker=marker,
                           edgecolors='k', linewidths=0.5,
                           alpha=0.4, label=label)

    ax.legend(loc='best', frameon=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    return node_types

## Partial graph matching training (parameterized by experiment)
# The experiment id selects the dataset AND the models/output subfolder, aligned with
# optimization.py (which reads/writes models/partial_graph_matching/<experiment>).
#   python pgm_training_ws_room_inc_WBCE_scratch_new.py fully_glob_95
#   python pgm_training_ws_room_inc_WBCE_scratch_new.py adj_no_glob_65 --no-resume
import argparse

_parser = argparse.ArgumentParser(
    description="From-scratch WBCE training for partial graph matching."
)
_parser.add_argument(
    "experiment", nargs="?", default="ws_room_dropout_noise_inc",
    help="Experiment id: dataset/models subfolder under partial_graph_matching "
         "(e.g. adj_no_glob_65, fully_glob_95, ws_room_dropout_noise_inc).",
)
_parser.add_argument(
    "--reference", default=None,
    help="Reference (A-graph) variant under graph_matching used for original.pkl. "
         "Default: inferred (adj_*->adj, fully_*->fully, otherwise equal).",
)
_parser.add_argument("--epochs", type=int, default=2000, help="Max training epochs.")
_parser.add_argument("--patience", type=int, default=150, help="Early-stopping patience.")
_parser.add_argument("--no-resume", action="store_true",
                     help="Ignore any existing checkpoint and train from scratch.")
_parser.add_argument("--real-dir", default=None,
                     help="Real validation folder under GNN. Default: inferred from the "
                          "reference variant (adj->real_adj, fully->real_fully, else real).")
# parse_known_args so the script still runs inside notebooks / with extra args
_cli_args, _ = _parser.parse_known_args()

EXPERIMENT = _cli_args.experiment
REFERENCE = _cli_args.reference or (
    "adj" if EXPERIMENT.startswith("adj")
    else "fully" if EXPERIMENT.startswith("fully")
    else "equal"
)

#load preprocessed dataset
gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", REFERENCE)
gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", EXPERIMENT)
models_path = os.path.join(GNN_PATH, 'models', "partial_graph_matching", EXPERIMENT)
os.makedirs(models_path, exist_ok=True)
print(f"Experiment: {EXPERIMENT} | reference (A-graph) variant: {REFERENCE}")

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
d1,d2,gt = train_list[0]
print(d1)
print(d2)
print(gt)
print(gt.shape)
plot_two_graphs_with_matching([d1,d2],gt_perm=gt,original_graphs=original_graphs,noise_graphs=noise_graphs,path=os.path.join(models_path, "train.png"))
train_dataset = GraphMatchingDataset(train_list)
val_dataset = GraphMatchingDataset(val_list)
test_dataset = GraphMatchingDataset(test_list)

# Percorsi per caricare i modelli
best_val_model_path = os.path.join(models_path, 'best_val_model.pt')
final_model_path = os.path.join(models_path, 'final_model.pt')

# Load best hyperparameters from the experiment's Optuna study.
# We read best_trial_results.json (written by optimization.py) instead of
# unpickling study.pkl: the pickled study also serializes the BoTorchSampler,
# which references optimization.py's `logNEI_candidates_func`. That reference
# can't be resolved from THIS module's __main__, so pickle.load would raise
# AttributeError. The JSON holds exactly {"val_loss":..., "params":{...}} and
# has no such dependency.
best_trial_path = os.path.join(models_path, 'best_trial_results.json')

with open(best_trial_path, 'r') as f:
    best_trial_results = json.load(f)

best_params = best_trial_results['params']
best_val_loss = best_trial_results['val_loss']

in_dim = train_dataset[0][0].x.size(1)
learning_rate = best_params['lr']
weight_decay = best_params['weight_decay']
hidden_dim = best_params['hidden_dim']
out_dim = best_params['out_dim']
batch_size = best_params['batch_size']
dropout_emb = best_params['dropout_emb']
attn_dropout = best_params['attn_dropout']
num_layers = best_params['num_layers']
heads = best_params['heads']
sinkhorn_max_iter = best_params['sinkhorn_max_iter']
sinkhorn_tau = best_params['sinkhorn_tau']

print(f"Best hyperparameters: {best_params}")
print(f"Best trial value (validation loss): {best_val_loss:.4f}")

# Loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# Modello e ottimizzatore
model = MatchingModel_MLPGATv2SinkhornWBCE(
    in_dim=in_dim,
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    attention_dropout=attn_dropout,
    dropout_emb=dropout_emb,
    num_layers=num_layers,
    heads=heads,
    sinkhorn_max_iter=sinkhorn_max_iter,
    sinkhorn_tau=sinkhorn_tau
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

model.to(device)

# Logger TensorBoard
writer = setup_tb_logger(
    base_dir="tb_logs",
    model_name=model._get_name(),
    dataset_name=f"PGM_{EXPERIMENT}_WBCE",
    experiment_name="scratch_new"
)

# Logger W&B
wandb.init(
    project="graph-matching",
    entity="arg-graph-matching",
    name=f"scratch_new_{EXPERIMENT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    group=EXPERIMENT,
    tags=[EXPERIMENT, "scratch_new"],
    config={
        "training_type": "scratch_new",
        "experiment": EXPERIMENT,
        "reference": REFERENCE,
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "batch_size": batch_size,
        "dropout_emb": dropout_emb,
        "attn_dropout": attn_dropout,
        "num_layers": num_layers,
        "heads": heads,
        "sinkhorn_max_iter": sinkhorn_max_iter,
        "sinkhorn_tau": sinkhorn_tau,
    }
)

#model summary
print(model)
print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Real (out-of-distribution) validation set: Prior=A-graph (g1), Online=S-graph (g2)
# Auto-select the connectivity-matched real folder (adj->real_adj, fully->real_fully, else real)
_real_variant = {"adj": "real_adj", "fully": "real_fully"}.get(REFERENCE, "real")
real_val_dir = _cli_args.real_dir or os.path.join(GNN_PATH, _real_variant)

# Normalize the real set with the SAME train stats used for this experiment's dataset.
_norm_stats = deserialize_norm_stats(gm_local_preprocessed_path)
if _norm_stats is None:
    _real_mean = _real_std = None
    print(f"\n[WARNING] norm_stats.pt not found in {gm_local_preprocessed_path} -> the real "
          f"validation set will NOT be normalized; its metrics are not comparable to training. "
          f"Regenerate the dataset with dataset_gen.py to produce norm_stats.pt.")
else:
    _real_mean, _real_std = _norm_stats

print(f"\nLoading real validation set from {real_val_dir} "
      f"(normalized={_norm_stats is not None})...")
real_val_pairs = build_real_validation_pairs(real_val_dir, in_dim,
                                             mean=_real_mean, std=_real_std)

print("\n" + "=" * 60)
print("TRAINING TYPE: FROM SCRATCH (Optuna hyperparameters - NEW)")
print(f"  Experiment            : {EXPERIMENT}")
print(f"  Weight initialization : random")
print(f"  Learning rate         : {learning_rate:.2e}  (from Optuna study)")
print(f"  Dataset               : preprocessed/partial_graph_matching/{EXPERIMENT}")
print(f"  Epochs / Patience     : {_cli_args.epochs} / {_cli_args.patience}")
print(f"  Resume from checkpoint: {not _cli_args.no_resume}")
print("=" * 60 + "\n")

train_losses, val_losses, val_embeddings_history = train_loop(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=_cli_args.epochs,
    writer=writer,
    best_model_path=best_val_model_path,
    final_model_path=final_model_path,
    patience=_cli_args.patience,
    resume=not _cli_args.no_resume,
    unfreeze_epoch=5,
    log_level_every=10,
    real_pairs=real_val_pairs,
    eval_real_every=10
)

plot_losses(train_losses, val_losses, os.path.join(models_path, 'losses.png'))

# Always evaluate the best validation checkpoint.
best_checkpoint = torch.load(best_val_model_path, map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.to(device)

# Evaluate on the test set
test_acc, test_prec, test_rec, test_f1, test_loss, test_embeddings, test_per_level = evaluate_sinkhorn(model, test_loader)
print(
    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Precision: {test_prec:.4f} "
    f"| Test Recall: {test_rec:.4f} | Test F1: {test_f1:.4f}"
)

print("\n--- Test metrics per noise level (Sinkhorn, mean over graphs) ---")
for level in sorted(test_per_level.keys()):
    d = test_per_level[level]
    if d['num_graphs'] > 0:
        lvl_acc  = float(np.mean(d['acc_list']))
        lvl_prec = float(np.mean(d['prec_list']))
        lvl_rec  = float(np.mean(d['rec_list']))
        lvl_f1   = float(np.mean(d['f1_list']))
        lvl_loss = float(np.mean(d['loss_list']))
        print(
            f"  Level {level:>3}%: Loss {lvl_loss:.4f} | Acc {lvl_acc:.4f} | "
            f"Prec {lvl_prec:.4f} | Rec {lvl_rec:.4f} | F1 {lvl_f1:.4f}"
        )
print("-----------------------------------------------\n")

inference_times = []
# use the model to predict the matching on a test graph
total_entries = 0
tp = fp = fn = tn = 0
per_level_hungarian: Dict[int, Dict] = {}

for i, (g1_out, g2_perm, gt_perm) in enumerate(test_list):
    start_time = time.time()
    result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)
    end_time = time.time()
    inference_times.append(end_time - start_time)
    errors = (result != gt_perm.to(result.device)).sum().item()
    if errors > 0:
        print(f"Graph {i}: Errors found: {errors}")

    # Metrics calculation after hungarian
    result = result.to(gt_perm.device)
    tpi, fpi, fni, tni = permutation_confusion_counts(result, gt_perm)
    tp += tpi
    fp += fpi
    fn += fni
    tn += tni
    total_entries += (tpi + fpi + fni + tni)

    level = getattr(g2_perm, 'noise_level', None)
    if level is not None:
        if level not in per_level_hungarian:
            per_level_hungarian[level] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'total_entries': 0}
        d = per_level_hungarian[level]
        d['tp'] += tpi; d['fp'] += fpi; d['fn'] += fni; d['tn'] += tni
        d['total_entries'] += tpi + fpi + fni + tni

accuracy = (tp + tn) / total_entries if total_entries > 0 else 0.0
precision, recall, f1 = permutation_precision_recall_f1(tp, fp, fn)
print(
    f"Test Metrics (after Hungarian): Acc {accuracy:.4f} | Precision {precision:.4f} "
    f"| Recall {recall:.4f} | F1 {f1:.4f}"
)

print("\n--- Test metrics per noise level (after Hungarian) ---")
for level in sorted(per_level_hungarian.keys()):
    d = per_level_hungarian[level]
    if d['total_entries'] > 0:
        lvl_acc = (d['tp'] + d['tn']) / d['total_entries']
        lvl_prec, lvl_rec, lvl_f1 = permutation_precision_recall_f1(d['tp'], d['fp'], d['fn'])
        print(
            f"  Level {level:>3}%: Acc {lvl_acc:.4f} | Prec {lvl_prec:.4f} | Rec {lvl_rec:.4f} | F1 {lvl_f1:.4f}"
        )
print("------------------------------------------------------\n")

mean_inference_time = np.mean(inference_times)
std_inference_time = np.std(inference_times)
print(f"Inference time: {mean_inference_time:.6f} seconds (mean) ± {std_inference_time:.6f} seconds (std)")
g1_out, g2_perm, gt_perm = test_list[0]
result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)

plot_two_graphs_with_matching(
    [g1_out, g2_perm],
    gt_perm=gt_perm,
    pred_perm=result,
    original_graphs=original_graphs,
    noise_graphs=noise_graphs,
    viz_rooms=True,
    viz_ws=True,
    match_display="wrong",
    path=os.path.join(models_path, "test.png")
)

if wandb.run is not None:
    wandb.log({
        "test/loss": test_loss, "test/acc": test_acc,
        "test/precision": test_prec, "test/recall": test_rec, "test/f1": test_f1,
        "test/hungarian_acc": accuracy, "test/hungarian_precision": precision,
        "test/hungarian_recall": recall, "test/hungarian_f1": f1,
    })
    for level in sorted(test_per_level.keys()):
        d = test_per_level[level]
        if d['num_graphs'] > 0:
            wandb.log({
                f"test/level_{level}/acc": float(np.mean(d['acc_list'])),
                f"test/level_{level}/f1": float(np.mean(d['f1_list'])),
                f"test/level_{level}/loss": float(np.mean(d['loss_list'])),
            })
    for level in sorted(per_level_hungarian.keys()):
        d = per_level_hungarian[level]
        if d['total_entries'] > 0:
            lvl_acc = (d['tp'] + d['tn']) / d['total_entries']
            lvl_prec, lvl_rec, lvl_f1 = permutation_precision_recall_f1(d['tp'], d['fp'], d['fn'])
            wandb.log({
                f"test/level_{level}/hungarian_acc": lvl_acc,
                f"test/level_{level}/hungarian_f1": lvl_f1,
            })
    wandb.finish()

