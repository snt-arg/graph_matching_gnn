# %% [markdown]
# <a href="https://colab.research.google.com/github/mgiorgi13/GNN_Notebooks/blob/main/Graph%20Matching.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Setup

# %%
#Colab
# from google.colab import drive
# drive.mount('/content/drive')
# GNN_PATH = '/content/drive/MyDrive/Colab Notebooks/GNN/'

#Local
GNN_PATH = './GNN/'
import os
if not os.path.exists(GNN_PATH):
    os.makedirs(GNN_PATH)

# %%
# Install packages
import subprocess
#subprocess.run(["pip", "install", "torch", "torch-geometric", "scikit-learn", "pandas", "shapely", "seaborn", "pygmtools", "moviepy", "matplotlib", "numpy"], check=True)
#check if pygmtools is installed
#try:
#   import pygmtools
#except ImportError:#pygmtools library
#    subprocess.run(["pip", "install", "git+https://github.com/Thinklab-SJTU/pygmtools.git"], check=True)

# Check pytorch version and make sure you use a GPU Kernel
import torch
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)

# # Check CUDA version
# subprocess.run(["nvcc", "--version"], check=True)

# # Check GPU
# subprocess.run(["nvidia-smi"], check=True)

#set device as cuda if available to load model and data on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# ─── Standard library ──────────────────────────────────────────────────────────
import copy
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple

# ─── Third-party libraries ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from shapely.affinity import translate
from shapely.geometry import Polygon
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, GCNConv

# from moviepy.editor import ImageSequenceClip

import pygmtools
pygmtools.BACKEND = 'pytorch'

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

def plot_a_graph(graphs_list, viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=False, viz_walls=True):
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
    plt.show()

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
            graph.nodes[n]['center'] +
            graph.nodes[n]['normal'] +
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


def split_graphs(graphs, seed: int = 42):
    """
    Split graphs into 70% train, 15% validation, 15% test.
    """
    train, temp = train_test_split(graphs, test_size=0.3, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    return train, val, test


def generate_matching_pair_as_data(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    pairs_list: List[Tuple[Data, Data, torch.Tensor]]
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


def plot_two_graphs_with_matching(graphs_list, gt_perm, original_graphs, noise_graphs=None, pred_perm=None,
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
    plt.show()

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


#----------------------------------------
#            TRAINING UTILS
#----------------------------------------

# Create the plot
def plot_losses(train_losses, val_losses, output_path):
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

    # Save the plot to the specified path
    plt.savefig(output_path)
    plt.close()

# def apply_inverse_permutation_to_graph(g2: nx.DiGraph, inv_perm: torch.Tensor) -> nx.DiGraph:
#     """
#     Riordina i nodi del grafo g2 secondo la permutazione inversa.
#     """
#     reordered = nx.DiGraph()
#     mapping = {}

#     for i, original_idx in enumerate(inv_perm.tolist()):
#         node_data = g2.nodes[original_idx]
#         reordered.add_node(i, **node_data)
#         mapping[original_idx] = i

#     # TODO: edge copy is wrong, but it doesn't matter for the moment
#     for u, v, data in g2.edges(data=True):
#         if u in mapping and v in mapping:
#             reordered.add_edge(mapping[u], mapping[v], **data)

#     reordered.graph.update(g2.graph)  # copy graph-level attributes
#     return reordered

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


def train_epoch_sinkhorn(model, loader, optimizer):
    model.train()
    total_loss = 0
    all_embeddings = []

    device = next(model.parameters()).device

    for batch1, batch2, perm_list in loader:
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        perm_list = [p.to(device) for p in perm_list]

        optimizer.zero_grad()
        batch_idx1 = batch1.batch
        batch_idx2 = batch2.batch

        pred_perm_list, batch_embeddings = model(batch1, batch2, batch_idx1, batch_idx2)

        loss = sum(F.binary_cross_entropy(pred, target) for pred, target in zip(pred_perm_list, perm_list))
        loss /= len(perm_list)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_embeddings.extend(batch_embeddings)

    avg_loss = total_loss / len(loader)
    return avg_loss, all_embeddings


def evaluate_sinkhorn(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_embeddings = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch1, batch2, perm_list in loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            perm_list = [p.to(device) for p in perm_list]

            batch_idx1 = batch1.batch
            batch_idx2 = batch2.batch

            pred_perm_list, batch_embeddings = model(batch1, batch2, batch_idx1, batch_idx2)

            for pred, target in zip(pred_perm_list, perm_list):
                pred_indices = pred.argmax(dim=1)
                target_indices = target.argmax(dim=1)

                correct += (pred_indices == target_indices).sum().item()
                total += pred.size(0)

            loss = sum(F.binary_cross_entropy(pred, target) for pred, target in zip(pred_perm_list, perm_list))
            total_loss += loss.item()
            all_embeddings.extend(batch_embeddings)

    avg_acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader)
    return avg_acc, avg_loss, all_embeddings


def predict_matching_matrix(model, data1, data2, use_hungarian=True):
    model.eval()
    with torch.no_grad():
        data1 = data1.to(device)
        data2 = data2.to(device)
        batch_idx1 = torch.zeros(data1.num_nodes, dtype=torch.long, device=device)
        batch_idx2 = torch.zeros(data2.num_nodes, dtype=torch.long, device=device)

        sim_matrix_list, _ = model(data1, data2, batch_idx1, batch_idx2)
        sim = sim_matrix_list[0].unsqueeze(0)  # [1, N1, N2]

        n1 = torch.tensor([sim.shape[1]], dtype=torch.int32)
        n2 = torch.tensor([sim.shape[2]], dtype=torch.int32)

        if use_hungarian:
            return pygmtools.hungarian(sim, n1=n1, n2=n2).squeeze(0)
        else:
            return sim

def train_loop(model, optimizer, train_loader, val_loader, num_epochs,
               best_model_path='checkpoint.pt', final_model_path='final_model.pt',
               patience=10, resume=False):
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    start_epoch = 0

    train_losses = []
    val_losses = []

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

    print("Starting training...")

    try:
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_loss, train_embeddings = train_epoch_sinkhorn(model, train_loader, optimizer)
            # Evaluate
            val_acc, val_loss, val_embeddings = evaluate_sinkhorn(model, val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

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

            print(f"Epoch {epoch:03} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best was {best_epoch}.")
                break

    except KeyboardInterrupt:
        print("Training interrupted manually (Ctrl+C).")

    # Save final model
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'best_epoch': best_epoch
    }, final_model_path)
    print("Final model saved.")

    return train_losses, val_losses

#----------------------------------------
#            XAI UTILS
#----------------------------------------

# def plt2arr(fig):
#     # Convert a Matplotlib figure to a NumPy array (RGB format)
#     rgb_str = fig.canvas.tostring_rgb()
#     (w, h) = fig.canvas.get_width_height()
#     return np.frombuffer(rgb_str, dtype=np.uint8).reshape((h, w, -1))

# def visualize(h, color, epoch):
#     # Visualize embeddings using t-SNE and return the visualization as an image array
#     fig = plt.figure(figsize=(5,5), frameon=False)
#     fig.suptitle(f'Epoch index = {epoch}')
#     z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())  # Reduce dimensions to 2D
#     plt.xticks([]); plt.yticks([])  # Remove axis ticks
#     plt.scatter(z[:, 0], z[:, 1],
#                 s=70,
#                 c=color.detach().cpu().numpy(),
#                 cmap="Set2")  # Scatter plot with color labels
#     fig.canvas.draw()
#     arr = plt2arr(fig)  # Convert the figure to an array
#     plt.close(fig)  # Close the figure to free memory
#     return arr

# def create_embedding_gif(all_embeddings, output_path, fps=1):
#     """
#     Create a GIF to visualize the evolution of embeddings over epochs.
    
#     all_embeddings : List of tuples (h1, h2), torch.Tensor each [Ni * D]
#     output_path    : str, path to save the GIF, e.g., "embeddings.gif"
#     fps            : int, frames per second for the GIF
#     """
#     num = len(all_embeddings)
#     step = max(1, int(num * 0.05))  # Select a subset of embeddings for visualization
#     indices = list(range(0, num, step))
    
#     images = []
#     for idx in indices:
#         h1, h2 = all_embeddings[idx]
#         h = torch.cat([h1, h2], dim=0)  # Concatenate embeddings from two graphs
#         labels = torch.cat([
#             torch.zeros(h1.size(0), dtype=torch.long),  # Label for graph 1
#             torch.ones(h2.size(0),  dtype=torch.long)  # Label for graph 2
#         ], dim=0)
#         images.append(visualize(h, color=labels, epoch=idx))  # Generate visualization for the current epoch
    
#     clip = ImageSequenceClip(images, fps=fps)  # Create a GIF from the sequence of images
#     clip.write_gif(output_path, fps=fps)  # Save the GIF to the specified path
#     print(f"GIF saved at: {output_path}")

# # Example usage:
# # create_embedding_gif(all_embeddings, "embeddings_evolution.gif", fps=1)

# # %% [markdown]
# # # Unit Test

# # %%
# # Helper: generates a DiGraph with nodes A0…A4, random attributes, and p_edge as edge probability
# def make_random_graph(name, n_nodes=5, p_edge=0.4):
#     G = nx.DiGraph(name=name)
#     for i in range(n_nodes):
#         nid = f'N{i}'
#         # random type, center uniformly in [0,10], normal random unit vector, random length
#         t = random.choice(['room','ws'])
#         center = [random.uniform(0,10), random.uniform(0,10)]
#         v = torch.randn(2).tolist()
#         norm = [v[0]/(abs(v[0])+abs(v[1])+1e-6), v[1]/(abs(v[0])+abs(v[1])+1e-6)]
#         length = random.uniform(1,5)
#         G.add_node(nid, type=t, center=center, normal=norm, length=length)
#     # adds edges with probability p_edge
#     for u in G.nodes():
#         for v in G.nodes():
#             if u!=v and random.random() < p_edge:
#                 G.add_edge(u, v, weight=random.uniform(0.1,1.0))
#     return G

# # 1) Generate 6 graphs
# graphs = [make_random_graph(f'G{i}') for i in range(6)]
# print("Total graphs generated:", len(graphs))

# # 2) Split 70/15/15
# train, val, test = split_graphs(graphs, seed=42)
# print("Split sizes → train:", len(train), "val:", len(val), "test:", len(test))
# assert len(train)==4 and len(val)==1 and len(test)==1, "Split not conforming"

# # 3) Test full matching (g0 vs g0 itself)
# pairs_full = []
# generate_matching_pair_as_data(graphs[0], graphs[0], pairs_full)
# pyg1_f, pyg2_f, P_full = pairs_full[0]
# n = len(graphs[0].nodes())
# print("Full match shape:", P_full.shape)
# assert P_full.shape == (n, n)
# assert torch.allclose(P_full.sum(dim=0), torch.ones(n))
# assert torch.allclose(P_full.sum(dim=1), torch.ones(n))
# print("Full matching OK")

# # 4) Test partial matching (remove 2 nodes from G1)
# g_partial = graphs[1].copy()
# to_remove = random.sample(list(g_partial.nodes()), 2)
# for u in to_remove:
#     g_partial.remove_node(u)
# pairs_part = []
# generate_matching_pair_as_data(graphs[1], g_partial, pairs_part)
# pyg1_p, pyg2_p, P_part = pairs_part[0]
# print("Partial match shape:", P_part.shape, 
#       f"(original {len(graphs[1].nodes())} vs partial {len(g_partial.nodes())})")
# # each column max 1, each row max 1
# assert (P_part.sum(dim=0) <= 1).all()
# assert (P_part.sum(dim=1) <= 1).all()
# print("Partial matching OK\n", P_part)

# # 5) Test round-trip Data→NX on one of the matches
# recon = pyg_data_to_nx_digraph(pyg2_p, [graphs[1], g_partial])
# assert set(recon.nodes()) == set(g_partial.nodes())
# print("Round-trip PyG→NX OK")

# # plot_two_graphs_with_matching(
# #     graphs_list=[pyg1_p, pyg2_p],
# #     gt_perm=P_part,
# #     original_graphs=[graphs[1]],
# #     noise_graphs=[g_partial],
# #     pred_perm=P_part,
# #     match_display="all"
# # )

# print("All advanced tests passed!")


# # %% [markdown]
# # # MSD dataset for SLAM

# # %% [markdown]
# # ## Graph Matching with GNN from scratch

# # %% [markdown]
# # ### Dataset visualization and Preprocessing

# # %%
# original_graphs, noise_graphs, dimensions = deserialize_MSD_dataset("data")

# # Check the number of graphs
# print(f"Number of original graphs: {len(original_graphs)}")
# print(f"Number of noise graphs: {len(noise_graphs)}")

# # %% [markdown]
# #  Checks on dataset

# # %%
# assert len(original_graphs) == len(dimensions), "Number of original and dimensions must be the same dim"
# tot_graphs = 0
# for i in range(len(dimensions)):
#     tot_graphs += dimensions[i]
# assert len(noise_graphs) == tot_graphs, "Number of noise graphs must be equal to the sum of dimensions"

# for i, graph in enumerate(original_graphs):
#     assert i == int(graph.graph['name']), "Graph name must match the index"

# blocks = []
# start = 0
# for size in dimensions:
#     end = start + size
#     blocks.append(noise_graphs[start:end])
#     start = end

# def assert_block_names(blocks):
#     for i, block in enumerate(blocks):
#         base_names = [g.graph['name'].split("_")[0] for g in block]
#         all_same = all(name == base_names[0] for name in base_names)
#         assert all_same, f"Blocco {i+1}: nomi base diversi {base_names}"

# assert_block_names(blocks)

# # %%
# plot_a_graph([original_graphs[0]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# plot_a_graph([noise_graphs[60]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)
# plot_a_graph([noise_graphs[61]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# # %%
# print(original_graphs[53].graph['name'])
# print(len(original_graphs[53].nodes))
# print(len(original_graphs[53].edges))
# # remove all node which has type != "room" or "ws"
# def remove_non_room_ws(graphs):
#     for graph in graphs:
#         nodes_to_remove = [n for n, d in graph.nodes(data=True) if d['type'] not in ['room', 'ws']]
#         graph.remove_nodes_from(nodes_to_remove)
#     return graphs
# original_graphs = remove_non_room_ws(original_graphs)
# noise_graphs = remove_non_room_ws(noise_graphs)

# # convert id to int
# for i, graph in enumerate(original_graphs):
#     original_graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
# for i, graph in enumerate(noise_graphs):
#     noise_graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')


# print(len(original_graphs[53].nodes))
# print(len(original_graphs[53].edges))
# plot_a_graph([original_graphs[53]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=False)
# print(original_graphs[53].nodes(data=True))
# print(original_graphs[53].edges(data=True))

# # %% [markdown]
# # #### Generate G1,G2,GT dataset

# # %%
# train, val, test = split_graphs(original_graphs)

# train_list = []
# for i, g1 in enumerate(train):
#     generate_matching_pair_as_data(g1, g1, train_list)

# val_list = []
# for i, g1 in enumerate(val):
#     generate_matching_pair_as_data(g1, g1, val_list)

# test_list = []
# for i, g1 in enumerate(test):
#     generate_matching_pair_as_data(g1, g1, test_list)


# # compute mean and std
# mean, std = compute_mean_std(train_list)

# # Normalizzazione dei set
# train_pairs_norm = normalize_data_pairs(train_list, mean, std)
# val_pairs_norm = normalize_data_pairs(val_list, mean, std)
# test_pairs_norm = normalize_data_pairs(test_list, mean, std)


# # Visualize the two graphs
# g1_out, g2_perm, gt_perm = train_list[0]

# print(g1_out)
# print("G1 nodes:", g1_out.x[0])
# print(g2_perm)
# print("G2 permuted nodes:", g2_perm.x[0])
# print("Ground truth permutation:\n", gt_perm[0])

# g1_out = pyg_data_to_nx_digraph(g1_out, original_graphs)
# g2_perm = pyg_data_to_nx_digraph(g2_perm, original_graphs)

# plot_a_graph(
#     graphs_list=[g1_out],
#     viz_rooms=True,
#     viz_ws=True,
#     viz_openings=False,
#     viz_room_connection=True,
#     viz_normals=False,
#     viz_room_normals=False,
#     viz_walls=False
# )

# # %%
# class GraphMatchingDataset(Dataset):
#     def __init__(self, pairs):  # lista di (Data, Data, P)
#         self.pairs = pairs

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         return self.pairs[idx]  # data1, data2, P

# def collate_pyg_matching(batch):
#     data1_list, data2_list, perm_list = zip(*batch)
#     batch1 = Batch.from_data_list(data1_list)
#     batch2 = Batch.from_data_list(data2_list)
#     return batch1, batch2, perm_list

# train_dataset = GraphMatchingDataset(train_list)
# val_dataset = GraphMatchingDataset(val_list)
# test_dataset = GraphMatchingDataset(test_list)

# # %% [markdown]
# # ### Model

# # %%
# class MatchingModel_2GCN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.gnn = nn.ModuleList([
#             GCNConv(in_dim, hidden_dim),
#             GCNConv(hidden_dim, out_dim)
#         ])

#     def encode(self, x, edge_index):
#         for i, conv in enumerate(self.gnn):
#             x = conv(x, edge_index)
#             if i < len(self.gnn) - 1:
#                 x = F.relu(x)
#         return x  # shape [num_nodes, out_dim]

#     def forward(self, x1, edge_index1, x2, edge_index2):
#         """
#         Args:
#             x1, x2: node features of graph 1 and 2 (shapes [N, in_dim])
#             edge_index1, edge_index2: edge indices of graph 1 and 2 ([2, num_edges])

#         Returns:
#             scores: similarity matrix [N, N], where scores[i, j] = node i in G1 vs node j in G2
#         """
#         h1 = self.encode(x1, edge_index1)  # [N, D]
#         h2 = self.encode(x2, edge_index2)  # [N, D]
#         scores = torch.matmul(h1, h2.T)    # [N, N]
#         return scores

# # %% [markdown]
# # ### Training

# # %%
# def train_epoch(model, loader, optimizer):
#     """
#     Train the model for one epoch and return the average loss and the embeddings.

#     Returns:
#         avg_loss, h1_all (list of h1), h2_all (list of h2)
#     """
#     model.train()
#     total_loss = 0
#     h1_all, h2_all = [], []

#     for batch1, batch2, P in loader:
#         optimizer.zero_grad()

#         x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#         x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
#         batch_idx1 = batch1.batch.to(device)
#         batch_idx2 = batch2.batch.to(device)

#         h1 = model.encode(x1, edge1)
#         h2 = model.encode(x2, edge2)

#         loss = 0.0
#         B = len(P)

#         for i in range(B):
#             h1_i = h1[batch_idx1 == i]
#             h2_i = h2[batch_idx2 == i]

#             sim = torch.matmul(h1_i, h2_i.T)
#             loss += F.binary_cross_entropy_with_logits(sim, P[i])

#             h1_all.append(h1_i)
#             h2_all.append(h2_i)

#         loss /= B
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     avg_loss = total_loss / len(loader)
#     return avg_loss, h1_all, h2_all


# def evaluate(model, loader):
#     """
#     Evaluate the model on a validation/test set and return both accuracy and loss.

#     Returns:
#         accuracy, avg_loss
#     """
#     model.eval()
#     correct = 0
#     total = 0
#     total_loss = 0

#     with torch.no_grad():
#         for batch1, batch2, P in loader:
#             x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#             x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
#             batch_idx1 = batch1.batch.to(device)
#             batch_idx2 = batch2.batch.to(device)


#             h1 = model.encode(x1, edge1)
#             h2 = model.encode(x2, edge2)

#             B = len(P)

#             for i in range(B):
#                 h1_i = h1[batch_idx1 == i]
#                 h2_i = h2[batch_idx2 == i]

#                 sim = torch.matmul(h1_i, h2_i.T)

#                 pred = sim.argmax(dim=1)
#                 target = P[i].argmax(dim=1)

#                 correct += (pred == target).sum().item()
#                 total += h1_i.size(0)

#                 total_loss += F.binary_cross_entropy_with_logits(sim, P[i], reduction='sum').item()

#     avg_loss = total_loss / total if total > 0 else 0.0
#     accuracy = correct / total if total > 0 else 0.0
#     return accuracy, avg_loss


# # %%
# # Paths for saving models
# best_val_model_path = os.path.join(GNN_PATH, 'best_val_model.pt')
# final_model_path = os.path.join(GNN_PATH, 'final_model.pt')

# # Define hyperparameters
# in_dim = 9  # Dimension of node features
# hidden_dim = 64  # Hidden dimension for GNN
# out_dim = 32  # Output dimension for GNN
# num_epochs = 100
# learning_rate = 0.001
# batch_size = 2

# # Early stopping parameters
# best_val_loss = float('inf')
# patience = 20
# patience_counter = 0
# best_epoch = -1  # To track when best model was found


# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# model = MatchingModel_2GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# train_losses = []
# val_losses = []

# print("Starting training...")

# for epoch in range(num_epochs):
#     # Train for one epoch
#     train_loss, _, _ = train_epoch(model, train_loader, optimizer)
#     # Evaluate on validation set
#     _, val_loss = evaluate(model, val_loader)
#     # Accumulate Losses
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)

#     # Check if validation improved
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         best_epoch = epoch
#         # Save the best model seen so far
#         torch.save(model.state_dict(), best_val_model_path)
#     else:
#         patience_counter += 1

#     # Print progress every 100 epochs
#     if epoch % 100 == 0 or epoch == num_epochs - 1:
#         print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#     # Early stopping check
#     if patience_counter >= patience:
#         print(f"Early stopping triggered at epoch {epoch}.")
#         break

# # Save the final model
# torch.save(model.state_dict(), final_model_path)
# print("\nTraining completed.")

# # %% [markdown]
# # ### Training and Evalutation results

# # %%
# plot_losses(train_losses, val_losses, os.path.join(GNN_PATH, 'losses.png'))

# # %%
# # Load the best model
# model.load_state_dict(torch.load(best_val_model_path, map_location=device))
# model.to(device)

# # Evaluate on the test set
# test_acc, test_loss = evaluate(model, test_loader)
# print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# # %% [markdown]
# # ### Inference

# # %%
# def predict_matching_matrix(model, data1, data2, hard=True):
#     model.eval()
#     with torch.no_grad():
#         data1 = data1.to(device)
#         data2 = data2.to(device)
#         h1 = model.encode(data1.x, data1.edge_index)
#         h2 = model.encode(data2.x, data2.edge_index)
#         sim = torch.matmul(h1, h2.T)  # [N1, N2]

#         if hard:
#             pred = sim.argmax(dim=1)
#             P_pred = torch.zeros_like(sim)
#             P_pred[torch.arange(sim.size(0)), pred] = 1
#             return P_pred
#         else:
#             return F.softmax(sim, dim=1)

# # use the model to predict the matching on a test graph
# g1_out, g2_perm, gt_perm = test_list[0]
# start_time = time.time()
# P_pred = predict_matching_matrix(model, g1_out, g2_perm, hard=True)
# end_time = time.time()

# inference_time = end_time - start_time
# print(f"Inference time: {inference_time:.6f} seconds")
# print(P_pred)  # matrice binaria di permutazione predetta


# # %% [markdown]
# # ### Visualization of results

# # %%
# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=P_pred,
#     original_graphs=original_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all",
# )

# # %% [markdown]
# # # Dataset Preprocessing
# # 

# # %% [markdown]
# # ### Folder Creation
# # 

# # %%
# # Construct the folders if they don't exist
# # GNN_PATH
# # ├── models
# # │   ├── graph matching
# # │   │   ├── equal
# # │   │   ├── global
# # │   │   ├── global_local
# # │   │   └── local
# # │   └── partial graph matching
# # │       ├── room_dropout
# # │       └── ws_dropout
# # ├── preprocessed
# # │   ├── graph matching
# # │   │   ├── equal
# # │   │   ├── global
# # │   │   ├── global_local
# # │   │   └── local
# # │   └── partial graph matching
# # │       ├── room_dropout
# # │       └── ws_dropout
# # └── raw
# #     ├── graph matching
# #     │   ├── equal
# #     │   ├── global
# #     │   ├── global_local
# #     │   └── local
# #     └── partial graph matching
# #         ├── room_dropout
# #         └── ws_dropout

# def create_dir_structure(base_dir="GNN"):
#     categories = [
#         "graph_matching/equal",
#         "graph_matching/global",
#         "graph_matching/global_local",
#         "graph_matching/local",
#         "partial_graph_matching/room_dropout",
#         "partial_graph_matching/ws_dropout",
#     ]

#     levels = ["models", "preprocessed", "raw"]

#     for level in levels:
#         for category in categories:
#             path = os.path.join(base_dir, level, category)
#             os.makedirs(path, exist_ok=True)

# if __name__ == "__main__":
#     create_dir_structure(GNN_PATH)


# # %% [markdown]
# # ### GM Equal

# # %%
# # graph matching-equal path
# gm_path = os.path.join(GNN_PATH, "raw", "graph_matching")
# original_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="equal")

# # %%
# # Check the number of graphs
# print(f"Number of original graphs: {len(original_graphs)}")
# print(original_graphs[0])
# print(original_graphs[0].nodes(data=True))
# print(original_graphs[0].edges(data=True))
# plot_a_graph([original_graphs[0]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# # %%
# ## Checks on dataset
# # assert len(original_graphs) == len(dimensions), "Number of original and dimensions must be the same dim"
# # tot_graphs = 0
# # for i in range(len(dimensions)):
# #     tot_graphs += dimensions[i]
# # assert len(noise_graphs) == tot_graphs, "Number of noise graphs must be equal to the sum of dimensions"

# # for i, graph in enumerate(original_graphs):
# #     assert i == int(graph.graph['name']), "Graph name must match the index"

# # blocks = []
# # start = 0
# # for size in dimensions:
# #     end = start + size
# #     blocks.append(noise_graphs[start:end])
# #     start = end

# # def assert_block_names(blocks):
# #     for i, block in enumerate(blocks):
# #         base_names = [g.graph['name'].split("_")[0] for g in block]
# #         all_same = all(name == base_names[0] for name in base_names)
# #         assert all_same, f"Blocco {i+1}: nomi base diversi {base_names}"

# # assert_block_names(blocks)

# # %%
# # print(original_graphs[53].graph['name'])
# # print(len(original_graphs[53].nodes))
# # print(len(original_graphs[53].edges))
# # # remove all node which has type != "room" or "ws"
# # def remove_non_room_ws(graphs):
# #     for graph in graphs:
# #         nodes_to_remove = [n for n, d in graph.nodes(data=True) if d['type'] not in ['room', 'ws']]
# #         graph.remove_nodes_from(nodes_to_remove)
# #     return graphs
# # original_graphs = remove_non_room_ws(original_graphs)

# # # convert id to int
# # for i, graph in enumerate(original_graphs):
# #     original_graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')


# # print(len(original_graphs[53].nodes))
# # print(len(original_graphs[53].edges))
# # plot_a_graph([original_graphs[53]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=False)
# # print(original_graphs[53].nodes(data=True))
# # print(original_graphs[53].edges(data=True))

# # %% [markdown]
# # #### Generate G1,G2,GT dataset

# # %%
# pair_gt_list = []
# for i, g1 in enumerate(original_graphs):
#     generate_matching_pair_as_data(g1, g1, pair_gt_list)

# train, val, test = split_graphs(pair_gt_list)

# # compute mean and std
# mean, std = compute_mean_std(train)

# # Normalizzazione dei set
# train_pairs_norm = normalize_data_pairs(train, mean, std)
# val_pairs_norm = normalize_data_pairs(val, mean, std)
# test_pairs_norm = normalize_data_pairs(test, mean, std)

# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
# serialize_graph_matching_dataset(
#     train_pairs_norm,
#     gm_equal_preprocessed_path,
#     "train_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     val_pairs_norm,
#     gm_equal_preprocessed_path,
#     "valid_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     test_pairs_norm,
#     gm_equal_preprocessed_path,
#     "test_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     original_graphs,
#     gm_equal_preprocessed_path,
#     "original.pkl"
# )

# # Visualize the two graphs
# g1_out, g2_perm, gt_perm = train[0]

# print(g1_out)
# print("G1 nodes:", g1_out.x[0])
# print(g2_perm)
# print("G2 permuted nodes:", g2_perm.x[0])
# print("Ground truth permutation:\n", gt_perm[0])

# # %%
# # Visualize the two graphs
# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=gt_perm,
#     original_graphs=original_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all",
# )

# # %% [markdown]
# # ### GM Local Noise

# # %%
# # graph matching path
# gm_path = os.path.join(GNN_PATH, "raw", "graph_matching")
# noise_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="local")

# # %%
# # Check the number of graphs
# print(f"Number of original graphs: {len(noise_graphs)}")
# print(noise_graphs[0])
# print(noise_graphs[0].nodes(data=True))
# print(noise_graphs[0].edges(data=True))
# plot_a_graph([noise_graphs[0]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# # %%
# # print(noise_graphs[53].graph['name'])
# # print(len(noise_graphs[53].nodes))
# # print(len(noise_graphs[53].edges))
# # # remove all node which has type != "room" or "ws"
# # def remove_non_room_ws(graphs):
# #     for graph in graphs:
# #         nodes_to_remove = [n for n, d in graph.nodes(data=True) if d['type'] not in ['room', 'ws']]
# #         graph.remove_nodes_from(nodes_to_remove)
# #     return graphs

# # noise_graphs = remove_non_room_ws(noise_graphs)

# # # convert id to int
# # for i, graph in enumerate(noise_graphs):
# #     noise_graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')

# # print(len(noise_graphs[53].nodes))
# # print(len(noise_graphs[53].edges))
# # plot_a_graph([noise_graphs[53]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=False)

# # %% [markdown]
# # #### Generate G1,G2,GT dataset

# # %%
# pair_gt_list = []
# for i, g1 in enumerate(original_graphs):
#     generate_matching_pair_as_data(g1, noise_graphs[i], pair_gt_list)

# train, val, test = split_graphs(pair_gt_list)

# # compute mean and std
# mean, std = compute_mean_std(train)

# # Normalizzazione dei set
# train_pairs_norm = normalize_data_pairs(train, mean, std)
# val_pairs_norm = normalize_data_pairs(val, mean, std)
# test_pairs_norm = normalize_data_pairs(test, mean, std)

# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "local")
# serialize_graph_matching_dataset(
#     train_pairs_norm,
#     gm_equal_preprocessed_path,
#     "train_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     val_pairs_norm,
#     gm_equal_preprocessed_path,
#     "valid_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     test_pairs_norm,
#     gm_equal_preprocessed_path,
#     "test_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     noise_graphs,
#     gm_equal_preprocessed_path,
#     "noise.pkl"
# )

# # Visualize the two graphs
# g1_out, g2_perm, gt_perm = train[0]

# print(g1_out)
# print("G1 nodes:", g1_out.x[0])
# print(g2_perm)
# print("G2 permuted nodes:", g2_perm.x[0])
# print("Ground truth permutation:\n", gt_perm[0])

# # %%
# # Visualize the two graphs
# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=gt_perm,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all",
# )

# # %% [markdown]
# # ### GM Global Noise

# # %%
# # graph matching path
# gm_path = os.path.join(GNN_PATH, "raw", "graph_matching")
# noise_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="global")

# # %%
# # Check the number of graphs
# print(f"Number of original graphs: {len(noise_graphs)}")
# print(noise_graphs[0])
# print(noise_graphs[0].nodes(data=True))
# print(noise_graphs[0].edges(data=True))
# plot_a_graph([noise_graphs[0]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# # %%
# # print(noise_graphs[53].graph['name'])
# # print(len(noise_graphs[53].nodes))
# # print(len(noise_graphs[53].edges))
# # # remove all node which has type != "room" or "ws"
# # def remove_non_room_ws(graphs):
# #     for graph in graphs:
# #         nodes_to_remove = [n for n, d in graph.nodes(data=True) if d['type'] not in ['room', 'ws']]
# #         graph.remove_nodes_from(nodes_to_remove)
# #     return graphs

# # noise_graphs = remove_non_room_ws(noise_graphs)

# # # convert id to int
# # for i, graph in enumerate(noise_graphs):
# #     noise_graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')

# # print(len(noise_graphs[53].nodes))
# # print(len(noise_graphs[53].edges))
# # plot_a_graph([noise_graphs[53]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=False)

# # %% [markdown]
# # #### Generate G1,G2,GT dataset

# # %%
# pair_gt_list = []
# for i, g1 in enumerate(original_graphs):
#     generate_matching_pair_as_data(g1, noise_graphs[i], pair_gt_list)

# train, val, test = split_graphs(pair_gt_list)

# # compute mean and std
# mean, std = compute_mean_std(train)

# # Normalizzazione dei set
# train_pairs_norm = normalize_data_pairs(train, mean, std)
# val_pairs_norm = normalize_data_pairs(val, mean, std)
# test_pairs_norm = normalize_data_pairs(test, mean, std)

# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "global")
# serialize_graph_matching_dataset(
#     train_pairs_norm,
#     gm_equal_preprocessed_path,
#     "train_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     val_pairs_norm,
#     gm_equal_preprocessed_path,
#     "valid_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     test_pairs_norm,
#     gm_equal_preprocessed_path,
#     "test_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     noise_graphs,
#     gm_equal_preprocessed_path,
#     "noise.pkl"
# )

# # Visualize the two graphs
# g1_out, g2_perm, gt_perm = train[0]

# print(g1_out)
# print("G1 nodes:", g1_out.x[0])
# print(g2_perm)
# print("G2 permuted nodes:", g2_perm.x[0])
# print("Ground truth permutation:\n", gt_perm[0])

# # %%
# # Visualize the two graphs
# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=gt_perm,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all",
# )

# # %% [markdown]
# # ### GM Global + Local Noise

# # %%
# # graph matching path
# gm_path = os.path.join(GNN_PATH, "raw", "graph_matching")
# noise_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="global_local")

# # %%
# # Check the number of graphs
# print(f"Number of original graphs: {len(noise_graphs)}")
# print(noise_graphs[0])
# print(noise_graphs[0].nodes(data=True))
# print(noise_graphs[0].edges(data=True))
# plot_a_graph([noise_graphs[0]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# # %%
# # print(noise_graphs[53].graph['name'])
# # print(len(noise_graphs[53].nodes))
# # print(len(noise_graphs[53].edges))
# # # remove all node which has type != "room" or "ws"
# # def remove_non_room_ws(graphs):
# #     for graph in graphs:
# #         nodes_to_remove = [n for n, d in graph.nodes(data=True) if d['type'] not in ['room', 'ws']]
# #         graph.remove_nodes_from(nodes_to_remove)
# #     return graphs

# # noise_graphs = remove_non_room_ws(noise_graphs)

# # # convert id to int
# # for i, graph in enumerate(noise_graphs):
# #     noise_graphs[i] = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')

# # print(len(noise_graphs[53].nodes))
# # print(len(noise_graphs[53].edges))
# # plot_a_graph([noise_graphs[53]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=False)

# # %% [markdown]
# # #### Generate G1,G2,GT dataset

# # %%
# pair_gt_list = []
# for i, g1 in enumerate(original_graphs):
#     generate_matching_pair_as_data(g1, noise_graphs[i], pair_gt_list)

# train, val, test = split_graphs(pair_gt_list)

# # compute mean and std
# mean, std = compute_mean_std(train)

# # Normalizzazione dei set
# train_pairs_norm = normalize_data_pairs(train, mean, std)
# val_pairs_norm = normalize_data_pairs(val, mean, std)
# test_pairs_norm = normalize_data_pairs(test, mean, std)

# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "global_local")
# serialize_graph_matching_dataset(
#     train_pairs_norm,
#     gm_equal_preprocessed_path,
#     "train_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     val_pairs_norm,
#     gm_equal_preprocessed_path,
#     "valid_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     test_pairs_norm,
#     gm_equal_preprocessed_path,
#     "test_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     noise_graphs,
#     gm_equal_preprocessed_path,
#     "noise.pkl"
# )

# # Visualize the two graphs
# g1_out, g2_perm, gt_perm = train[0]

# print(g1_out)
# print("G1 nodes:", g1_out.x[0])
# print(g2_perm)
# print("G2 permuted nodes:", g2_perm.x[0])
# print("Ground truth permutation:\n", gt_perm[0])

# # %%
# # Visualize the two graphs
# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=gt_perm,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all",
# )

# # %% [markdown]
# # ### WS dropout

# # %%
# # graph matching-equal path
# gm_path = os.path.join(GNN_PATH, "raw", "graph_matching")
# original_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="equal")
# # graph matching path
# gm_path = os.path.join(GNN_PATH, "raw", "partial_graph_matching")
# noise_graphs, _, _ = deserialize_MSD_dataset(data_path=gm_path, original_path="ws_dropout")

# # %%
# # Check the number of graphs
# print(f"Number of original graphs: {len(noise_graphs)}")
# print(noise_graphs[0])
# print(noise_graphs[0].nodes(data=True))
# print(noise_graphs[0].edges(data=True))
# plot_a_graph([noise_graphs[0]], viz_rooms=True, viz_ws=True, viz_openings=False, viz_room_connection=True, viz_normals=False, viz_room_normals=True, viz_walls=True)

# # %% [markdown]
# # #### Generate G1,G2,GT dataset

# # %%
# pair_gt_list = []
# for i, g1 in enumerate(original_graphs):
#     generate_matching_pair_as_data(g1, noise_graphs[i], pair_gt_list)

# train, val, test = split_graphs(pair_gt_list)

# # compute mean and std
# mean, std = compute_mean_std(train)

# # Normalizzazione dei set
# train_pairs_norm = normalize_data_pairs(train, mean, std)
# val_pairs_norm = normalize_data_pairs(val, mean, std)
# test_pairs_norm = normalize_data_pairs(test, mean, std)

# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "ws_dropout")
# serialize_graph_matching_dataset(
#     train_pairs_norm,
#     gm_equal_preprocessed_path,
#     "train_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     val_pairs_norm,
#     gm_equal_preprocessed_path,
#     "valid_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     test_pairs_norm,
#     gm_equal_preprocessed_path,
#     "test_dataset.pkl"
# )
# serialize_graph_matching_dataset(
#     noise_graphs,
#     gm_equal_preprocessed_path,
#     "noise.pkl"
# )

# g1_out, g2_perm, gt_perm = train[0]

# print(g1_out)
# print("G1 nodes:", g1_out.x[0])
# print(g2_perm)
# print("G2 permuted nodes:", g2_perm.x[0])
# print("Ground truth permutation:\n", gt_perm[0])

# # %%
# # Visualize the two graphs
# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=gt_perm,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all"
# )

# # %% [markdown]
# # # Graph matching

# # %% [markdown]
# # ## Equal graphs

# # %%
# #load preprocessed dataset
# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
# models_path = os.path.join(GNN_PATH, 'models', 'graph_matching', 'equal')

# original_graphs = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "original.pkl"
# )
# train_list = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "train_dataset.pkl"
# )
# val_list = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "valid_dataset.pkl"
# )
# test_list = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "test_dataset.pkl"
# )

# # %%
# d1,d2,gt = train_list[0]
# print(d1)
# print(d2)
# print(gt)
# plot_two_graphs_with_matching([d1,d2],gt_perm=gt,original_graphs=original_graphs)

# # %%
# train_dataset = GraphMatchingDataset(train_list)
# val_dataset = GraphMatchingDataset(val_list)
# test_dataset = GraphMatchingDataset(test_list)

# # %%
# class MatchingModel_GATv2Sinkhorn(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.gnn = nn.ModuleList([
#             GATv2Conv(in_dim, hidden_dim),
#             GATv2Conv(hidden_dim, out_dim)
#         ])
#         self.inst_norm = nn.InstanceNorm2d(1, affine=True)

#     def encode(self, x, edge_index):
#         for i, conv in enumerate(self.gnn):
#             x = conv(x, edge_index)
#             if i < len(self.gnn) - 1:
#                 x = F.relu(x)
#         return x

#     def forward(self, batch1, batch2, batch_idx1=None, batch_idx2=None):
#         device = next(self.parameters()).device
    
#         x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#         x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
    
#         # Use existing batch indices if provided, otherwise extract from Data object
#         if batch_idx1 is None:
#             batch_idx1 = batch1.batch.to(device)
#         if batch_idx2 is None:
#             batch_idx2 = batch2.batch.to(device)
    
#         h1 = self.encode(x1, edge1)
#         h2 = self.encode(x2, edge2)
    
#         B = batch_idx1.max().item() + 1 if batch_idx1 is not None else 1
#         perm_pred_list = []
#         all_embeddings = []
    
#         for i in range(B):
#             h1_i = h1[batch_idx1 == i]
#             h2_i = h2[batch_idx2 == i]
    
#             sim = torch.matmul(h1_i, h2_i.T)  # [N1, N2]
#             sim_batched = sim.unsqueeze(0).unsqueeze(1)  # [1, 1, N1, N2]
#             sim_normed = self.inst_norm(sim_batched).squeeze(1)  # [1, N1, N2]
    
#             n1 = torch.tensor([h1_i.size(0)], dtype=torch.int32, device=device)
#             n2 = torch.tensor([h2_i.size(0)], dtype=torch.int32, device=device)
    
#             S = pygmtools.sinkhorn(sim_normed, n1=n1, n2=n2, dummy_row=False)[0]
#             perm_pred_list.append(S)
#             all_embeddings.append((h1_i, h2_i))
    
#         return perm_pred_list, all_embeddings


# # %%
# # Percorsi per salvare i modelli
# best_val_model_path = os.path.join(models_path, 'best_val_model.pt')
# final_model_path = os.path.join(models_path, 'final_model.pt')

# # Iperparametri
# in_dim = 7
# hidden_dim = 64
# out_dim = 32
# num_epochs = 100
# learning_rate = 0.001
# batch_size = 3

# # Early stopping
# best_val_loss = float('inf')
# patience = 10
# patience_counter = 0
# best_epoch = -1

# # Loader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# # Modello e ottimizzatore
# model = MatchingModel_GATv2Sinkhorn(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# #model summary
# print(model)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# # %%
# train_losses, val_losses = train_loop(
#     model=model,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=num_epochs,
#     best_model_path=best_val_model_path,
#     final_model_path=final_model_path,
#     patience=patience,
#     resume=False
# )


# # %%
# plot_losses(train_losses, val_losses, os.path.join(models_path, 'losses.png'))

# # %%
# checkpoint = torch.load(best_val_model_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model.to(device)

# # Evaluate on the test set
# test_acc, test_loss, test_embeddings = evaluate_sinkhorn(model, test_loader)
# print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# inference_time = 0
# # use the model to predict the matching on a test graph
# for i, (g1_out, g2_perm, gt_perm) in enumerate(test_list):
#     start_time = time.time()
#     result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)
#     end_time = time.time()
#     inference_time += end_time - start_time
#     errors = (result != gt_perm.to(result.device)).sum().item()
#     if errors > 0:
#         print(f"Graph {i}: Errors found: {errors}")

# print(f"Inference time: {inference_time/len(test_list):.6f} seconds")

# # %%
# g1_out, g2_perm, gt_perm = test_list[1]
# result = predict_matching_matrix(model, g1_out, g2_perm)

# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=result,
#     original_graphs=original_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="all",
# )

# # %% [markdown]
# # ## Local noise

# # %%
# #load preprocessed dataset
# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
# gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "local")
# models_path = os.path.join(GNN_PATH, 'models', 'graph_matching', 'local')

# original_graphs = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "original.pkl"
# )
# noise_graphs = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "noise.pkl"
# )

# train_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "train_dataset.pkl"
# )
# val_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "valid_dataset.pkl"
# )
# test_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "test_dataset.pkl"
# )

# # %%
# d1,d2,gt = train_list[0]
# print(d1)
# print(d2)
# print(gt)
# plot_two_graphs_with_matching([d1,d2],gt_perm=gt,original_graphs=original_graphs,noise_graphs=noise_graphs)

# # %%
# train_dataset = GraphMatchingDataset(train_list)
# val_dataset = GraphMatchingDataset(val_list)
# test_dataset = GraphMatchingDataset(test_list)

# # %%
# class MatchingModel_GATv2Sinkhorn(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.gnn = nn.ModuleList([
#             GATv2Conv(in_dim, hidden_dim),
#             GATv2Conv(hidden_dim, out_dim)
#         ])
#         self.inst_norm = nn.InstanceNorm2d(1, affine=True)

#     def encode(self, x, edge_index):
#         for i, conv in enumerate(self.gnn):
#             x = conv(x, edge_index)
#             if i < len(self.gnn) - 1:
#                 x = F.relu(x)
#         return x

#     def forward(self, batch1, batch2, batch_idx1=None, batch_idx2=None):
#         device = next(self.parameters()).device
    
#         x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#         x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
    
#         # Use existing batch indices if provided, otherwise extract from Data object
#         if batch_idx1 is None:
#             batch_idx1 = batch1.batch.to(device)
#         if batch_idx2 is None:
#             batch_idx2 = batch2.batch.to(device)
    
#         h1 = self.encode(x1, edge1)
#         h2 = self.encode(x2, edge2)
    
#         B = batch_idx1.max().item() + 1 if batch_idx1 is not None else 1
#         perm_pred_list = []
#         all_embeddings = []
    
#         for i in range(B):
#             h1_i = h1[batch_idx1 == i]
#             h2_i = h2[batch_idx2 == i]
    
#             sim = torch.matmul(h1_i, h2_i.T)  # [N1, N2]
#             sim_batched = sim.unsqueeze(0).unsqueeze(1)  # [1, 1, N1, N2]
#             sim_normed = self.inst_norm(sim_batched).squeeze(1)  # [1, N1, N2]
    
#             n1 = torch.tensor([h1_i.size(0)], dtype=torch.int32, device=device)
#             n2 = torch.tensor([h2_i.size(0)], dtype=torch.int32, device=device)
    
#             S = pygmtools.sinkhorn(sim_normed, n1=n1, n2=n2, dummy_row=False)[0]
#             perm_pred_list.append(S)
#             all_embeddings.append((h1_i, h2_i))
    
#         return perm_pred_list, all_embeddings


# # %%
# # Percorsi per salvare i modelli
# best_val_model_path = os.path.join(models_path, 'best_val_model.pt')
# final_model_path = os.path.join(models_path, 'final_model.pt')

# # Iperparametri
# in_dim = 7
# hidden_dim = 64
# out_dim = 32
# num_epochs = 100
# learning_rate = 0.001
# batch_size = 3

# # Early stopping
# best_val_loss = float('inf')
# patience = 10
# patience_counter = 0
# best_epoch = -1

# # Loader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# # Modello e ottimizzatore
# model = MatchingModel_GATv2Sinkhorn(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# #model summary
# print(model)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# # %%
# train_losses, val_losses = train_loop(
#     model=model,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=num_epochs,
#     best_model_path=best_val_model_path,
#     final_model_path=final_model_path,
#     patience=patience,
#     resume=False
# )


# # %%
# plot_losses(train_losses, val_losses, os.path.join(models_path, 'losses.png'))

# # %%
# checkpoint = torch.load(best_val_model_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model.to(device)

# # Evaluate on the test set
# test_acc, test_loss, test_embeddings = evaluate_sinkhorn(model, test_loader)
# print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# inference_time = 0
# # use the model to predict the matching on a test graph
# for i, (g1_out, g2_perm, gt_perm) in enumerate(test_list):
#     start_time = time.time()
#     result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)
#     end_time = time.time()
#     inference_time += end_time - start_time
#     errors = (result != gt_perm.to(result.device)).sum().item()
#     if errors > 0:
#         print(f"Graph {i}: Errors found: {errors}")

# print(f"Inference time: {inference_time/len(test_list):.6f} seconds")

# # %%
# g1_out, g2_perm, gt_perm = test_list[23]
# result = predict_matching_matrix(model, g1_out, g2_perm)

# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=result,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="wrong",
# )

# # %% [markdown]
# # ## Global noise

# # %%
# #load preprocessed dataset
# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
# gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "global")
# models_path = os.path.join(GNN_PATH, 'models', 'graph_matching', 'global')

# original_graphs = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "original.pkl"
# )
# noise_graphs = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "noise.pkl"
# )

# train_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "train_dataset.pkl"
# )
# val_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "valid_dataset.pkl"
# )
# test_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "test_dataset.pkl"
# )

# # %%
# d1,d2,gt = train_list[0]
# print(d1)
# print(d2)
# print(gt)
# plot_two_graphs_with_matching([d1,d2],gt_perm=gt,original_graphs=original_graphs,noise_graphs=noise_graphs)

# # %%
# train_dataset = GraphMatchingDataset(train_list)
# val_dataset = GraphMatchingDataset(val_list)
# test_dataset = GraphMatchingDataset(test_list)

# # %%
# class MatchingModel_GATv2Sinkhorn(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.gnn = nn.ModuleList([
#             GATv2Conv(in_dim, hidden_dim),
#             GATv2Conv(hidden_dim, out_dim)
#         ])
#         self.inst_norm = nn.InstanceNorm2d(1, affine=True)

#     def encode(self, x, edge_index):
#         for i, conv in enumerate(self.gnn):
#             x = conv(x, edge_index)
#             if i < len(self.gnn) - 1:
#                 x = F.relu(x)
#         return x

#     def forward(self, batch1, batch2, batch_idx1=None, batch_idx2=None):
#         device = next(self.parameters()).device
    
#         x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#         x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
    
#         # Use existing batch indices if provided, otherwise extract from Data object
#         if batch_idx1 is None:
#             batch_idx1 = batch1.batch.to(device)
#         if batch_idx2 is None:
#             batch_idx2 = batch2.batch.to(device)
    
#         h1 = self.encode(x1, edge1)
#         h2 = self.encode(x2, edge2)
    
#         B = batch_idx1.max().item() + 1 if batch_idx1 is not None else 1
#         perm_pred_list = []
#         all_embeddings = []
    
#         for i in range(B):
#             h1_i = h1[batch_idx1 == i]
#             h2_i = h2[batch_idx2 == i]
    
#             sim = torch.matmul(h1_i, h2_i.T)  # [N1, N2]
#             sim_batched = sim.unsqueeze(0).unsqueeze(1)  # [1, 1, N1, N2]
#             sim_normed = self.inst_norm(sim_batched).squeeze(1)  # [1, N1, N2]
    
#             n1 = torch.tensor([h1_i.size(0)], dtype=torch.int32, device=device)
#             n2 = torch.tensor([h2_i.size(0)], dtype=torch.int32, device=device)
    
#             S = pygmtools.sinkhorn(sim_normed, n1=n1, n2=n2, dummy_row=False)[0]
#             perm_pred_list.append(S)
#             all_embeddings.append((h1_i, h2_i))
    
#         return perm_pred_list, all_embeddings


# # %%
# # Percorsi per salvare i modelli
# best_val_model_path = os.path.join(models_path, 'best_val_model.pt')
# final_model_path = os.path.join(models_path, 'final_model.pt')

# # Iperparametri
# in_dim = 7
# hidden_dim = 64
# out_dim = 32
# num_epochs = 100
# learning_rate = 0.001
# batch_size = 3

# # Early stopping
# best_val_loss = float('inf')
# patience = 10
# patience_counter = 0
# best_epoch = -1

# # Loader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# # Modello e ottimizzatore
# model = MatchingModel_GATv2Sinkhorn(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# #model summary
# print(model)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# # %%
# train_losses, val_losses = train_loop(
#     model=model,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=num_epochs,
#     best_model_path=best_val_model_path,
#     final_model_path=final_model_path,
#     patience=patience,
#     resume=False
# )


# # %%
# plot_losses(train_losses, val_losses, os.path.join(models_path, 'losses.png'))

# # %%
# checkpoint = torch.load(best_val_model_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model.to(device)

# # Evaluate on the test set
# test_acc, test_loss, test_embeddings = evaluate_sinkhorn(model, test_loader)
# print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# inference_time = 0
# # use the model to predict the matching on a test graph
# for i, (g1_out, g2_perm, gt_perm) in enumerate(test_list):
#     start_time = time.time()
#     result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)
#     end_time = time.time()
#     inference_time += end_time - start_time
#     errors = (result != gt_perm.to(result.device)).sum().item()
#     if errors > 0:
#         print(f"Graph {i}: Errors found: {errors}")

# print(f"Inference time: {inference_time/len(test_list):.6f} seconds")

# # %%
# g1_out, g2_perm, gt_perm = test_list[39]
# result = predict_matching_matrix(model, g1_out, g2_perm)

# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=result,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="wrong",
# )

# # %% [markdown]
# # ## Global + Local noise

# # %%
# #load preprocessed dataset
# gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
# gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "global_local")
# models_path = os.path.join(GNN_PATH, 'models', 'graph_matching', 'global_local')

# original_graphs = deserialize_graph_matching_dataset(
#     gm_equal_preprocessed_path,
#     "original.pkl"
# )
# noise_graphs = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "noise.pkl"
# )

# train_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "train_dataset.pkl"
# )
# val_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "valid_dataset.pkl"
# )
# test_list = deserialize_graph_matching_dataset(
#     gm_local_preprocessed_path,
#     "test_dataset.pkl"
# )

# # %%
# d1,d2,gt = train_list[0]
# print(d1)
# print(d2)
# print(gt)
# plot_two_graphs_with_matching([d1,d2],gt_perm=gt,original_graphs=original_graphs,noise_graphs=noise_graphs)

# # %%
# train_dataset = GraphMatchingDataset(train_list)
# val_dataset = GraphMatchingDataset(val_list)
# test_dataset = GraphMatchingDataset(test_list)

# # %%
# class MatchingModel_GATv2Sinkhorn(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.gnn = nn.ModuleList([
#             GATv2Conv(in_dim, hidden_dim),
#             GATv2Conv(hidden_dim, out_dim)
#         ])
#         self.inst_norm = nn.InstanceNorm2d(1, affine=True)

#     def encode(self, x, edge_index):
#         for i, conv in enumerate(self.gnn):
#             x = conv(x, edge_index)
#             if i < len(self.gnn) - 1:
#                 x = F.relu(x)
#         return x

#     def forward(self, batch1, batch2, batch_idx1=None, batch_idx2=None):
#         device = next(self.parameters()).device
    
#         x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#         x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
    
#         # Use existing batch indices if provided, otherwise extract from Data object
#         if batch_idx1 is None:
#             batch_idx1 = batch1.batch.to(device)
#         if batch_idx2 is None:
#             batch_idx2 = batch2.batch.to(device)
    
#         h1 = self.encode(x1, edge1)
#         h2 = self.encode(x2, edge2)
    
#         B = batch_idx1.max().item() + 1 if batch_idx1 is not None else 1
#         perm_pred_list = []
#         all_embeddings = []
    
#         for i in range(B):
#             h1_i = h1[batch_idx1 == i]
#             h2_i = h2[batch_idx2 == i]
    
#             sim = torch.matmul(h1_i, h2_i.T)  # [N1, N2]
#             sim_batched = sim.unsqueeze(0).unsqueeze(1)  # [1, 1, N1, N2]
#             sim_normed = self.inst_norm(sim_batched).squeeze(1)  # [1, N1, N2]
    
#             n1 = torch.tensor([h1_i.size(0)], dtype=torch.int32, device=device)
#             n2 = torch.tensor([h2_i.size(0)], dtype=torch.int32, device=device)
    
#             S = pygmtools.sinkhorn(sim_normed, n1=n1, n2=n2, dummy_row=False)[0]
#             perm_pred_list.append(S)
#             all_embeddings.append((h1_i, h2_i))
    
#         return perm_pred_list, all_embeddings


# # %%
# # Percorsi per salvare i modelli
# best_val_model_path = os.path.join(models_path, 'best_val_model.pt')
# final_model_path = os.path.join(models_path, 'final_model.pt')

# # Iperparametri
# in_dim = 7
# hidden_dim = 64
# out_dim = 32
# num_epochs = 100
# learning_rate = 0.001
# batch_size = 3

# # Early stopping
# best_val_loss = float('inf')
# patience = 10
# patience_counter = 0
# best_epoch = -1

# # Loader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# # Modello e ottimizzatore
# model = MatchingModel_GATv2Sinkhorn(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# #model summary
# print(model)
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# # %%
# train_losses, val_losses = train_loop(
#     model=model,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=num_epochs,
#     best_model_path=best_val_model_path,
#     final_model_path=final_model_path,
#     patience=patience,
#     resume=False
# )


# # %%
# plot_losses(train_losses, val_losses, os.path.join(models_path, 'losses.png'))

# # %%
# checkpoint = torch.load(best_val_model_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model.to(device)

# # Evaluate on the test set
# test_acc, test_loss, test_embeddings = evaluate_sinkhorn(model, test_loader)
# print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# inference_time = 0
# # use the model to predict the matching on a test graph
# for i, (g1_out, g2_perm, gt_perm) in enumerate(test_list):
#     start_time = time.time()
#     result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)
#     end_time = time.time()
#     inference_time += end_time - start_time
#     errors = (result != gt_perm.to(result.device)).sum().item()
#     if errors > 0:
#         print(f"Graph {i}: Errors found: {errors}")

# print(f"Inference time: {inference_time/len(test_list):.6f} seconds")

# # %%
# g1_out, g2_perm, gt_perm = test_list[3]
# result = predict_matching_matrix(model, g1_out, g2_perm)

# plot_two_graphs_with_matching(
#     [g1_out, g2_perm],
#     gt_perm=gt_perm,
#     pred_perm=result,
#     original_graphs=original_graphs,
#     noise_graphs=noise_graphs,
#     viz_rooms=True,
#     viz_ws=True,
#     match_display="wrong",
# )

# %% [markdown]
# # Partial graph matching

# %% [markdown]
# ## Import of AFA-U module and TopK from ThinkMatch

# %%
import sys
import os
destination_dir = os.path.join('AFAT')

# Ensure the destination directory is in sys.path
if destination_dir not in sys.path:
    sys.path.append(destination_dir)

# %% [markdown]
# ## Ws dropout

# %%
#load preprocessed dataset
gm_equal_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal")
gm_local_preprocessed_path = os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", "ws_dropout")
models_path = os.path.join(GNN_PATH, 'models', "partial_graph_matching", "ws_dropout")

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

# %%
d1,d2,gt = train_list[0]
print(d1)
print(d2)
print(gt)
print(gt.shape)
plot_two_graphs_with_matching([d1,d2],gt_perm=gt,original_graphs=original_graphs,noise_graphs=noise_graphs)

# %%
train_dataset = GraphMatchingDataset(train_list)
val_dataset = GraphMatchingDataset(val_list)
test_dataset = GraphMatchingDataset(test_list)

# %%
# class MatchingModel_GATv2SinkhornTopK(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         # Two-layer GATv2 encoder
#         self.gnn = nn.ModuleList([
#             GATv2Conv(in_dim, hidden_dim),
#             GATv2Conv(hidden_dim, out_dim)
#         ])
#         # Instance normalization for similarity scores
#         self.inst_norm = nn.InstanceNorm2d(1, affine=True)
#         # AFA-U module to predict inlier count K
#         self.kpred = KPredNet()

#     def encode(self, x, edge_index):
#         """
#         Encode node features through stacked GATv2 layers,
#         applying ReLU between layers to introduce non-linearity.
#         """
#         for i, conv in enumerate(self.gnn):
#             x = conv(x, edge_index)
#             if i < len(self.gnn) - 1:
#                 x = F.relu(x)
#         return x

#     def forward(self, batch1, batch2, batch_idx1=None, batch_idx2=None):
#         """
#         Perform partial graph matching:
#         1. Encode node features with GATv2.
#         2. Compute soft matching via Sinkhorn.
#         3. Predict inlier count K with AFA-U.
#         4. Apply differentiable Top-K to obtain hard matches.

#         Returns:
#             perm_pred_list: list of [N1 x N2] hard match matrices
#             all_embeddings: list of (h1_i, h2_i) tuples for analysis
#         """
#         device = next(self.parameters()).device

#         # Prepare batched tensors
#         x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
#         x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)
        
#         # Use existing batch indices if provided, otherwise extract from Data object
#         if batch_idx1 is None:
#             batch_idx1 = batch1.batch.to(device)
#         if batch_idx2 is None:
#             batch_idx2 = batch2.batch.to(device)

#         # Encode graphs
#         h1 = self.encode(x1, edge1)
#         h2 = self.encode(x2, edge2)

#         B = int(batch_idx1.max().item()) + 1
#         perm_pred_list = []  # will hold final partial match matrices
#         all_embeddings = []  # store embeddings for debugging or analysis

#         for b in range(B):
#             # Extract per-graph embeddings
#             h1_i = h1[batch_idx1 == b]
#             h2_i = h2[batch_idx2 == b]

#             # Pairwise similarity
#             sim = torch.matmul(h1_i, h2_i.T)  # [N1, N2]
#             sim_batched = sim.unsqueeze(0).unsqueeze(1)  # [1,1,N1,N2]
#             sim_normed = self.inst_norm(sim_batched).squeeze(1)  # [N1,N2]

#             # Prepare sizes for Sinkhorn
#             n1 = torch.tensor([h1_i.size(0)], dtype=torch.int32, device=device)
#             n2 = torch.tensor([h2_i.size(0)], dtype=torch.int32, device=device)

#             # Soft matching via Sinkhorn
#             soft_matching = pygmtools.sinkhorn(sim_normed, n1=n1, n2=n2)[0]

#             # Predict number of inliers K using AFA-U
#             # AFA-U expects inputs shaped (batch, nodes, features) for each graph
#             row_emb, col_emb = self.kpred(
#                 h1_i.unsqueeze(0),  # shape (1, N1, d)
#                 h2_i.unsqueeze(0),  # shape (1, N2, d)
#                 sim_normed # shape (1, N1, N2)
#             )

#             k_logits = self.kpred_head(row_emb, col_emb)
#             # Clamp predicted k between 1 and size of graph2, and convert to long
#             k_pred = torch.clamp(k_logits.squeeze(), min=1, max=h2_i.size(0)).long()


#             # Hard partial matching: differentiable Top-K
#             hard_match = soft_topk(
#                 soft_matching.unsqueeze(0),  # add batch dimension
#                 k_pred.unsqueeze(0),         # inlier counts per batch
#                 max_iter=20,
#                 tau=5e-2,
#                 nrows=n1,
#                 ncols=n2,
#                 return_prob=False
#             )[0]

#             perm_pred_list.append(hard_match)
#             all_embeddings.append((h1_i, h2_i))

#         return perm_pred_list, all_embeddings

# %%
# AFA-U inlier predictor and Top-K matching from AFAT
from k_pred_net import Encoder as AFAUEncoder
from sinkhorn_topk import soft_topk

class MatchingModel_GATv2SinkhornTopK(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        sinkhorn_max_iter: int = 20,
        sinkhorn_tau: float = 5e-2,
    ):
        super().__init__()
        # ─── 1) GNN backbone: two-layer GATv2
        # First GATv2Conv projects in_dim → hidden_dim, apply ReLU
        # Second GATv2Conv projects hidden_dim → out_dim, no activation afterwards
        self.gnn = nn.ModuleList([
            GATv2Conv(in_dim, hidden_dim),
            GATv2Conv(hidden_dim, out_dim),
        ])
        # InstanceNorm to normalize each [N1×N2] similarity map
        self.inst_norm = nn.InstanceNorm2d(1, affine=True)

        # ─── 2) AFA-U “unified” module to predict number of inliers K
        #  univ_size = maximum graph size, used to pad all embeddings to fixed length
        self.k_top_encoder = AFAUEncoder()

        # Two small MLPs to reduce pooled embedding → scalar in [0,1]
        self.final_row = nn.Sequential(
            nn.Linear(out_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.final_col = nn.Sequential(
            nn.Linear(out_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # Sinkhorn-TopK hyperparams
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_tau      = sinkhorn_tau

    def encode(self, x, edge_index):
        """
        Pass input features x through the two GATv2Conv layers.
        Apply ReLU after the first, but not after the last.
        """
        for i, conv in enumerate(self.gnn):
            x = conv(x, edge_index)
            if i < len(self.gnn) - 1:
                x = F.relu(x)
        return x

    def forward(self, batch1, batch2, batch_idx1=None, batch_idx2=None):
        """
        batch1, batch2: PyG Data objects for the two graphs in each pair.
        batch_idx1, batch_idx2: optional precomputed batch assignments.
        Returns a list of final hard match matrices (perm_pred_list) and
        the raw embeddings for each graph pair (all_embeddings).
        """
        # device = next(self.parameters()).device

        # ─── 1) Unpack node features & edge indices, move to GPU/CPU
        x1, edge1 = batch1.x.to(device), batch1.edge_index.to(device)
        x2, edge2 = batch2.x.to(device), batch2.edge_index.to(device)

        # ─── 2) Determine which node belongs to which graph in the batch
        #    If not supplied, read from Data.batch
        batch_idx1 = batch1.batch.to(device) if batch_idx1 is None else batch_idx1.to(device)
        batch_idx2 = batch2.batch.to(device) if batch_idx2 is None else batch_idx2.to(device)

        # ─── 3) Encode both sets of nodes via the GNN
        h1 = self.encode(x1, edge1)  # [total_nodes1, out_dim]
        h2 = self.encode(x2, edge2)  # [total_nodes2, out_dim]

        # How many graph pairs in this minibatch?
        B = batch_idx1.max().item() + 1

        perm_pred_list = []
        all_embeddings = []

        for b in range(B):
            # Isolate embeddings for the b-th graph pair
            h1_i = h1[batch_idx1 == b]  # shape [N1, d]
            h2_i = h2[batch_idx2 == b]  # shape [N2, d]
            N1, N2 = h1_i.size(0), h2_i.size(0)

            # ─── 4) Compute raw similarity: dot product between all node pairs
            sim = torch.matmul(h1_i, h2_i.T)    # [N1, N2]
            # Normalize per-instance so Sinkhorn is stable
            sim_b = sim.unsqueeze(0).unsqueeze(1)   # [1,1,N1,N2]
            sim_n = self.inst_norm(sim_b).squeeze(1)  # [1,N1,N2]

            # Prepare row/col sizes for pygmtools
            n1_t = torch.tensor([N1], dtype=torch.int32, device=device)
            n2_t = torch.tensor([N2], dtype=torch.int32, device=device)

            # Soft Sinkhorn → soft_match [N1,N2]
            soft_S = pygmtools.sinkhorn(sim_n, n1=n1_t, n2=n2_t, dummy_row=False)[0]

            # ─── 5) AFA-U predicts inlier count K from soft matching
            #   a) Expand dims to batch form
            row_emb = h1_i.unsqueeze(0)      # [1, N1, d]
            col_emb = h2_i.unsqueeze(0)      # [1, N2, d]
            cost_mat = sim_n                 # [1, N1, N2]

            #   b) Run the bipartite-attention encoder
            out_r, out_c = self.k_top_encoder(row_emb, col_emb, cost_mat) # [1, N1, d], [1, N2, d]
            
            #   c) Dynamic max over nodes
            g_r = out_r.max(dim=1).values     # [1, d]
            g_c = out_c.max(dim=1).values     # [1, d]

            #   d) Small MLPs → fraction in [0,1]
            k_r = self.final_row(g_r).squeeze(-1)  # [1]
            k_c = self.final_col(g_c).squeeze(-1)  # [1]
            ks  = (k_r + k_c) / 2                  # [1] average of row/col predictions

            # ─── 6) Top-K matching
            if self.training:
                # use ground-truth K
                ks_gt = torch.tensor([N2], dtype=torch.long, device=device)
                hard_S, soft_S = soft_topk(
                    sim_n, ks_gt,
                    max_iter=self.sinkhorn_max_iter,
                    tau=self.sinkhorn_tau,
                    nrows=n1_t, ncols=n2_t,
                    return_prob=True
                )
                # for loss use soft_S[0], for logging or evaluation you can also inspect hard_S[0]
                perm_pred_list.append(soft_S[0])
            else:
                ks_eff = (ks * N2).long()
                hard_S = soft_topk(
                    sim_n, ks_eff,
                    max_iter=self.sinkhorn_max_iter,
                    tau=self.sinkhorn_tau,
                    nrows=n1_t, ncols=n2_t,
                    return_prob=False
                )
                perm_pred_list.append(hard_S[0])


            # ─── 7) Collect outputs for this pair
            all_embeddings.append((h1_i, h2_i))   # store embeddings for any downstream use

        return perm_pred_list, all_embeddings

# %%
# Percorsi per salvare i modelli
best_val_model_path = os.path.join(models_path, 'best_val_model.pt')
final_model_path = os.path.join(models_path, 'final_model.pt')

# Iperparametri
in_dim = 7
hidden_dim = 64
out_dim = 32
num_epochs = 100
learning_rate = 0.001
batch_size = 3

# Early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_epoch = -1

# Loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg_matching, generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg_matching)

# Modello e ottimizzatore
model = MatchingModel_GATv2SinkhornTopK(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#model summary
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# %%
train_losses, val_losses = train_loop(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    best_model_path=best_val_model_path,
    final_model_path=final_model_path,
    patience=patience,
    resume=False
)


# %%
plot_losses(train_losses, val_losses, os.path.join(models_path, 'losses.png'))

# %%
checkpoint = torch.load(best_val_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.to(device)

# Evaluate on the test set
test_acc, test_loss, test_embeddings = evaluate_sinkhorn(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

inference_time = 0
# use the model to predict the matching on a test graph
for i, (g1_out, g2_perm, gt_perm) in enumerate(test_list):
    start_time = time.time()
    result = predict_matching_matrix(model, g1_out, g2_perm, use_hungarian=True)
    end_time = time.time()
    inference_time += end_time - start_time
    errors = (result != gt_perm.to(result.device)).sum().item()
    if errors > 0:
        print(f"Graph {i}: Errors found: {errors}")

print(f"Inference time: {inference_time/len(test_list):.6f} seconds")

# %%
g1_out, g2_perm, gt_perm = test_list[3]
result = predict_matching_matrix(model, g1_out, g2_perm)

plot_two_graphs_with_matching(
    [g1_out, g2_perm],
    gt_perm=gt_perm,
    pred_perm=result,
    original_graphs=original_graphs,
    noise_graphs=noise_graphs,
    viz_rooms=True,
    viz_ws=True,
    match_display="wrong",
)

# %% [markdown]
# ## Room Dropout


