#!/usr/bin/env python3 (runs on PGM_ENV)
"""
Standalone PGM Inference Script
Runs in its own Python 3.11 environment, called by ROS2 node via subprocess.
"""


import sys
import pickle
import torch
import networkx as nx
import os


# Add PGM path
PGM_PATH = '/root/workspace/src/graph_matching_gnn/graph_matching/graph_matching'
if PGM_PATH not in sys.path:
    sys.path.append(PGM_PATH)
# edge_model.py (the only loader for the edge-feature models) sits next to this script
EDGE_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))
if EDGE_MODEL_PATH not in sys.path:
    sys.path.append(EDGE_MODEL_PATH)


# Model to load — pick one:
#   "ws_room_dropout_noise"               → MatchingModel_GATv2SinkhornTopK  (original, TopK)
#   "ws_room_dropout_noise_inc_BCE"       → MatchingModel_MLPGATv2SinkhornBCE  (MLP + BCE)
#   "ws_room_dropout_noise_inc_BCE_noMLP" → MatchingModel_GATv2Sinkhorn  (no MLP, BCE)
#   "ws_room_dropout_noise_inc_WBCE"      → MatchingModel_MLPGATv2SinkhornWBCE  (MLP + weighted BCE)
#   "adj_glob_65_edgefeat_rooms"          → edge-feature model, loaded through edge_model.EdgeMatcher
#   "adj_no_glob_65_edge"                 → edge-feature model, loaded through edge_model.EdgeMatcher
MODEL = "adj_glob_65_edgefeat_rooms"

# Edge-feature models: node x = [type, length], all geometry in a 9-dim edge_attr that
# edge_model builds from the graph at load time. PartialGraphMatching cannot load them.
_EDGE_MODELS = ("adj_glob_65_edgefeat_rooms", "adj_no_glob_65_edge")

# Map model name → (model_class name, preprocessed_data_subfolder)
_MODEL_CONFIGS = {
    "ws_room_dropout_noise":               ("MatchingModel_GATv2SinkhornTopK",    "ws_room_dropout_noise"),
    "ws_room_dropout_noise_inc_BCE":       ("MatchingModel_MLPGATv2SinkhornBCE",  "ws_room_dropout_noise_inc"),
    "ws_room_dropout_noise_inc_BCE_noMLP": ("MatchingModel_GATv2Sinkhorn",        "ws_room_dropout_noise_inc"),
    "ws_room_dropout_noise_inc_WBCE":      ("MatchingModel_MLPGATv2SinkhornWBCE", "ws_room_dropout_noise_inc"),
}

GNN_PATH = '/root/workspace/src/graph_matching_gnn/GNN'


def load_pgm_model():
    if MODEL not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown MODEL '{MODEL}'. Choose from: "
                         f"{list(_MODEL_CONFIGS.keys()) + list(_EDGE_MODELS)}")

    import PGM_class  # only the node-feature models need it
    class_name, data_subfolder = _MODEL_CONFIGS[MODEL]
    model_class = getattr(PGM_class, class_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PGM using device: {device}, model: {MODEL}")

    model_save_path = os.path.join(GNN_PATH, 'models', "partial_graph_matching", MODEL)
    data_paths = {
        "equal":   os.path.join(GNN_PATH, "preprocessed", "graph_matching", "equal"),
        "partial": os.path.join(GNN_PATH, "preprocessed", "partial_graph_matching", data_subfolder),
    }

    pgm_model = PGM_class.PartialGraphMatching(
        model_class=model_class,
        data_paths=data_paths,
        model_save_path=model_save_path,
        device=device,
        in_dim=7,
    )
    pgm_model.load_best_model()
    print(f"PGM model initialized: {MODEL}")

    return pgm_model


class _Wrapped:
    """EdgeMatcher.match reads `.graph`, the attribute of the dashboard's GraphWrapper."""
    def __init__(self, g):
        self.graph = g


def edge_model_matching(g1, g2):
    """Hungarian 0/1 matrix [|g1|, |g2|] from an edge-feature model, rows/cols in g1/g2 node order."""
    import edge_model
    matcher = edge_model.EdgeMatcher(os.path.join(GNN_PATH, 'models', "partial_graph_matching", MODEL))
    _, ints, g1_nodes, g2_nodes = matcher.match(_Wrapped(g1), _Wrapped(g2))
    assert list(g1_nodes) == list(g1.nodes()) and list(g2_nodes) == list(g2.nodes())
    return torch.as_tensor(ints["perm"])


def run_pgm_inference(graph1, graph2, output_path):
    """Run PGM inference on two graphs and save the matching matrix."""
    """ Args:
            graph1: Path to pickled NetworkX graph 1
            graph2: Path to pickled NetworkX graph 2
            output_path: Path where to save the matching matrix for the two graphs
        """
    try:
        # Load graphs
        with open(graph1, 'rb') as f:
            g1 = pickle.load(f)
        with open(graph2, 'rb') as f:
            g2 = pickle.load(f)


        print(f"Graph 1: {g1.number_of_nodes()} nodes, {g1.number_of_edges()} edges")
        print(f"Graph 2: {g2.number_of_nodes()} nodes, {g2.number_of_edges()} edges")

        # Perform inference
        print("Running PGM inference...")
        if MODEL in _EDGE_MODELS:
            print(f"PGM model: {MODEL} (edge-feature, via edge_model.EdgeMatcher)")
            matching_matrix = edge_model_matching(g1, g2)
        else:
            pgm_model = load_pgm_model()
            matching_matrix = pgm_model.infer_matching(g1, g2, discrete=True)


        # Save result (matching matrix and nodes of the two graphs that correspond to the matrix indices)
        print(f"Saving result to {output_path}")


        result = {
            'matching_matrix': matching_matrix.cpu().numpy(),
            'g1_nodes': list(g1.nodes()),
            'g2_nodes': list(g2.nodes())
        }
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Matching matrix saved to {output_path}")

        print("Inference completed successfully")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pgm_inference_wrapper.py <graph1_path> <graph2_path> <output_path>")
        sys.exit(1)


    g1_path = sys.argv[1]
    g2_path = sys.argv[2]
    output_path = sys.argv[3]


    exit_code = run_pgm_inference(g1_path, g2_path, output_path)
    sys.exit(exit_code)
