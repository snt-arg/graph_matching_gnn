"""
Rotation+translation-invariant edge features for the `adj_glob`-style graph-matching
models, per arXiv:2409.11972 (Millan-Romera et al.) Sec III-B, Eq. 1-2 -- with `phi_ij`
corrected to an explicit local-frame construction, signed (cos, sin) angle encodings, and
two additions that give room nodes real relative geometry:

  Mechanism 1  room->room edges are anchored in the frame of a wall the two rooms share
               (rooms have no intrinsic frame of their own, and none can be fabricated:
               a rectangular room's 4 wall normals average to exactly zero, and PCA fails
               identically -- no continuous canonical-frame assignment survives symmetric
               configurations). Emitted as PARALLEL edges, one per accepted shared wall,
               so nothing depends on an argmin that can flip between the two graphs.
  Mechanism 2  cross-room ws-ws edges between the two surfaces of one physical wall, so
               inter-room geometry can travel r1 -> ws -> ws -> r2 on normal-derived
               angles (the most partial-observation-robust channel measured: alpha drifts
               0.10 deg between Prior and Online) instead of through the room nodes.

Geometry alone leaves two of the three possible signatures ambiguous, so every row also
carries a 4-way edge TYPE one-hot (columns 5-8). See the block above the type constants below
for the measured signature table, and for the one distinction that is deliberately NOT given a
column because it is already readable from the row.

A row is expressed in ONE endpoint's frame and is aggregated at THAT SAME endpoint -- frame
owner == receiver, see build_edge_index_and_attr. So each room-ws pair contributes two rows:
the ROOM-FRAMED one the room receives (d only -- a room has no frame for a bearing) and the
WALL-FRAMED one the wall receives (d + bearing to the room). Arrow notation is avoided
throughout because in a GNN an arrow reads as message flow, which points the other way.

NOT done here, and deliberately: giving the ROOM-FRAMED row a bearing by borrowing the
neighbouring wall's frame. It was measured on all three real environments and is degenerate --
walls face inward, so the bearing from the room centre to a wall in that wall's own frame is
~180 deg for essentially every wall (47_basement: mean 175.7 deg, std 9.7, 82% within 15 deg
of 180). It would add a channel and no information. Do not re-propose it.

Related, and worth knowing before expecting much from the wall-framed bearing: it is nearly
constant on real data for the same reason. Real cos(phi) on those rows averages +0.966
(std 0.077, 80% above 0.95) because every real room is rectangular so its centre sits along
the wall's inward normal. Synthetic is far more varied (+0.604, std 0.547) since MSD rooms are
irregular -- a train/deploy distribution gap on a channel carried by ~20% of edges.

Shared by `nx_to_pyg_data_preserve_order` (pgm_training_adj_glob_edgefeat.py,
dataset_gen_adj_glob_edgefeat.py) and `_real_nx_to_pyg` (real-scene validation, both
pgm_training_adj_glob_edgefeat.py and optimization_adj_glob_edgefeat.py) so the feature
definition, the wall-pairing rule and the edge-direction-swap rule all live in one place.
"""
import numpy as np
import torch

# [d_ij, cos_phi, sin_phi, cos_alpha, sin_alpha, <4-way type one-hot>]
EDGE_ATTR_DIM = 9

_EPS = 1e-6

# ---------------------------------------------------------------------------------------
# Edge type one-hot (columns 5-8).
#
# Every row falls into one of three geometry signatures. Two of the three are AMBIGUOUS --
# they map to more than one relation -- and the one-hot exists exactly to split them
# (300 synthetic adj graphs, with the room-room fallback dropped and length_w removed):
#
#   phi!=0  alpha!=0    ws-ws intra=45114,  ws-ws inter=7784    AMBIGUOUS
#   phi!=0  alpha==0    wall-framed=20223,  room-room=6101      AMBIGUOUS
#   phi==0  alpha==0    room-framed=20223                       unique
#
# "room-framed" / "wall-framed" are the two halves of a ROOM_WS pair, named by whose frame the
# row is in -- equivalently, by which node RECEIVES it (see build_edge_index_and_attr).
#
# Note how the schema arrived here, because it is not obvious:
#
#   * Dropping the room-room fallback EMPTIED the `phi==0, alpha==0` bucket of everything
#     except room-framed rows, so that bucket is unambiguous on its own. It used to be the worst
#     one: 20.6% of rows were a bit-identical [d, 0,0,0,0].
#   * Dropping length_w CREATED the `phi!=0, alpha==0` collision. room-room and wall-framed were
#     previously separated by len_w>0 on the room-room side. They no longer are.
#
# So the label count stays at four and the number of jobs it does stays at two -- but one job
# was handed to it by the length_w removal. Do NOT conclude from "room-room and wall-framed look
# alike" that one of them is redundant; that resemblance is exactly what the one-hot fixes.
#
# Endpoint node types are NOT a substitute. self.mlp runs BEFORE encode(), so no GAT layer
# ever sees the raw node one-hot, and from hop 2 on x_i/x_j are learned mixtures --
# reconstruction-through-depth is the failure mode this whole schema exists to avoid. The
# edge one-hot instead reaches GATv2Conv's attention pre-activation through lin_edge, so it
# changes what ATTENTION can condition on, not merely what the message carries.
#
# One distinction is deliberately NOT encoded, because a column is only added for information
# that is not already in the row:
#
#   * which half of a ROOM_WS pair a row is. Within ROOM_WS, phi==(0,0) is exactly the
#     room-framed row (received by the room) and phi!=(0,0) is exactly the wall-framed row
#     (received by the wall): 27291/27291 synthetic and 128/128 real, 0 violations. This is the
#     principle that also killed has_phi/has_alpha -- a (cos,sin) unit pair encodes
#     "undefined" as the zero vector, a value the valid range cannot produce, and
#     relu(c)+relu(-c)+relu(s)+relu(-s) reads it off the raw row in the first layer with a
#     hard gap between 0 and 1.
#
# There is likewise no is_wall_anchored flag, and now nothing for it to mean: an unanchored
# room-room edge is not emitted at all, so every ROOM_ROOM row has phi defined by construction.
#
# WS_WS_INTRA/WS_WS_INTER is assigned by PROVENANCE ("already in the topology" vs "emitted by
# mechanism 2"), not a room-membership lookup -- see build_edge_index_and_attr for why, and for
# the one measured case where that makes the constant's name inexact. It is kept even though
# cos_alpha's VALUE usually separates them (inter mean -0.997, 99.3% below -0.9; intra mean
# +0.024, 2.7%) -- "usually" is precisely what the label removes.
# ---------------------------------------------------------------------------------------
ROOM_ROOM = 0
ROOM_WS = 1
WS_WS_INTRA = 2
WS_WS_INTER = 3
_N_EDGE_TYPES = 4

EDGE_TYPE_NAMES = ("room_room", "room_ws", "ws_ws_intra", "ws_ws_inter")
EDGE_TYPE_COL0 = 5                  # type k occupies column EDGE_TYPE_COL0 + k


def _type_onehot(edge_kind):
    onehot = [0.0] * _N_EDGE_TYPES
    onehot[edge_kind] = 1.0
    return onehot

# ---------------------------------------------------------------------------------------
# Wall-pairing rule -- PORTED VERBATIM, do not retune independently.
#
# Source of truth: graph_matching/graph_matching/graph_matching_node.py, lines ~929-975
# (repo `graph_matching`, the live ROS node). That node uses exactly these three checks to
# decide that two ws are the two opposing faces of one physical wall, and it is what CREATES
# the room->room edges in the real Online graph. Reusing it here keeps the training-time
# feature builder and the deployment graph builder in agreement about which walls pair --
# if they disagree, the model sees one topology in training and a different one at inference.
#
# It cannot simply be imported: graph_matching_node.py does `import rclpy` at module top.
#
# !! If WALL_NORMAL_DIST_THRESHOLD is ever retuned in graph_matching_node.py, it MUST be
# !! changed here too, or the training features silently desynchronize from deployment.
#
# Check 2 deserves a note: projecting both centers onto the normal axis measures wall
# thickness ONLY, and is invariant to the wall midpoint sliding along its own length --
# which is exactly how partially-observed walls degrade. A plain segment-to-segment distance
# would conflate that slide with genuine separation.
# ---------------------------------------------------------------------------------------
WALL_NORMAL_DIST_THRESHOLD = 0.25   # meters -- max separation along the wall normal
TANGENT_OVERLAP_MIN_RATIO = 0.01    # >= 1% overlap of the shorter segment


def _xy(value, default=(0.0, 0.0)):
    if value is None:
        return np.asarray(default, dtype=float)
    return np.asarray(value, dtype=float).ravel()[:2]


def _center(graph, n):
    return _xy(graph.nodes[n].get('center'))


def _normal(graph, n):
    return _xy(graph.nodes[n].get('normal'))


def _limits(graph, n):
    """Segment endpoints as two 2-vectors, or None when the node carries no extent."""
    lims = graph.nodes[n].get('limits')
    if lims is None:
        return None
    arr = np.asarray(lims, dtype=float)
    if arr.shape[0] < 2:
        return None
    return _xy(arr[0]), _xy(arr[1])


def _length(graph, n):
    """Physical length, or 0.0 for nodes that have none.

    Synthetic graphs omit the 'length' key on rooms; real scenes instead carry an explicit
    -1.0 sentinel there. Both mean "no length" and -1 is never a valid length.
    """
    value = graph.nodes[n].get('length', -1)
    if value is None:
        return 0.0
    value = float(value)
    return value if value != -1.0 else 0.0


def _tangent_overlap_ratio(lims_a, lims_b, tangent_2d):
    """Overlap / min-segment-length for projections of two segments onto tangent_2d.

    Ported from graph_matching_node.py's helper of the same name.
    """
    t = _xy(tangent_2d)
    pa = sorted([float(np.dot(_xy(ep), t)) for ep in lims_a])
    pb = sorted([float(np.dot(_xy(ep), t)) for ep in lims_b])
    overlap = max(0.0, min(pa[1], pb[1]) - max(pa[0], pb[0]))
    min_len = min(pa[1] - pa[0], pb[1] - pb[0])
    if min_len < _EPS:
        return 0.0
    return overlap / min_len


def same_physical_wall(graph, a, b):
    """True when ws nodes `a` and `b` are the two opposing faces of one physical wall.

    The three checks are graph_matching_node.py's, in its order:
      1. normals antiparallel -- dot > 0.0 means the same half-space, so not opposing faces
      2. centers projected onto the normal axis are within WALL_NORMAL_DIST_THRESHOLD
      3. segment projections onto the tangent overlap by >= TANGENT_OVERLAP_MIN_RATIO
    """
    n_a, n_b = _normal(graph, a), _normal(graph, b)
    mag_a, mag_b = np.linalg.norm(n_a), np.linalg.norm(n_b)
    if mag_a < _EPS or mag_b < _EPS:
        return False
    n_a_hat = n_a / mag_a
    if float(np.dot(n_a_hat, n_b / mag_b)) > 0.0:
        return False                                                    # Check 1

    proj_a = float(np.dot(_center(graph, a), n_a_hat))
    proj_b = float(np.dot(_center(graph, b), n_a_hat))
    if abs(proj_a - proj_b) > WALL_NORMAL_DIST_THRESHOLD:
        return False                                                    # Check 2

    lims_a, lims_b = _limits(graph, a), _limits(graph, b)
    if lims_a is None or lims_b is None:
        return False                                                    # matches the ROS node
    tangent = np.array([-n_a_hat[1], n_a_hat[0]])
    return _tangent_overlap_ratio(lims_a, lims_b, tangent) >= TANGENT_OVERLAP_MIN_RATIO   # Check 3


def room_membership(graph):
    """(ws -> room, room -> [ws]) derived from the room/ws edges.

    Synthetic graphs label these `ws_belongs_room`; real scenes carry NO edge `type` at all
    (every edge attribute is None), so membership is inferred from the endpoint node types --
    the same convention the training scripts already use.
    """
    ws_room = {}
    room_ws = {}
    for u, v in graph.edges():
        t_u = graph.nodes[u].get('type')
        t_v = graph.nodes[v].get('type')
        if t_u == 'room' and t_v == 'ws':
            room, ws = u, v
        elif t_v == 'room' and t_u == 'ws':
            room, ws = v, u
        else:
            continue
        ws_room[ws] = room
        bucket = room_ws.setdefault(room, [])
        if ws not in bucket:
            bucket.append(ws)
    return ws_room, room_ws


def compute_local_frame_edge_features(graph, u, v, edge_kind, frame_normal=None):
    """Invariant descriptor from i=u to j=v, in i's frame (or in `frame_normal` if given).

    Columns: [d_ij, cos_phi, sin_phi, cos_alpha, sin_alpha] + type one-hot(4)

    d_ij      Euclidean distance between centroids -- always defined.
    phi_ij    bearing from i to j expressed in the frame, as (cos, sin). Storing the signed
              pair rather than a wrapped scalar avoids the discontinuity at +-pi.
    alpha_ij  angle between normal_i and normal_j, as (cos, sin) -- SIGNED, so a neighbour
              rotated +45 deg is distinguishable from one at -45 deg. The previous
              arccos form was unsigned on [0, pi] and threw away chirality on every ws-ws edge.
              There is deliberately NO length_w channel. The anchoring wall's length behaves
              well in training and not at deployment: over synthetic ws-dropout its
              Prior-vs-Online correlation is 0.990 / 0.976 / 0.963 at 15 / 35 / 65% dropout
              (mean rel. error 3.0 / 8.0 / 10.7%), but on the real GT-matched room-room pairs
              it is -0.190 with 29.6% error. d_ij on the SAME pairs is corr 1.000 / 0.0% error
              in every domain, real included. A channel the model learns to trust in training
              and that carries no signal at inference is worse than no channel. Mechanism: the
              SET of walls satisfying the pairing rule changes on 8 / 21 / 27% of pairs at those
              dropout levels, so length_w is not even measuring the same wall in the two graphs;
              and length is the quantity partial observation damages most.
              Caveat for whoever revisits this: the real-side figure is n=3. There are only 22
              real room-room pairs, 18 anchored, and requiring both endpoints GT-matched AND
              anchored in both graphs leaves three. The synthetic evidence says the opposite.
    type      4-way one-hot naming the relation: ROOM_ROOM / ROOM_WS / WS_WS_INTRA /
              WS_WS_INTER. `edge_kind` is REQUIRED -- there is no sensible default, and the
              caller always knows it from where in build_edge_index_and_attr it is emitting.
              See the signature table above the constants for what it buys and, just as
              importantly, for the one distinction it deliberately does NOT encode
              (which half of a ROOM_WS pair a row is). That is already
              readable from phi being the zero vector rather than a unit vector, which is
              also why there are no has_phi / has_alpha flags: an angle stored as a
              (cos, sin) UNIT pair encodes "undefined" as (0, 0), a value the valid range
              cannot produce, and a 4-unit ReLU after edge_proj recovers it at every
              edge_hidden_dim in the search space {8, 16, 32}.
              (has_phi/has_alpha were inherited from an older schema that stored phi as a
              scalar in [-pi, pi], where 0.0 IS a legitimate bearing -- ~6% of edges have it
              -- so a flag was genuinely needed then. The justification did not survive the
              change to (cos, sin).)

    d_ij/phi_ij were validated against real ground-truth-matched edges in the 47_basement
    scan (corr 0.92, median error 1.5 deg); alpha_ij mean abs error 0.10 deg (near-exact).
    """
    c_u, c_v = _center(graph, u), _center(graph, v)
    n_u, n_v = _normal(graph, u), _normal(graph, v)
    disp = c_v - c_u
    d_ij = float(np.linalg.norm(disp))

    frame = _xy(frame_normal) if frame_normal is not None else n_u
    frame_mag = np.linalg.norm(frame)
    if frame_mag > _EPS and d_ij > _EPS:
        n_hat = frame / frame_mag
        t_hat = np.array([-n_hat[1], n_hat[0]])          # CCW perpendicular
        cos_phi = float(np.dot(disp, n_hat)) / d_ij
        sin_phi = float(np.dot(disp, t_hat)) / d_ij
    else:
        cos_phi = sin_phi = 0.0          # zero vector == "no frame"; a unit pair cannot be (0,0)

    mag_u, mag_v = np.linalg.norm(n_u), np.linalg.norm(n_v)
    if mag_u > _EPS and mag_v > _EPS:
        n_u_hat, n_v_hat = n_u / mag_u, n_v / mag_v
        cos_alpha = float(np.dot(n_u_hat, n_v_hat))
        sin_alpha = float(n_u_hat[0] * n_v_hat[1] - n_u_hat[1] * n_v_hat[0])
    else:
        cos_alpha = sin_alpha = 0.0      # same convention: zero vector == "not both oriented"

    return [d_ij, cos_phi, sin_phi, cos_alpha, sin_alpha] + _type_onehot(edge_kind)


def shared_wall_candidates(graph, r1, r2, room_ws):
    """Walls of r1 that form one physical wall with some wall of r2.

    Every accepted wall becomes its own parallel room->room edge rather than being reduced to
    a single argmin pick: on synthetic clean-vs-degraded pairs an argmin selects a DIFFERENT
    wall 20-26% of the time, and when it does the resulting frame is ~90 deg wrong. Emitting
    all of them removes the selection step, so there is nothing to flip.
    """
    walls_2 = room_ws.get(r2, ())
    return [a for a in room_ws.get(r1, ())
            if any(same_physical_wall(graph, a, b) for b in walls_2)]


def cross_room_surface_pairs(graph, ws_room, room_ws):
    """Unordered ws pairs in DIFFERENT rooms that are the two faces of one physical wall.

    These edges exist nowhere in the `adj` topology -- `ws_same_wall` is filtered out of it,
    and cross-room ws-ws edge count is zero in both synthetic and real graphs -- so they are
    derived here rather than read off an edge label.
    """
    walls = list(ws_room.keys())
    pairs = []
    for idx, a in enumerate(walls):
        room_a = ws_room[a]
        n_a = _normal(graph, a)
        mag_a = np.linalg.norm(n_a)
        if mag_a < _EPS:
            continue
        n_a_hat = n_a / mag_a
        for b in walls[idx + 1:]:
            if ws_room[b] == room_a:
                continue
            n_b = _normal(graph, b)
            mag_b = np.linalg.norm(n_b)
            # cheap rejection before the full three-check rule
            if mag_b < _EPS or float(np.dot(n_a_hat, n_b / mag_b)) > 0.0:
                continue
            if same_physical_wall(graph, a, b):
                pairs.append((a, b))
    return pairs


def build_edge_index_and_attr(graph, id_map):
    """Build (edge_index, edge_attr) for a networkx DiGraph.

    DIRECTION SWAP: node i's update must see e_ij (computed in i's own frame), so for the
    descriptor from i to j the PyG column is (source=id_map[j], target=id_map[i]) -- matching
    PyG's MessagePassing convention that messages come from x[source] and are aggregated at
    x[target].

    THE CONSEQUENCE THAT KEEPS CONFUSING READERS, so state it plainly:
    **the frame owner and the receiver are the SAME node.** A descriptor computed in i's frame
    is aggregated AT i. It is not sent to j. So a row built "from the room to its wall" is the
    row THE ROOM RECEIVES, and it is d-only because a room has no frame to express a bearing
    in. The row the WALL receives is the one built in the wall's frame, and it carries d plus
    the bearing to the room. Verified on 47_basement/Prior: 28 room-ws rows land on a room and
    every one has phi == (0,0); 28 land on a ws and every one has phi defined.

    This is why an arrow notation is avoided below -- in a GNN an arrow reads as message flow,
    which here points the opposite way from the frame. Rows are named by their FRAME instead
    (room-framed / wall-framed), which is also their receiver. The input graph is reciprocal, so the resulting edge_index *set* is unchanged
    from a naive construction; only which edge_attr row lands on which column differs. This is
    the single easiest thing to get backwards and it fails SILENTLY -- see the directionality
    checks in the smoke test before trusting any training run built on this.

    Topology augmentation (mechanisms 1 and 2) happens HERE rather than in a separate
    transform callers must remember to apply: it only ever adds edges, never nodes, so id_map
    stays valid, and there is then no code path that can build features without them.
    """
    ws_room, room_ws = room_membership(graph)

    cols = []
    rows = []

    def emit(i, j, feat):
        cols.append((id_map[j], id_map[i]))
        rows.append(feat)

    for u, v in graph.edges():
        t_u = graph.nodes[u].get('type')
        t_v = graph.nodes[v].get('type')
        if t_u == 'room' and t_v == 'room':
            # Mechanism 1: one parallel edge per shared wall, each in that wall's frame.
            #
            # shared_wall_candidates gates edge EXISTENCE, not just the frame: a room-room
            # edge with no accepted shared wall is DROPPED, not emitted d-only. Two rooms
            # that share no wall have no relative geometry to express, and at deployment the
            # ROS node's wall rule is what creates room-room edges in the first place, so
            # keeping them would train on a relation inference never sees.
            #
            # These are not doors whose wall got dropped -- rooms are emitted
            # complete-or-absent, and all four unanchored pairs across the real scans have
            # all 4 walls of both rooms present. They are MSD door annotations without
            # supporting geometry. Scale: 10.9% of synthetic room-room pairs (372/3423),
            # 18.2% of real (4/22).
            #
            # Consequence to be aware of: dropping them can SPLIT the graph. 47_basement goes
            # 1 -> 2 connected components in both Prior and Online, and 101 of the 157
            # synthetic graphs (of 300) that have such an edge gain a component. No node is
            # isolated -- a room keeps its four ws_belongs_room edges.
            walls = shared_wall_candidates(graph, u, v, room_ws)
            disp = _center(graph, v) - _center(graph, u)
            for w in walls:
                n_w = _normal(graph, w)
                if float(np.dot(n_w, disp)) < 0.0:
                    n_w = -n_w                           # orient the frame toward r2
                emit(u, v, compute_local_frame_edge_features(
                    graph, u, v, ROOM_ROOM, frame_normal=n_w))
            continue
        if t_u == 'room' or t_v == 'room':
            kind = ROOM_WS                    # one label for BOTH directions -- phi separates
        else:
            # ws-ws edge already present in the graph. The label is PROVENANCE, not a
            # room_ws lookup: "was already in the topology" vs "added by mechanism 2".
            #
            # Provenance is deliberate, not laziness. A room_ws lookup would make the label
            # depend on whether the ROOM node survived, and 38-43% of rooms vanish between
            # Prior and Online -- so the identical wall pair would be labelled intra in
            # Prior and something else in Online, which is precisely the instability the
            # whole invariant schema exists to avoid. Provenance cannot flip that way, and
            # it also sidesteps orphan walls, where both lookups return None and
            # `None == None` would call two walls of two DIFFERENT dropped rooms intra.
            #
            # Caveat, measured, so nobody re-derives it from the constant's name: this is
            # "intra" for 400/400 synthetic graphs and 5 of the 6 real graphs, but CF12
            # Prior carries 6 directed ws-ws edges between walls of different rooms. None
            # of those 6 satisfy same_physical_wall, so mechanism 2 would never emit them
            # and the two labels stay disjoint -- they are simply pre-existing topology
            # that this label calls intra. Small (6 of 594 real rows) and, more to the
            # point, computed identically in training and at deployment.
            kind = WS_WS_INTRA
        emit(u, v, compute_local_frame_edge_features(graph, u, v, kind))

    # Mechanism 2: cross-room ws-ws edges, both directions so the graph stays reciprocal.
    for a, b in cross_room_surface_pairs(graph, ws_room, room_ws):
        if graph.has_edge(a, b) or graph.has_edge(b, a):
            continue                                     # never duplicate an existing edge
        emit(a, b, compute_local_frame_edge_features(graph, a, b, WS_WS_INTER))
        emit(b, a, compute_local_frame_edge_features(graph, b, a, WS_WS_INTER))

    if not cols:
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, EDGE_ATTR_DIM), dtype=torch.float32))

    edge_index = torch.tensor(cols, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(rows, dtype=torch.float32)
    return edge_index, edge_attr


def node_features(graph, node_type_mapping):
    """
    Reduced node feature: [type_onehot(2), length_or_-1(1)]. 'center'/'normal' are
    deliberately dropped -- see module docstring / arXiv:2409.11972 Sec III-B ("the initial
    node embedding is limited to its length").

    Rooms carry no real length and are encoded as -1.0. That is the convention every other
    script in this repo already uses (PGM_class.py:1372, dataset_gen.py:419, optimization.py:607,
    ... -- all `float(d.get('length', -1))` fed straight into the feature vector) and the one
    the deployed adj_glob_65 checkpoint was trained under.

    Do NOT re-encode the sentinel as 0.0. Measured over 1000 synthetic graphs: 3.13% of ws
    lengths are <= 1 mm and 12.47% <= 1 cm (min 8.9e-16 m), so 0 lands inside a dense cluster
    of degenerate short walls, while -1 is outside the valid range entirely. (This collision is
    also part of why the edge schema no longer carries a wall-length channel at all -- see
    compute_local_frame_edge_features.)

    Two spellings of "no length" occur and BOTH must be handled: the key is absent (every
    synthetic room, and 3 of 5 rooms in CF12/Online) or is an explicit -1.0 (rooms in the
    other five real scans, plus the remaining 2 in CF12/Online). This is NOT a clean
    per-source split -- both spellings co-occur inside a single real graph -- so never branch
    on the source. `.get('length', -1)` collapses them to the same -1.0.

    No has_length flag: "has a length" is exactly "type == ws" (0 ws missing a length, 0 rooms
    carrying one, across 1000 synthetic + all 6 real graphs), so it is fully recoverable from
    the type one-hot -- the same redundancy test that kept has_phi/has_alpha out of edge_attr.
    It would not have survived anyway: normalize_data_pairs z-scores every node column, flags
    included, so a {0,1} indicator reaches the model as an arbitrary shifted pair.
    """
    node_ids = list(graph.nodes())
    feats = []
    for n in node_ids:
        d = graph.nodes[n]
        length_val = d.get('length', -1)
        length_val = -1.0 if length_val is None else float(length_val)
        feats.append(torch.tensor(
            node_type_mapping[d['type']] + [length_val],
            dtype=torch.float32,
        ))
    return torch.stack(feats)


# ---------------------------------------------------------------------------------------
# edge_attr normalization
#
# Exactly ONE column is unbounded: d_ij (col 0, spans ~0-15 m). Columns 1-4 are (cos, sin)
# pairs on [-1, 1] and columns 5-8 are a one-hot, so touching either would only destroy their
# meaning. Without this the single edge_proj Linear has to absorb a two-order-of-magnitude
# scale gap between col 0 and the angle channels.
#
# The former length_w channel (old col 5) is gone -- see compute_local_frame_edge_features for
# the drift measurements. Its bespoke "scale but do not mean-shift" handling went with it.
# Because compute_edge_mean_std returns mean 0 / std 1 on every column outside
# EDGE_NORM_COLS, appending the type block was a normalization no-op by construction.
# Contrast the NODE features, where normalize_data_pairs z-scores every column, indicators
# included -- which is one of the reasons a has_length flag could not have survived there.
# ---------------------------------------------------------------------------------------
EDGE_NORM_COLS = (0,)
_EDGE_COL_D = 0


def compute_edge_mean_std(edge_attr_list):
    """Per-column (mean, std) for edge_attr, as identity outside EDGE_NORM_COLS.

    Returns full-width vectors (mean 0 / std 1 on every untouched column) so callers can
    apply them unconditionally with a single elementwise expression.
    """
    all_attr = torch.cat([e for e in edge_attr_list if e.numel()], dim=0)
    mean = torch.zeros(EDGE_ATTR_DIM, dtype=torch.float32)
    std = torch.ones(EDGE_ATTR_DIM, dtype=torch.float32)

    mean[_EDGE_COL_D] = all_attr[:, _EDGE_COL_D].mean()
    std[_EDGE_COL_D] = all_attr[:, _EDGE_COL_D].std()

    std = torch.where(std > _EPS, std, torch.ones_like(std))
    return mean, std


def normalize_edge_attr(edge_attr, mean, std):
    if edge_attr is None or edge_attr.numel() == 0:
        return edge_attr
    return (edge_attr - mean) / (std + 1e-8)
