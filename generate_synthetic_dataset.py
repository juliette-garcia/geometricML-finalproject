import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import random
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
from tqdm import tqdm

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CONFIG = {
    "graph_sizes": [50, 100, 200],
    "graphs_per_size": {50: 3, 100: 3, 200: 3},
    "avg_degree_range": (3, 12),
    "random_seed": 42,
    "output_dir": "synthetic-graphs",
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
}

# --------------------------------------------------
# Graph generators
# --------------------------------------------------

def erdos_renyi(n, d, seed):
    p = min(d / (n - 1), 0.9)
    return nx.erdos_renyi_graph(n, p, seed=seed)

def barabasi_albert(n, d, seed):
    m = max(1, int(d / 2))
    return nx.barabasi_albert_graph(n, m, seed=seed)

def stochastic_block(n, d, seed):
    rng = np.random.default_rng(seed)
    k = rng.integers(3, 6)
    sizes = rng.multinomial(n, np.ones(k) / k)
    p_in = d / n * 2.5
    p_out = d / n * 0.4
    P = np.full((k, k), p_out)
    np.fill_diagonal(P, p_in)
    return nx.stochastic_block_model(sizes.tolist(), P, seed=seed)

def random_geometric(n, d, seed):
    r = np.sqrt(d / (np.pi * n))
    return nx.random_geometric_graph(n, min(r, 0.5), seed=seed)

def hybrid_tree(n, d, seed):
    G = nx.random_labeled_tree(n, seed=seed)
    target_edges = int(d * n / 2)
    rng = random.Random(seed)
    while G.number_of_edges() < target_edges:
        u, v = rng.sample(range(n), 2)
        G.add_edge(u, v)
    return G

GENERATORS = {
    "ER": erdos_renyi,
    "BA": barabasi_albert,
    "SBM": stochastic_block,
    "RGG": random_geometric,
    "TREE+": hybrid_tree,
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def preprocess_graph(G: nx.Graph, seed: int) -> nx.Graph:
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.lognormal(0.0, 0.5))
    return G

def select_sources(G: nx.Graph, seed: int) -> List[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(list(G.nodes()), rng.randint(1, 3)))

# --------------------------------------------------
# Feature computation
# --------------------------------------------------

def node_features(G: nx.Graph, sources: List[int]) -> torch.Tensor:
    n = G.number_of_nodes()
    deg = dict(G.degree())
    clustering = nx.clustering(G)
    x = torch.zeros((n, 3))
    for i in range(n):
        x[i, 0] = deg[i]
        x[i, 1] = clustering[i]
        x[i, 2] = 1.0 if i in sources else 0.0
    return x

def edge_features(G: nx.Graph) -> torch.Tensor:
    bridges = set(nx.bridges(G))
    feats = []
    for u, v in G.edges():
        feats.append([
            G[u][v]["weight"],
            1.0 if (u, v) in bridges or (v, u) in bridges else 0.0
        ])
    return torch.tensor(feats, dtype=torch.float32)

# --------------------------------------------------
# Redundancy labels
# --------------------------------------------------

def compute_edge_labels(
    G: nx.Graph,
    sources: List[int],
    compute_redundancy_score,
    params: Dict
) -> np.ndarray:
    labels = []
    for u, v in tqdm(G.edges(), desc="  Scoring edges", leave=False):
        try:
            R, _ = compute_redundancy_score(G, sources, [(u, v)], params)
            labels.append(R)
        except Exception:
            labels.append(np.nan)
    return np.array(labels)

# --------------------------------------------------
# PyG conversion
# --------------------------------------------------

def to_pyg(G, sources, edge_labels) -> Data:
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    return Data(
        x=node_features(G, sources),
        edge_index=edge_index,
        edge_attr=edge_features(G),
        y=torch.tensor(edge_labels, dtype=torch.float32).view(-1, 1)
    )

# --------------------------------------------------
# Dataset generation
# --------------------------------------------------

def generate_dataset(compute_redundancy_score):
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    data_list = []
    params = dict(k_eig=20, tau=0.1, lambda_penalty=1000.0, T_max=10.0, dt=0.01)

    for n in CONFIG["graph_sizes"]:
        total = CONFIG["graphs_per_size"][n]
        generator_names = list(GENERATORS.keys())
        random.shuffle(generator_names)

        print(f"\nGenerating graphs of size {n}...")

        for i in tqdm(range(total), desc=f"Graphs n={n}"):
            gen_name = generator_names[i % len(generator_names)]
            gen = GENERATORS[gen_name]

            seed = CONFIG["random_seed"] + i
            d = np.random.uniform(*CONFIG["avg_degree_range"])

            G = gen(n, d, seed)
            G = preprocess_graph(G, seed)

            if G.number_of_nodes() < 0.8 * n:
                continue

            sources = select_sources(G, seed)
            labels = compute_edge_labels(G, sources, compute_redundancy_score, params)

            if np.any(np.isnan(labels)):
                continue

            data = to_pyg(G, sources, labels)
            data.generator = gen_name
            data_list.append(data)

    return data_list

# --------------------------------------------------
# Train / val / test split
# --------------------------------------------------

def split_dataset(data):
    random.shuffle(data)
    n = len(data)
    n_train = int(CONFIG["train_split"] * n)
    n_val = int(CONFIG["val_split"] * n)
    return data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]
