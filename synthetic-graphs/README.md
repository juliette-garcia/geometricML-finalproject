# Synthetic Graph Dataset for Redundancy Prediction

This directory contains a synthetic dataset of graphs with edge-level redundancy scores,. In creating the dataset, we tried to make it so that they model different types of failure sensitivity in networked systems such as supply chains, logistics networks, etc.

---

## Graph Generation Specs

### Graph Sizes

Graphs are generated at three fixed sizes to encourage generalization across scale:

- 50 nodes  
- 100 nodes  
- 200 nodes  

### Graph Counts

For each graph size, a fixed number of graphs is generated:

- 50 nodes: 14 graphs  
- 100 nodes: 13 graphs  
- 200 nodes: 12 graphs  

Total graphs: 39

Each graph is generated independently with randomized parameters.

### Graph Generator Families

Each graph is sampled from one of five generator families to ensure diversity in structure, redundancy patterns, and failure modes.

#### 1. Erdős–Rényi (ER)

- Generator: `networkx.erdos_renyi_graph`
- Structure: Random connectivity, low clustering
- Redundancy profile: Minimal inherent redundancy, mostly uniform

#### 2. Barabási–Albert (BA)

- Generator: `networkx.barabasi_albert_graph`
- Structure: Scale-free degree distribution with hubs
- Redundancy profile: High vulnerability at hub-adjacent edges

#### 3. Stochastic Block Model (SBM)

- Generator: `networkx.stochastic_block_model`
- Structure: Community-based with dense intra-block connections
- Redundancy profile: Redundant edges within communities, fragile inter-block links

#### 4. Random Geometric Graph (RGG)

- Generator: `networkx.random_geometric_graph`
- Structure: Spatial proximity graph with many short cycles
- Redundancy profile: Strong topological redundancy via local loops

#### 5. Hybrid Tree + Shortcuts

- Generator: Custom
- Structure: Random tree backbone with added shortcut edges
- Redundancy profile: Clear contrast between bridge edges and redundant shortcuts

### Graph Post-Processing

All generated graphs undergo the following steps:

1. Connectivity  
   Only the largest connected component is retained.

2. Node relabeling  
   Nodes are relabeled to integers `0 .. n-1`.

3. Edge weights  
   Each edge is assigned a positive weight drawn from a log-normal distribution  
   (mean = 0.0, sigma = 0.5), stored as `edge["weight"]`.

Graphs that shrink to fewer than 80% of the target node count after connectivity filtering are discarded.

---

## Node and Edge Features

Each node has a feature vector of length 3:

1. Degree  
2. Local clustering coefficient 
3. Source indicator (1 if source, 0 otherwise). Each graph was generated to have 1-3 random source nodes that act as diffusion origins for redundancy computation.

Each edge has a feature vector of length 2:

1. Edge weight  
2. Bridge indicator (1 if the edge is a bridge in the graph)  

---

## The Database

Each graph is stored as a `torch_geometric.data.Data` object with:

- x: node features `[num_nodes, 3]`  
- edge_index: edge connectivity `[2, num_edges]`  
- edge_attr: edge features `[num_edges, 2]`  
- y: edge-level redundancy labels `[num_edges, 1]`  

Additional attributes:

- generator: string identifier of the graph generator  

Edge ordering in `edge_index`, `edge_attr`, and `y` is consistent.

---

## Loading the Dataset

```python
import torch

train_data = torch.load("synthetic-graphs/train.pt")
val_data = torch.load("synthetic-graphs/val.pt")
test_data = torch.load("synthetic-graphs/test.pt")

graph = train_data[0]
print(graph.x.shape)
print(graph.edge_attr.shape)
print(graph.y.shape)
print(graph.generator)
