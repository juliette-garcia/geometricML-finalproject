# Predicting Route Redundancy in Supply Chain Networks via Structural and Topological Inductive Biases

This repository contains an analysis of edge-level redundancy prediction in faux supply chain networks using Graph Neural Networks (GNNs). The project compares the effectiveness of structural metrics, topological metrics, and their combination for predicting real redundancy in networked systems.

## Goal

Predict **real** supply chain redundancy at the edge level by comparing:
- **Structural metrics**: Local geometry and path-based properties (degree, clustering, betweenness, etc.)
- **Topological metrics**: Higher-order connectivity and redundancy patterns (cycles, persistence homology, etc.)
- **Combined approach**: Using both structural and topological signals

## Motivation

Traditional supply chain audits rely on simple metrics like degree, number of suppliers, and number of routes. However, these metrics fail to detect **shared failure modes**, meaning that "redundant" edges or nodes may not actually be redundant. This project addresses the **illusion of redundancy** by developing a diffusion-based redundancy metric and training GNNs to predict it.

## Notebook Structure

The `redundancy-prediction.ipynb` notebook is organized as follows:

1. **Introduction & Motivation**: Problem statement and approach
2. **Redundancy Definition**: Mathematical formulation of the redundancy metric
3. **Redundancy Scorer Implementation**: Spectral approximation using heat equation
4. **Synthetic Dataset Creation**: Graph generation and redundancy computation
5. **Dataset Exploration**: Statistical analysis and feature correlations
6. **Model Training**: Training loop for all four GNN variants and performance metrics and visualizations

### About the Models

Four GNN architectures are implemented and compared:

1. **BaselineGNN**: Basic model using only node features and edge weights
2. **StructuralPriorGNN**: Adds structural features (degree statistics, clustering, common neighbors, Jaccard coefficient)
3. **TopologicalPriorGNN**: Adds topological features (cycle closure flags, triangle count, shortest cycle length, Forman curvature)
4. **CombinedGNN**: Uses both structural and topological features

All models use Graph Convolutional Network (GCN) layers with edge-level prediction heads.

### About the Dataset

The project produces and uses a **synthetic dataset** of graphs with edge-level redundancy scores. Below are the used specifics, though they can be changed:

See `synthetic-graphs/README.md` for detailed dataset specifications.

## Usage

1. Generate the synthetic dataset (if not already present):
   ```python
   python generate_synthetic_dataset.py
   ```

2. Open and run `redundancy-prediction.ipynb`:
   - Execute cells sequentially
   - Training will automatically run for all four models
   - Results will be saved to `models/` and `results/` directories

