# ChebNet-Cora-GNN
This repository contains a Jupyter Notebook that implements ChebNet, a Graph Neural Network (GNN) based on Chebyshev polynomials, for semi-supervised node classification on the Cora citation network dataset.

## Project Overview

This notebook demonstrates the foundational concepts and implementation of ChebNet, one of the influential spectral convolutional GNNs. The goal is to classify research papers (nodes) into their respective academic fields (labels) using their citation network (graph structure) and abstract content (node features).



*   **Cora Dataset Loading:** Efficient loading and preprocessing of the Cora citation dataset, including features, labels, and adjacency matrix.
*   **Graph Laplacian Construction:** Building normalized and scaled graph Laplacians, crucial for spectral graph convolutions.
*   **Chebyshev Polynomials Implementation:** Implementation of Chebyshev polynomial approximation for spectral filters.
*   **ChebNet Layer:** A custom ChebNet layer demonstrating how to aggregate information based on Chebyshev coefficients.
*   **Semi-Supervised Node Classification:** Training a single-layer ChebNet model for semi-supervised node classification, where only a small subset of nodes are labeled for training.
*   **Manual Gradient Descent:** Training loop with manual gradient computation and Adam optimizer.
*   **Influence of K:** Exploration of the `K` parameter (order of Chebyshev polynomials and receptive field size) on model performance.
