# GNN_Notebooks

A collection of Jupyter notebooks and scripts for learning and experimenting with **Graph Neural Networks (GNNs)** and graph-related algorithms.  
The repository includes both introductory material and more advanced experiments such as **graph matching using GNN-based models**.

> **Goal**: provide practical examples, experiments, and small pipelines for typical graph-related tasks like node classification, graph-level regression, and graph matching.

---

## Repository Structure

- **Graph Basics and Algorithms.ipynb**  
  Introduction to graph theory and basic algorithms using Python and NetworkX. Covers graph representations, traversal, and metrics.

- **Node Classification.ipynb**  
  Example notebook demonstrating node classification using GNNs. Includes data loading, preprocessing, model definition, training, and evaluation.

- **Graph Level Regression.ipynb**  
  Experiments focused on predicting graph-level targets. Explores GNN-based regression pipelines.

- **graph_matching/**  
  Contains notebooks and scripts dedicated to **graph matching** tasks using **Graph Attention Networks (GAT)**.  
  Inside this folder you will find:
  - Implementations and experiments for graph matching pipelines.
  - Code for computing similarity/affinity matrices and visualizing matching results.
  - A dedicated `requirements_graph_note.txt` file listing the dependencies required for this environment.

- **requirements_graph_note.txt**  
  Top-level dependencies required to reproduce most experiments in the root notebooks.

- **LICENSE**  
  This project is licensed under GPL-3.0.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mgiorgi13/GNN_Notebooks.git
cd GNN_Notebooks
```

### 2. Set up the general GNN environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows

pip install --upgrade pip
pip install -r requirements_graph_note.txt
```

### 3. Set up Graph Matching environment

```bash
cd graph_matching

python -m venv venv_graph_matching
source venv_graph_matching/bin/activate        # macOS/Linux
venv_graph_matching\Scripts\activate.bat       # Windows

pip install --upgrade pip
pip install -r requirements_graph_note.txt
```
