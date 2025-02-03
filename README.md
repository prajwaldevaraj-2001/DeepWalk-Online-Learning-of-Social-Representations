# 🧠 DeepWalk: Online Learning of Social Representations

This repository contains an implementation of **DeepWalk**, a fundamental graph representation learning algorithm that learns node embeddings using **random walks and Skip-gram models**.

📄 **Paper:** [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652)  
📂 **GitHub Repository:** [DeepWalk](https://github.com/prajwaldevaraj-2001/DeepWalk-Online-Learning-of-Social-Representations)  

---

## 🚀 Overview

DeepWalk is an **unsupervised learning technique** that learns node embeddings in a graph by simulating **random walks** and applying a **Skip-gram neural network model** (similar to Word2Vec). The learned embeddings capture the structural similarity of nodes, making them useful for various machine learning tasks like **node classification, clustering, and link prediction**.

## 🔹 Key Highlights
- Learns **continuous vector representations** of nodes in a graph.
- Uses **random walks** to explore local graph neighborhoods.
- Trains embeddings using **Skip-gram (Word2Vec)**.
- Scalable and effective for large-scale networks.

---

## 🛠️ Implementation Details

## 📌 Technologies Used
- **Python** 🐍
- **NetworkX** – Graph processing
- **NumPy** – Mathematical operations
- **Gensim** – Word2Vec for learning embeddings
- **Matplotlib/Seaborn** – Visualization


## ⚙️ Installation & Setup

1. 🔹 Clone the Repository
git clone https://github.com/prajwaldevaraj-2001/DeepWalk-Online-Learning-of-Social-Representations.git
cd DeepWalk-Online-Learning-of-Social-Representations

2. 🔹 Install Dependencies
pip install -r requirements.txt


## 🔧 Usage
📌 Step 1: Prepare the Graph  
- Load a graph using NetworkX.
- Convert real-world datasets (e.g., social networks) into graph structures.
📌 Step 2: Generate Random Walks
- from random_walks import generate_walks
- walks = generate_walks(graph, num_walks=10, walk_length=40)
📌 Step 3: Train Node Embeddings
- from train_embeddings import train_deepwalk_model
- embeddings = train_deepwalk_model(walks, dimensions=128, window_size=5)
📌 Step 4: Visualize Embeddings (Optional)
- from visualize_embeddings import plot_embeddings
- plot_embeddings(embeddings, graph)

## 📊 Applications
DeepWalk can be used in various graph-based machine learning tasks, such as:
- 🔗 Link Prediction – Predict missing/future edges in networks.
- 🔍 Node Classification – Assign labels to nodes based on their embeddings.
- 📏 Graph Clustering – Identify communities in social networks.
- 📡 Recommender Systems – Improve recommendation accuracy using graph relationships.

## 🚀 Future Enhancements
- ✅ Implement Hierarchical Softmax for faster training.
- ✅ Support for GraphSAGE and node2vec for comparison.
- ✅ Extend to heterogeneous graphs with multiple node types.
- ✅ Integrate PyTorch/TF for deep learning-based embeddings.

## 📜 Citation
If you find this work useful, please cite:
@inproceedings{perozzi2014deepwalk,
  title={DeepWalk: Online learning of social representations},
  author={Perozzi, Bryan and Al-Rfou, Rami and Skiena, Steven},
  booktitle={Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={701–710},
  year={2014}
}

## 📂 Project Structure

```plaintext
DeepWalk/
│
├── deepwalk.py                 # Main implementation of DeepWalk
├── random_walks.py              # Generates random walks on a given graph
├── train_embeddings.py          # Trains node embeddings using Skip-gram
├── visualize_embeddings.py      # Plots the learned embeddings
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
└── datasets/                    # Example graph datasets (if applicable)
