import torch
import torch.optim as optim
import networkx as nx
import pickle
import numpy as np
import os
import sys
from model import GraphSAGE

input_graph     = 'models/gastro_graph.gpickle'
output_dir      = 'models'
mapping_file    = os.path.join(output_dir, 'node2idx.pkl')
model_file      = os.path.join(output_dir, 'graphsage_model.pth')
embeddings_file = os.path.join(output_dir, 'node_embeddings.npy')

EPOCHS     = 100
LR         = 0.01
HIDDEN_DIM = 128
OUTPUT_DIM = 64
SEED       = 42

def load_graph():
    print(f"Loading graph from {input_graph}...")
    with open(input_graph, 'rb') as f:
        G = pickle.load(f)
    return G

def prepare_data(G):
    print("Preparing adjacency matrix and mappings...")
    nodes    = list(G.nodes())
    node2idx = {n: i for i, n in enumerate(nodes)}

    with open(mapping_file, 'wb') as f:
        pickle.dump({'node2idx': node2idx, 'nodes': nodes}, f)

    num_nodes = len(nodes)
    adj = torch.zeros((num_nodes, num_nodes))

    edges_count = 0
    for u, v, d in G.edges(data=True):
        i, j = node2idx[u], node2idx[v]
        w = d.get('weight', 1.0)
        adj[i][j] = w
        adj[j][i] = w
        edges_count += 1

    print(f"Adjacency matrix built. {edges_count} edges.")

    row_sum = torch.sum(adj, dim=1)
    row_sum[row_sum == 0] = 1
    r_inv = 1.0 / row_sum
    adj_norm = adj * r_inv.unsqueeze(1)

    return adj_norm, node2idx, list(G.edges())

def get_loss(z, edges, node2idx, batch_size=1024):
    num_nodes = z.shape[0]

    indices  = np.random.choice(len(edges), batch_size, replace=True)
    pos_edges = [edges[i] for i in indices]

    u_pos_idx = [node2idx[u] for u, v in pos_edges]
    v_pos_idx = [node2idx[v] for u, v in pos_edges]

    u_pos = z[torch.tensor(u_pos_idx)]
    v_pos = z[torch.tensor(v_pos_idx)]

    pos_score = torch.sum(u_pos * v_pos, dim=1)

    u_neg_idx = np.random.randint(0, num_nodes, batch_size)
    v_neg_idx = np.random.randint(0, num_nodes, batch_size)

    u_neg = z[torch.tensor(u_neg_idx)]
    v_neg = z[torch.tensor(v_neg_idx)]

    neg_score = torch.sum(u_neg * v_neg, dim=1)

    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-6).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-6).mean()

    return pos_loss + neg_loss

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    G = load_graph()
    adj_norm, node2idx, all_edges = prepare_data(G)

    num_nodes = len(G.nodes())
    model     = GraphSAGE(num_nodes, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        z    = model(adj_norm)
        loss = get_loss(z, all_edges, node2idx, batch_size=2048)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    print("Training complete.")

    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    model.eval()
    with torch.no_grad():
        final_z = model(adj_norm)
        np.save(embeddings_file, final_z.numpy())
    print(f"Embeddings saved to {embeddings_file}")

if __name__ == "__main__":
    train()
