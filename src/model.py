import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, num_nodes, embed_dim=128, hidden_dim=128, output_dim=64):
        super(GraphSAGE, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.lin1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, adj):
        h0 = self.embedding.weight
        h0_neigh = torch.mm(adj, h0)
        h0_cat = torch.cat([h0, h0_neigh], dim=1)
        h1 = self.relu(self.lin1(h0_cat))
        h1 = self.dropout(h1)
        h1_neigh = torch.mm(adj, h1)
        h1_cat = torch.cat([h1, h1_neigh], dim=1)
        h2 = self.lin2(h1_cat)
        h2 = F.normalize(h2, p=2, dim=1)
        return h2
