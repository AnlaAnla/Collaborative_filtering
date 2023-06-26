import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embeddings(user_ids)
        item_embeddings = self.item_embeddings(item_ids)
        user_biases = self.user_biases(user_ids)
        item_biases = self.item_biases(item_ids)
        dot = torch.sum(user_embeddings * item_embeddings, dim=1)
        return dot + user_biases.squeeze() + item_biases.squeeze()
