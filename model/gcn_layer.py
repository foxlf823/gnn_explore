
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, data):
        super(GCNLayer, self).__init__()

        self.W = torch.nn.Linear(data.HP_hidden_dim, data.HP_hidden_dim)
        self.dropout = nn.Dropout(data.HP_dropout)

        if data.HP_gpu >= 0 and torch.cuda.is_available():
            self.W = self.W.cuda(data.HP_gpu)
            self.dropout = self.dropout.cuda(data.HP_gpu)

    def forward(self, adj, text_embeddings, text_mask):

        denom = adj.sum(2).unsqueeze(2) + 1
        gcn_inputs = text_embeddings

        Ax = adj.bmm(gcn_inputs)
        AxW = self.W(Ax)
        AxW = AxW + self.W(gcn_inputs)  # self loop
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        gAxW = self.dropout(gAxW)

        gcn_outputs = gAxW * text_mask.unsqueeze(-1).float()

        return gcn_outputs