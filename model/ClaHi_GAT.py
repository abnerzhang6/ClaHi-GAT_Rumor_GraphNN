import re, os
import json
import torch
import torch.nn as nn
import torchtext.vocab as Vocab
from torchtext.data import get_tokenizer
from model import layers
import numpy as np

cache_dir = r'D:\pycharmProjects\PHEME\.vector_cache'

glove = Vocab.GloVe(name='6B', dim=300, cache=cache_dir)


def clean_text(text):
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    return text


class ClaHi_GAT(nn.Module):
    def __init__(self):
        super(ClaHi_GAT, self).__init__()
        self.embedding = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
        self.embedding.weight.data.copy_(glove.vectors)
        self.lstm_encoder = nn.LSTM(input_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                                    batch_first=True,
                                    dropout=0.2)
        self.sigmoid = nn.Sigmoid()
        self.gat = layers.GraphAttentionLayer(in_features=2400, out_features=1200, dropout=0.2, alpha=0.01)
        self.linear = nn.Linear(2400, 2)
        self.softmax = nn.Softmax(dim=1)
        self.params = nn.ParameterDict(
            {
                "W": nn.Parameter(torch.rand(1200, 1)),
                "U": nn.Parameter(torch.rand(1200, 1)),
                "FC1": nn.Parameter(torch.rand(4800, 9600)),
                "FC2": nn.Parameter(torch.rand(9600, 1))
            }
        )

    def gate_module(self, h):
        h_x = h[1:]  # h_x.shape: torch.Size([9, 1200])
        h_c = h[0]
        h_c = torch.unsqueeze(h_c, dim=0)  # h_c.shape: torch.Size([1, 1200])
        h_c = h_c.repeat(h_x.shape[0], 1)  # h_c.shape: torch.Size([9, 1200])
        g = self.sigmoid(torch.mm(h_x, self.params["W"]) + torch.mm(h_c, self.params["U"]))  # g.shape: torch.Size([
        # 9, 1])
        g = g.repeat(1, h_x.shape[0])
        h_tuta = torch.mm(g, h_x) + torch.mm((torch.ones(g.shape) - g), h_c)
        h_tuta = torch.cat((h_c[0].unsqueeze(dim=0), h_tuta), dim=0)  # h_tuta.shape: torch.Size([10, 1200])
        return h_tuta

    def forward(self, input, adj):
        input = input.squeeze(dim=0)
        encode = self.embedding(input)
        encode = encode.view(encode.shape[0], encode.shape[1], -1)
        o, (h_n, c_n) = self.lstm_encoder(encode)
        hs_0 = h_n.transpose(0, 1)  # hs_0.shape == torch.Size([10, 4, 300])

        hs_0 = hs_0.contiguous().view(hs_0.shape[0], -1)  # hs_0.shape: torch.Size([10, 1200])

        hs_tuta0 = self.gate_module(hs_0)
        hs_head0 = torch.concat((hs_0, hs_tuta0), dim=1)  # hs_head0.shape: torch.Size([10, 2400])
        hs_1 = self.gat(hs_head0, adj)  # out.shape: torch.Size([10, 1200])
        hs_1 = hs_1.squeeze(dim=0)

        # ---------------------------------------------------------------------------------------- #
        hs_tuta1 = self.gate_module(hs_1)
        hs_head1 = torch.concat((hs_1, hs_tuta1), dim=1)
        hs_2 = self.gat(hs_head1, adj)  # hs_2.shape: torch.Size([10, 1200])
        hs_2 = hs_2.squeeze(dim=0)

        hc_2 = hs_2[0]
        hc_2 = hc_2.unsqueeze(dim=0)  # hc_2.shape: torch.Size([1, 1200])
        hc_2 = hc_2.repeat(346, 1)  # hc_2.shape: torch.Size([10, 1200])

        hc_3 = torch.cat((hc_2, hs_2, hc_2 * hs_2, hc_2 - hs_2), dim=1)
        hc_3 = torch.tanh(torch.mm(hc_3, self.params["FC1"]))  # hc_3.shape: torch.Size([10, 9600])
        b = torch.tanh(torch.mm(hc_3, self.params["FC2"]))  # b.shape: torch.Size([10, 1])
        ea_out1 = torch.mm(b.t(), hs_2)  # ea_out1.shape: torch.Size([1, 1200])
        ea_out = torch.cat((torch.mean(hs_2, dim=0, keepdim=True), ea_out1), dim=1)
        output = self.softmax(self.linear(ea_out))

        return output
