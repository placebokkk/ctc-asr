import torch
from torch import nn, autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_


class Net(nn.Module):
    def __init__(self, paras):
        super(Net, self).__init__()
        input_dim = paras.get('input_dim')
        hidden_dim = paras.get('hidden_dim')
        num_layers = paras.get('num_layers')
        vocab_size = paras.get('output_dim')
        print(input_dim, hidden_dim, vocab_size)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, xs):
        self.hs, _ = self.lstm(xs)
        self.hs_d = F.dropout(self.hs, 0.9, self.training)
        return self.fc(self.hs_d)
