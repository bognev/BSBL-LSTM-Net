import torch
import torchvision
import torch.nn as nn

class SBL_LSTM_STACK(nn.Module)
    def __init__(self, input, prev_hs, prev_cs, input_size, rnn_size, num_layers):
        super(SBL_LSTM_STACK, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.next_hs = {}
        self.next_cs = {}
        self.all_layers = {}
        self.x = {}
        for L in range(self.num_layers):
            if L == 1:
                x = input
                x_size = self.input_size
            else:
                x = self.next_hs[L - 1]
                x_size = self.rnn_size
            self.l_i2h = nn.Linear(x_size, 4 * rnn_size)
            self.l_h2h = nn.Linear(rnn_size, 4 * rnn_size)
            self.l_bn = nn.BatchNorm1d(4 * rnn_size)
            self.prev_c = prev_cs[L]
            self.prev_h = prev_hs[L]



    def forwarf(self, input, prev_hs, prev_cs):





