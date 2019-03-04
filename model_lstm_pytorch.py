import torch
import torchvision
import torch.nn as nn

class SBL_LSTM_STACK(nn.Module)
    def __init__(self, input, prev_hs, prev_cs, input_size, rnn_size, num_layers):
        super(SBL_LSTM_STACK, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.next_hs = []
        self.next_cs = []
        self.all_layers = []
        self.x = []
        self.x_size = []
        self.prev_c = []
        self.prev_h = []
        for L in range(self.num_layers):
            if L == 1:
                self.x.append(input)
                self.x_size.append(self.input_size)
            else:
                self.x.append(self.next_hs[L - 1])
                self.x_size.append(self.rnn_size)

            self.l_i2h.append(nn.Linear(self.x_size[L], 4 * self.rnn_size))
            self.l_h2h.append(nn.Linear(self.rnn_size, 4 * self.rnn_size))
            self.l_bn.append(nn.BatchNorm1d(4 * self.rnn_size))
            self.prev_c.append(prev_cs[L])
            self.prev_h.append(prev_hs[L])


    def forward(self, input, prev_hs, prev_cs):
        for L in range(self.num_layers):
            if L == 1:
                i2h = self.l_i2h(self.x[L])
                h2h = self.l_h2h(self.prev_h[L])
                all_sums = torch.add(torch.cat(i2h, h2h))





