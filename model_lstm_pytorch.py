import torch
import torchvision
import torch.nn as nn


class SblLstmStack(nn.Module):

    def __init__(self, input, prev_hs, prev_cs, input_size, rnn_size, num_layers):
        super(SblLstmStack, self).__init__()

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
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
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
            i2h = self.l_i2h(self.x[L])
            h2h = self.l_h2h(self.prev_h[L])
            all_sums = i2h + h2h
            (n1, n2, n3, n4) = torch.split(all_sums, self.rnn_size, dim=1)  # it should return 4 tensors
            in_gate = self.sigmoid(n1)
            forget_gate = self.sigmoid(n3)
            out_gate = self.sigmoid(n3)
            in_transform = self.tanh(n4)
            next_c1 = forget_gate*self.prev_c[L]
            next_c2 = in_gate*in_transform
            next_c = next_c1 + next_c2
            next_h = out_gate * self.tanh(next_c)
            self.next_hs[L] = next_h
            self.next_cs[L] = next_c
        return self.next_hs, self.next_cs










