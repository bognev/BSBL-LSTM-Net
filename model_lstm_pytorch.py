import torch
import torchvision
import torch.nn as nn


def passthrough(x, **kwargs):
    return x


class BuildLstmStack(nn.Module):

    def __init__(self, input_size, rnn_size, num_layers):
        super(BuildLstmStack, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.all_layers = []
        self.x_size = []
        self.prev_c = []
        self.prev_h = []
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        #self.prev_cs = prev_cs
        #elf.prev_hs = prev_hs
        l_i2h_lst = []
        for L in range(self.num_layers):
            if L == 0:
                self.x_size = self.input_size
            else:
                self.x_size = self.rnn_size
            l_i2h_lst.append(nn.Linear(self.x_size, 4 * self.rnn_size))

        self.l_i2h = nn.ModuleList(l_i2h_lst)
        self.l_h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
        self.l_bn = nn.BatchNorm1d(4 * self.rnn_size)

    def forward(self, x, prev_hs, prev_cs):
        next_hs = []
        next_cs = []
        for L in range(self.num_layers):
            if L == 0:
                self.x = x
            else:
                self.x = next_hs[L-1]

            #self.l_i2h.append(nn.Linear(self.x_size[L], 4 * self.rnn_size))
            #self.l_h2h.append(nn.Linear(self.rnn_size, 4 * self.rnn_size))
            #self.l_bn.append(nn.BatchNorm1d(4 * self.rnn_size))
            self.prev_c = prev_cs[L]
            self.prev_h = prev_hs[L]

            i2h = self.l_i2h[L](self.x)
            h2h = self.l_h2h(self.prev_h)
            all_sums = i2h + h2h

            (n1, n2, n3, n4) = all_sums.chunk(4, dim=2)  # it should return 4 tensors self.rnn_size

            in_gate = self.sigmoid(n1)
            forget_gate = self.sigmoid(n2)
            out_gate = self.sigmoid(n3)
            in_transform = self.tanh(n4)
            next_c = forget_gate*self.prev_c + in_gate*in_transform
            next_h = out_gate * self.tanh(next_c)
            next_hs.append(next_h)
            next_cs.append(next_c)
        return next_hs, next_cs


class BuildLstmUnrollNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, input_size):
        super(BuildLstmUnrollNet, self).__init__()

        self.num_unroll = num_unroll
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.input_size = input_size
        #self.init_states_input = passthrough

        self.init_hs = []
        self.init_cs = []

        self.out = []
        self.out_states_lst = []
        self.outputs = []
        self.out_states = []
        self.output = []
        self.now_hs = []
        self.now_cs = []
        self.buildlstmstack = BuildLstmStack(self.input_size, self.rnn_size, self.num_layers)

    def forward(self, x, init_states_input):
        init_states = init_states_input.reshape((init_states_input.size(0),self.num_layers * 2, self.rnn_size))

        init_states_lst  = list(init_states.chunk(self.num_layers * 2,1))
        print()
        for i in range(self.num_layers):

            print(i)
            self.init_hs.append(init_states_lst[2*i])
            self.init_cs.append(init_states_lst[2*i+1])

        self.now_hs, self.now_cs = self.init_hs, self.init_cs

        for i in range(self.num_unroll):
            self.now_hs, self.now_cs = self.buildlstmstack(x, self.now_hs, self.now_cs)
            a=self.now_hs[len(self.now_hs) - 1]
            print(a.size())
            self.outputs.append(self.now_hs[len(self.now_hs)-1])

        for i in range(self.num_layers):
            self.out_states_lst.append(self.now_hs[i])
            self.out_states_lst.append(self.now_cs[i])


        #self.out_states = torch.cat(self.out_states_lst,1)
        # for i in range(2,self.num_unroll):
        # for j in range (1,self.num_layers):
        # do_share_parametrs

        self.output = torch.cat(self.outputs, 2)

        return self.output


class GetLstmNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, output_size, input_size):
        super(GetLstmNet,self).__init__()

        self.lstm_input = {}
        self.lstm_output = {}
        self.init_states = {}
        self.out_states = {}
        self.num_unroll, self.num_layers, self.rnn_size, self.output_size, self.input_size  = num_unroll, num_layers, rnn_size, output_size, input_size
        self.l_pred_l = nn.Linear(self.num_unroll * self.rnn_size, self.output_size)
        self.l_pred_bn = nn.BatchNorm1d(self.output_size)
        self.pred = {}
        self.lstmnet = BuildLstmUnrollNet(self.num_unroll, self.num_layers, self.rnn_size, self.input_size)

    def forward(self, x, init_states_input):
        self.lstm_output = self.lstmnet(x, init_states_input)
        print(self.lstm_output.size())
        self.pred = self.l_pred_l(self.lstm_output)
        print(self.pred.size())
        return self.pred



###########Usage#######################################

input_size = 20
output_size = 100
rnn_size = 100
num_layers = 3
num_unroll = 5
model = GetLstmNet(num_unroll, num_layers, rnn_size, output_size, input_size)
x = torch.rand(3,20)
z = torch.zeros(3,rnn_size * num_layers * 2)
#y = net.train()
output = model(x,z)
print(output.size())








