import torch
import torchvision
import torch.nn as nn

def passthrough(x, **kwargs):
    return x

class BuildLstmStack(nn.Module):

    def __init__(self, inp, prev_hs, prev_cs, input_size, rnn_size, num_layers):
        super(BuildLstmStack, self).__init__()

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
                self.x.append(inp)
                self.x_size.append(self.input_size)
            else:
                self.x.append(self.next_hs[L - 1])
                self.x_size.append(self.rnn_size)

            self.l_i2h.append(nn.Linear(self.x_size[L], 4 * self.rnn_size))
            self.l_h2h.append(nn.Linear(self.rnn_size, 4 * self.rnn_size))
            self.l_bn.append(nn.BatchNorm1d(4 * self.rnn_size))
            self.prev_c.append(prev_cs[L])
            self.prev_h.append(prev_hs[L])

    def forward(self, inp, prev_hs, prev_cs):
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


class BuildLstmUnrollNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, output_size):
        super(BuildLstmUnrollNet, self).__init__()

        self.num_unroll = num_unroll
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.output_size = output_size
        self.inp = passthrough
        self.init_states_input = passthrough
        self.init_hs = {}
        self.init_cs = {}
        self.out = {}
        self.out_states_lst = {}
        self.outputs = {}
        self.out_states = {}
        self.output = {}

    def forward(self):
        init_states = torch.reshape(self.init_states_input,(self.num_layers * 2, self.rnn_size))
        init_states_lst  = list(torch.chunk(init_states,self.rnn_size,1))
        for i in range(self.num_layers):
            self.init_hs[i],self.init_cs[i]  = init_states_lst[i].split(self.num_layers * 2,0)

        self.now_hs, self.now_cs = self.init_hs, self.init_cs

        for i in range(self.num_unroll):
            self.now_hs, self.now_cs = BuildLstmStack(inp, self.now_hs, self.now_cs, self.num_unroll, self.num_layers, self.rnn_size, self.output_size)
            self.outputs[i] = self.now_hs[len(self.now_hs)-1]

        for i in range(self.num_layers):
            self.out_states_lst[i*2-1] = self.now_hs[i]
            self.out_states_lst[i*2] = self.now_cs[i]

        self.out_states = torch.cat(self.out_states_lst,1)
#        for i in range(2,self.num_unroll):
            #for j in range (1,self.num_layers):
                #do_share_parametrs

        self.output = torch.cat(self.outputs, 1)

        return self.inp, self.output, self.init_states_input, self.out_states



class GetLstmNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, output_size):
        super(GetLstmNet,self).__init__()

        self.num_unroll, self.num_layers, self.rnn_size, self.output_size = num_unroll, num_layers, rnn_size, output_size
        self.lstm_input, self.lstm_output, self.init_states, self.out_states = BuildLstmUnrollNet(self.num_unroll, self.num_layers, self.rnn_size, self.output_size)
        self.l_pred_l = nn.Linear(self.num_unroll * self.rnn_size, self.output_size)
        self.l_pred_bn = nn.BatchNorm1d(self.output_size)
        self.pred = {}


    def forward(self):
        self.pred = self.l_pred_l(self.lstm_output)



###########Usage#######################################
input_size = 20
output_size = 100
rnn_size = 100
num_layers = 3
num_unroll = 5
net = GetLstmNet(output_size, rnn_size, num_layers, num_unroll)
x = torch.rand(5,20)
z = torch.zeros(5, rnn_size * num_layers * 2)
#y = net.train()
output = net(x,z)








