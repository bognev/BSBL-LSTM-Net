import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from mat4py import loadmat
# #from torchsummary import summary
# from graphviz import Digraph
# from torchviz import make_dot
# from graphviz import Source

import time

HOME = 1
c = 3 * 10 ** 8
dt = 10 ** (-7)
Ts = 1.6000e-06
L = int(Ts / dt)
T = 400
NOISE = 1
H = 0
R = 1

# if torch.cuda.is_available() and HOME == 0:
#     from google.colab import drive
#     drive.mount("/content/gdrive", force_remount=True)

class BuildGRUStack(nn.Module):

    def __init__(self, input_size, rnn_size, num_layers):
        super(BuildGRUStack, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        l_i2h_lst = [nn.Linear(self.input_size, 3 * self.rnn_size)]
        l_h2h_lst = [nn.Linear(self.rnn_size, 3 * self.rnn_size)]
#         l_bn_lst = [nn.BatchNorm1d(3 * self.rnn_size)]
#         self.l_do = nn.Dropout(0.25)
        for L in range(1, self.num_layers):
            l_i2h_lst.append(nn.Linear(self.rnn_size, 3 * self.rnn_size))
            l_h2h_lst.append(nn.Linear(self.rnn_size, 3 * self.rnn_size))
#             l_bn_lst.append(nn.BatchNorm1d(3 * self.rnn_size))
        self.l_i2h = nn.ModuleList(l_i2h_lst)
        self.l_h2h = nn.ModuleList(l_h2h_lst)
#         self.l_bn = nn.ModuleList(l_bn_lst)


    def forward(self, x, prev_hs):
        self.x_size = []
        self.prev_h = 0
        self.next_hs = []
        self.i2h = []
        self.h2h = []
        for L in range(self.num_layers):
            self.prev_h = prev_hs[L]
            if L == 0:
                self.x = x
            else:
                self.x = self.next_hs[L - 1]
#             self.i2h.append(self.l_do(self.l_bn[L](self.l_i2h[L](self.x))))
#             self.h2h.append(self.l_do(self.l_bn[L](self.l_h2h[L](self.prev_h))))
            self.i2h.append(self.l_i2h[L](self.x))
            self.h2h.append(self.l_h2h[L](self.prev_h))
            Wx1, Wx2, Wx3 = self.i2h[L].chunk(3, dim=1) # it should return 4 tensors self.rnn_size
            Uh1, Uh2, Uh3 = self.h2h[L].chunk(3, dim=1)
            zt = torch.sigmoid(Wx1 + Uh1)
            rt = torch.sigmoid(Wx2 + Uh2)
            h_candidate = torch.tanh(Wx3 + rt * Uh3)
            ht = (1-zt) * self.prev_h + zt * h_candidate
            self.next_hs.append(ht)
        return torch.stack(self.next_hs)


class BuildGRUUnrollNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, input_size):
        super(BuildGRUUnrollNet, self).__init__()
        self.num_unroll = num_unroll
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.input_size = input_size
        self.outputs = []
        self.output = []
        self.now_h = []
        self.buildGRUstack_lst = []
        for i in range(0, self.num_unroll):
            self.buildGRUstack_lst.append(BuildGRUStack(self.input_size, self.rnn_size, self.num_layers))
        self.buildGRUstack = nn.ModuleList(self.buildGRUstack_lst)

    def forward(self, x, init_states_input):

        self.init_hs = []
        self.now_hs = []
        self.outputs = []

        init_states = init_states_input.reshape((init_states_input.size(0), self.num_layers * 2, self.rnn_size))
        init_states_lst = list(init_states.chunk(self.num_layers * 2, 1))

        for i in range(self.num_layers):
            self.init_hs.append(init_states_lst[2 * i].reshape(init_states_input.size(0), self.rnn_size))

        self.now_hs.append(torch.stack(self.init_hs))

        for i in range(self.num_unroll):
            self.now_h = self.buildGRUstack[i](x, self.now_hs[i])
            self.now_hs.append(self.now_h)
            self.outputs.append(self.now_hs[i + 1][-1])
            # for L in range(self.num_layers):
            #     setattr(self, 'hid_%d_%d' %(i, L), self.now_hs[i][L])
            #     setattr(self, 'cell_%d_%d' %(i, L), self.now_cs[i][L])
        # for i in range(1, self.num_unroll):
        #     for j in range(self.num_layers):
        #         self.buildGRUstack[i].l_i2h[j].weight.data = self.buildGRUstack[0].l_i2h[j].weight.data
        #         self.buildGRUstack[i].l_h2h[j].weight.data = self.buildGRUstack[0].l_h2h[j].weight.data
        #         self.buildGRUstack[i].l_i2h[j].bias.data = self.buildGRUstack[0].l_i2h[j].bias.data
        #         self.buildGRUstack[i].l_h2h[j].bias.data = self.buildGRUstack[0].l_h2h[j].bias.data
        #         self.buildGRUstack[i].l_i2h[j].weight.grad = self.buildGRUstack[0].l_i2h[j].weight.grad
        #         self.buildGRUstack[i].l_h2h[j].weight.grad = self.buildGRUstack[0].l_h2h[j].weight.grad
        #         self.buildGRUstack[i].l_i2h[j].bias.grad = self.buildGRUstack[0].l_i2h[j].bias.grad
        #         self.buildGRUstack[i].l_h2h[j].bias.grad = self.buildGRUstack[0].l_h2h[j].bias.grad
        self.output = self.outputs[0]
        for i in range(1, self.num_unroll):
            self.output = torch.cat((self.output, self.outputs[i]), 1)

        return self.output


class GetGRUNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, output_size, input_size):
        super(GetGRUNet, self).__init__()
        self.num_unroll, self.num_layers, self.rnn_size, self.output_size, self.input_size = num_unroll, num_layers, rnn_size, output_size, input_size
        self.l_pred_l = nn.Linear(self.num_unroll * self.rnn_size, self.output_size)
        self.GRUnet = BuildGRUUnrollNet(self.num_unroll, self.num_layers, self.rnn_size, self.input_size)
        self.l_pred_bn = nn.BatchNorm1d(self.output_size)
        # setattr(self, 'GRUNetLinear', self.l_pred_l)

    def forward(self, x, init_states_input):
        self.GRU_output = self.GRUnet(x, init_states_input)
        self.pred = self.l_pred_bn(self.l_pred_l(self.GRU_output))
        return self.pred
    
class BuildConv1dStack(nn.Module):

    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride, padding):
        super(BuildConv1dStack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride, \
                                     padding=self.padding, dilation=1, groups=1)
        # self.conv1x1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                                kernel_size=1, stride=self.stride, \
        #                                padding=0, dilation=1, groups=1)
        self.output_size = (self.input_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.bn = nn.BatchNorm1d(self.out_channels)  # , self.output_size)
        self.relu1 = nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride, \
                                     padding=self.padding, dilation=1, groups=1)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False,
                                          ceil_mode=False)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        self.x = x
        self.x = self.conv1(self.x)
        self.x = self.bn(self.x)
        self.x = self.relu1(self.x)
        # self.x = self.maxpool(self.x)
        # print(self.x.shape)
        # self.x = self.conv2(self.x)#self.conv1x1(x) +
        # self.x = self.relu2(self.x)
        self.x = self.maxpool(self.x)
        return self.x


class BuildConv1dStackInterm(nn.Module):

    def __init__(self, N, input_size, fc_size):
        super(BuildConv1dStackInterm, self).__init__()
        self.N = N
        self.input_size = input_size
        self.fc_size = fc_size
        # self.fc_in = nn.Linear(self.input_size,self.input_size)
        self.conv1 = torch.nn.Conv1d(in_channels=self.N, out_channels=4 * self.N, kernel_size=19, stride=2, \
                                     padding=9, dilation=1, groups=1)
        self.bn = nn.BatchNorm1d(4 * self.N, self.input_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.ModuleList(
            [nn.Linear(self.input_size, self.fc_size), nn.Linear(self.input_size, self.fc_size)])

    def forward(self, x):
        self.x = x
        self.x = self.conv1(self.x)
        self.x = self.bn(self.x)
        self.x = self.relu(self.x)
        self.x = self.fc_out[1](x) + self.fc_out[0](self.x)
        self.x = self.relu(self.x)
        return self.x


class BuildConv1dUnrollNet(nn.Module):

    def __init__(self, N, num_unroll, input_size, fc_size):
        super(BuildConv1dUnrollNet, self).__init__()
        self.N = N
        self.num_unroll = num_unroll
        self.fc_size = fc_size

        buildConv1dstack_lst_in = []
        self.in_channels, self.out_channels, self.input_size, self.kernel_size, self.stride, self.padding = 8, 16, input_size, 19, 1, 9
        self.output_size = (self.input_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.output_size = self.output_size / 2
        print(self.output_size)
        buildConv1dstack_lst_in.append(BuildConv1dStack(self.in_channels, self.out_channels, self.input_size, \
                                                        self.kernel_size, self.stride, self.padding))

        self.in_channels, self.out_channels, self.input_size, self.kernel_size, self.stride, self.padding = 16, 32, self.output_size, 9, 1, 4
        self.output_size = (self.output_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.output_size = self.output_size / 2
        print(self.output_size)
        buildConv1dstack_lst_in.append(BuildConv1dStack(self.in_channels, self.out_channels, self.input_size, \
                                                        self.kernel_size, self.stride, self.padding))

        self.in_channels, self.out_channels, self.input_size, self.kernel_size, self.stride, self.padding = 32, 64, self.output_size, 5, 1, 2
        self.output_size = (self.output_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.output_size = self.output_size / 2
        print(self.output_size)
        buildConv1dstack_lst_in.append(BuildConv1dStack(self.in_channels, self.out_channels, self.input_size, \
                                                        self.kernel_size, self.stride, self.padding))

        self.in_channels, self.out_channels, self.input_size, self.kernel_size, self.stride, self.padding = 64, 128, self.output_size, 3, 1, 1
        self.output_size = (self.output_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.output_size = self.output_size / 2
        print(self.output_size)
        buildConv1dstack_lst_in.append(BuildConv1dStack(self.in_channels, self.out_channels, self.input_size, \
                                                        self.kernel_size, self.stride, self.padding))

        self.l_Conv1ds_in = nn.ModuleList(buildConv1dstack_lst_in)
        # self.l_Conv1ds_imd = BuildConv1dStackInterm(self.N, self.input_size, self.fc_size)
        # buildConv1dstack_lst_out = []
        # buildConv1dstack_lst_in.append(BuildConv1dStack(in_channels, out_channels, self.output_size, kernel_size, stride))
        # buildConv1dstack_lst_in.append(BuildConv1dStack(in_channels, out_channels, self.output_size, kernel_size, stride))
        # buildConv1dstack_lst_in.append(BuildConv1dStack(in_channels, out_channels, self.output_size, kernel_size, stride))
        # buildConv1dstack_lst_in.append(BuildConv1dStack(in_channels, out_channels, self.output_size, kernel_size, stride))

        # self.l_Conv1ds_out = nn.ModuleList(buildConv1dstack_lst_out)

    def forward(self, x):
        self.x = x
        for L in range(0, self.num_unroll):
            self.x = self.l_Conv1ds_in[L](self.x)
            # print(self.x.shape)
        # print("a")
        # self.x = self.l_Conv1ds_imd(self.x)
        # for L in range(0, self.num_unroll):
        #     self.x = self.l_Conv1ds_out[L](self.x)
        # self.res_out = [self.l_Conv1ds_out[0](self.res_int)]
        # for L in range(0, self.num_unroll-1):
        #     self.res_out.append(self.l_Conv1ds_out[L+1](self.res_out[L]))

        return self.x
        # return self.res_out[-1]


class GetConv1d(nn.Module):

    def __init__(self, N, num_unroll, input_size, fc_size, rnn_size, output_size, batch_size):
        super(GetConv1d, self).__init__()
        self.num_unroll, self.fc_size = num_unroll, fc_size
        self.in_channels = N
        self.out_channels = N
        self.input_size = input_size
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride, \
                                     padding=self.padding, dilation=1, groups=1)
        self.output_size = (self.input_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.l_bn_in = nn.BatchNorm1d(N)  # , self.input_size)
        self.l_Conv1d = BuildConv1dUnrollNet(N, self.num_unroll, self.input_size, self.fc_size)
        self.l_fc_out = nn.Linear(
            int(self.num_unroll * 2 * self.num_unroll ** 2 * self.input_size / self.num_unroll ** 2), self.fc_size)
        self.n_gru = GetGRUNet(1, num_layers, int(rnn_size/4), output_size, input_size)
        self.batch_zero_states = torch.zeros(batch_size, num_layers * int(rnn_size/4) * 2)

    def forward(self, x):
        self.x = x
        # self.x = self.conv1(self.x)
        # self.x = self.l_bn_in(self.x)
        self.x = self.l_Conv1d(self.x)
        self.x = self.n_gru(self.x, self.batch_zero_states)
        self.x = self.l_fc_out(self.x.view(self.x.shape[0], -1))
        return self.x


##########Usage#######################################
# #plot
# input_size = 400
# output_size = 144
# rnn_size = 10
# num_layers = 2
# num_unroll = 4
# model = GetConv1d(8, 4, input_size, output_size)
# # graph of net
# x = torch.rand(3, 8, input_size)
# out = model(x)
# print(model)
# temp = make_dot(out, params=dict(list(model.named_parameters())+ [('x', x)]))
# s = Source(temp, filename="test.gv", format="png")
# s.view()


class MultiClassNLLCriterion(torch.nn.Module):

    def __init__(self):
        super(MultiClassNLLCriterion, self).__init__()
        self.lsm = nn.LogSoftmax(dim=1)
        self.nll = nn.NLLLoss()
        self.output = 0
        self.outputs = 0

    def forward(self, inputs, target):
        self.output = self.lsm(inputs)
        shape = target.shape
        self.outputs = 0
        # print(self.output.shape)
        # print(target.shape)
        for i in range(0, shape[1]):
            self.outputs = self.outputs + self.nll(self.output, target[:, i].squeeze())
        return self.outputs  # /shape[1]


# match number
def AccS(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(num_nonz)  # ?!
    pred = pred.float()
    t_score = torch.zeros(label.shape).to(device)
    #     print(label.get_device())
    #     print(pred.get_device())
    for i in range(0, num_nonz):
        for j in range(0, num_nonz):
            t_score[:, i].add_(label[:, i].float().eq(pred[:, j]).float())
    return t_score.mean()


# loose match
def AccL(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(20)  # ?!
    pred = pred.float()
    t_score = torch.zeros(label.shape).to(device)
    for i in range(0, num_nonz):
        for j in range(0, 20):
            t_score[:, i].add_(
                label[:, i].float().eq(pred[:, j]).float())  # t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.mean()


# sctrict match
def AccM(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(num_nonz)  # ?!
    pred = pred.float()
    t_score = torch.zeros(label.shape).to(device)
    for i in range(0, num_nonz):
        for j in range(0, num_nonz):
            t_score[:, i].add_(
                label[:, i].float().eq(pred[:, j]).float())  # t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.sum(1).eq(num_nonz).sum().item() * 1. / pred.shape[0]


gpu = 1  # gpu id

if torch.cuda.is_available() and HOME == 0:
    batch_size = 256  # 10# training batch size
else:
    batch_size = 8  # 600000  #
lr = 0.002  # basic learning rate
lr_decay_startpoint = 250  # learning rate from which epoch
num_epochs = 200  # total training epochs
max_grad_norm = 5.0
clip_gradient = 4.0
N = 8  # the number of receivers
M = 3  # the number of transmitters
K = 3  # the number of targets

# task related parameters
# task: y = Ax, given A recovery sparse x from y
dataset = 'uniform'  # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
# num_nonz = K*N*M*2  # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
num_nonz = K  # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
input_size = T * 2  # dimension of observation vector y
output_size = 11*11  # dimension of sparse vector x
# # task related parameters
# # task: y = Ax, given A recovery sparse x from y
# dataset = 'uniform'  # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
# num_nonz = 3  # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
# input_size = 20  # dimension of observation vector y
# output_size = 100  # dimension of sparse vector x

# model hyper parameters
rnn_size = 425  # number of units in RNN cell
num_layers = 3  # number of stacked RNN layers
num_unroll = 4  # number of RNN unrolled time steps

# torch.set_num_threads(16)
# manualSeed = torch.randint(1,10000,(1,))
# print("Random seed " + str(manualSeed.item()))
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() and HOME == 0 else "cpu")
print(device)

# if torch.cuda.is_available() and HOME == 0:
#     train_size = 600000  #
#     valid_size = 100000  #
# else:
#     train_size = 100  # 600000  #
#     valid_size = 10  # 100000  #
# valid_data = torch.zeros(valid_size, input_size).to(device)
# valid_label = torch.zeros(valid_size, num_nonz).type(torch.LongTensor).to(device)
# batch_data = torch.zeros(batch_size, input_size).to(device)
# batch_label = torch.zeros(batch_size, num_nonz).to(device)  # for MultiClassNLLCriterion LOSS

if torch.cuda.is_available() and HOME == 0:
    train_size = int(256 * 800)  #
    valid_size = int(256 * 200)  #
else:
    train_size = 256  # 600000  #
    valid_size = 32  # 100000  #
print(device)
valid_data = torch.zeros(valid_size, N, input_size).to(device)
valid_label = torch.zeros(valid_size, num_nonz).type(torch.LongTensor).to(device)
batch_data = torch.zeros(batch_size, N, input_size).to(device)
batch_label = torch.zeros(batch_size, num_nonz).to(device)  # for MultiClassNLLCriterion LOSS
batch_zero_states = torch.zeros(batch_size, num_layers * rnn_size * 2).to(device)  # init_states for GRU

# AccM, AccL, Accs = 0, 0, 0


err = 0

model_all = "model_l_" + str(num_layers) + "t_" + str(num_unroll) + '_Conv1d_' + str(rnn_size)
logger_file = model_all + str(dataset) + "_" + str(num_nonz) + '.log'


# if torch.cuda.is_available() and HOME == 0:
#      logger_file = "/content/gdrive/My Drive/" + logger_file  # or torch.save(net, PATH)
# else:
#     logger_file = "./" + logger_file
# logger = open(logger_file, 'w')

# for k,v in pairs(opt) do logger:write(k .. ' ' .. v ..'\n') end
# logger:write('network have ' .. paras:size(1) .. ' parameters' .. '\n')
# logger:close()

# # torch.manual_seed(10)
# # mat_A = torch.rand(output_size,input_size)
# if torch.cuda.is_available() and HOME == 0:
#     mat_A = torch.load("/content/gdrive/My Drive/mat_A.pt").to(device)
# else:
#     mat_A = torch.load("./mat_A.pt").to(device)
# # mat_A = torch.load("/content/gdrive/My Drive/mat_A.pt").to(device)


# def gen_batch(batch_size, num_nonz, mat_A):
#     # mat_A = loadmat('matrix_corr_unit_20_100.mat')
#     # mat_A = torch.FloatTensor(mat_A['A']).t()
#     # print(mat_A.shape)
#     # mat_A = torch.rand(output_size, input_size)
#     batch_X = torch.Tensor(batch_size, 100).to(device)
#     batch_n = torch.Tensor(batch_size, num_nonz).to(device)
#     bs = batch_size
#     len = int(100 / num_nonz * num_nonz)
#     perm = torch.randperm(100)[range(len)].to(device)
#     #     batch_label = torch.zeros(batch_size, num_nonz).type(torch.LongTensor).to(device)  # for MultiClassNLLCriterion LOSS
#     for i in range(int(bs * num_nonz / len)):
#         perm = torch.cat((perm, torch.randperm(100)[range(len)].to(device)))
#     batch_label = perm[range(bs * num_nonz)].reshape([bs, num_nonz]).type(torch.LongTensor).to(device)
#     batch_X.zero_()
#     if dataset == 'uniform':
#         batch_n.uniform_(-0.5, 0.5)
#         batch_n[batch_n.gt(0)] = batch_n[batch_n.gt(0)] + 0.1
#         batch_n[batch_n.le(0)] = batch_n[batch_n.le(0)] - 0.1
#     #
#     # print(batch_X.shape)
#     #     print(batch_X.get_device())
#     #     print(mat_A.get_device())
#     #     print(batch_n.get_device())
#     for i in range(bs):
#         for j in range(num_nonz):
#             batch_X[i][batch_label[i][j]] = batch_n[i][j]
#     batch_data = torch.mm(batch_X, mat_A)  # +0.001*torch.randn(batch_size,input_size).to(device)
#     # print(batch_label.shape)
#     # print(batch_data.shape)
#     return batch_label, batch_data
x_r = np.array([1000, 2000, 2500, 2500, 2000, 1000, 500, 500])#*np.random.rand(1)# + 500 * (np.random.rand(N) - 0.5))  # \
y_r = np.array([500, 500, 1000, 2000, 2500, 2500, 2000, 1500])#*np.random.rand(1)# + 500 * (np.random.rand(N) - 0.5))  # \
# Position of transmitters
x_t = np.array([0, 4000, 4000, 0, 1500, 0, 4000, 2000])#+500*np.random.rand(1)
y_t = np.array([0, 0, 4000, 4000, 4000, 1500, 1500, 0])#+500*np.random.rand(1)
x_r=x_r.reshape(8,1)
y_r=y_r.reshape(8,1)
x_t=x_t.reshape(8,1)
y_t=y_t.reshape(8,1)
# 1500,3000,500,2500,1000,1500,500,3000,\
# 2500,3500,1000,3500,2000,4000,3000,3000]+500*(np.random.rand(N)-0.5))
# 3500,3500,500,4000,4000,2500,3000,500,\
# 3500,3000,2000,1000,2000,500,4000,1500]+500*(np.random.rand(N)-0.5))

s = np.zeros([M, L]) + 1j * np.zeros([M, L])
for m in range(M):
    s[m] = np.exp(1j * 2 * np.pi * (m) * np.arange(L) / M) / np.sqrt(L);#np.sqrt(0.5)*(np.random.randn(1,L)+1j*np.random.randn(1,L))/np.sqrt(L);#
Ls = 0
Le = Ls + 4000
dx = 333
dy = dx
dy = dx
x_grid = np.arange(Ls, Le, dx)
y_grid = np.arange(Ls, Le, dy)
size_grid_x = len(x_grid)
size_grid_y = len(y_grid)
grid_all_points = [[i, j] for i in x_grid for j in y_grid]
grid_all_points_a = np.array(grid_all_points)


const_sqrt_200000000000 = np.sqrt(200000000000)


def gen_mimo_samples(batch_size, SNR_dB, M, N, K, NOISE, H, R):
    x_r = np.array(
        [1000, 2000, 2500, 2500, 2000, 1000, 500, 500]) + 128 # + 500 * (np.random.rand(N) - 0.5))  # \
    y_r = np.array(
        [500, 500, 1000, 2000, 2500, 2500, 2000, 1500]) + 128# + 500 * (np.random.rand(N) - 0.5))  # \
    # Position of transmitters
    x_t = np.array([0, 4000, 4000, 0, 1500, 0, 4000, 2000]) + 128
    y_t = np.array([0, 0, 4000, 4000, 4000, 1500, 1500, 0]) + 128
    x_r = x_r.reshape(8, 1)
    y_r = y_r.reshape(8, 1)
    x_t = x_t.reshape(8, 1)
    y_t = y_t.reshape(8, 1)
    # 1500,3000,500,2500,1000,1500,500,3000,\
    # 2500,3500,1000,3500,2000,4000,3000,3000]+500*(np.random.rand(N)-0.5))
    # 3500,3500,500,4000,4000,2500,3000,500,\
    # 3500,3000,2000,1000,2000,500,4000,1500]+500*(np.random.rand(N)-0.5))

    s = np.zeros([M, L]) + 1j * np.zeros([M, L])
    for m in range(M):
        s[m] = np.exp(1j * 2 * np.pi * (m) * np.arange(L) / M) / np.sqrt(L);  # np.sqrt(0.5)*(np.random.randn(1,L)+1j*np.random.randn(1,L))/np.sqrt(L);#
    Ls = 0
    Le = Ls + 1000
    dx = 91
    dy = dx
    x_grid = np.arange(Ls, Le, dx)
    y_grid = np.arange(Ls, Le, dy)
    size_grid_x = len(x_grid)
    size_grid_y = len(y_grid)
    grid_all_points = [[i, j] for i in x_grid for j in y_grid]
    grid_all_points_a = np.array(grid_all_points)

    const_sqrt_200000000000 = np.sqrt(200000000000)
    grid_all_points_bs = np.repeat(grid_all_points_a[np.newaxis, ...], batch_size, axis=0)
    x_r_bs = np.repeat(x_r[np.newaxis, ...], batch_size, axis=0)
    y_r_bs = np.repeat(y_r[np.newaxis, ...], batch_size, axis=0)
    x_t_bs = np.repeat(x_t[np.newaxis, ...], batch_size, axis=0)
    y_t_bs = np.repeat(y_t[np.newaxis, ...], batch_size, axis=0)
    rk = np.zeros([batch_size, K, M, N, 1]);
    tk = np.zeros([batch_size, K, M, N, 1]);
    tau = np.zeros([batch_size, K, M, N, 1]);
    r = np.zeros([batch_size, size_grid_x * size_grid_y])
    DB = 10. ** (0.1 * SNR_dB)
    # NOISE = 1  # on/off noise
    # H = 1  # on/off êîýôôèöèåíòû îòðàæåíèÿ
    if NOISE == 0:
        x = np.zeros([batch_size, N, T]) + 1j * np.zeros([batch_size, N, T])
    else:
        x = (np.random.randn(batch_size, N, T) + 1j * np.random.randn(batch_size, N, T)) / np.sqrt(2)
    if H == 0:
        h = np.ones([batch_size, K, M, N])
    else:
        h = (np.random.randn(batch_size, K, M, N) + 1j * np.random.randn(batch_size, K, M, N)) / np.sqrt(2)


    k_random_grid_points = np.zeros([batch_size,K])
    # Position of targets
    # a=np.random.randint(0,size_grid_x*size_grid_y,K)
    a = np.random.randint(0, size_grid_x * size_grid_y, (batch_size, K, 1))
    if R == 0:
        x_k = grid_all_points_a[a[:,:,0]][:,:,0].reshape((batch_size,K,1))
        y_k = grid_all_points_a[a[:,:,0]][:,:,1].reshape((batch_size,K,1))
    else:
        x_k = np.random.randint(Ls,Le,(batch_size,K,1))+np.random.rand(batch_size,K,1)
        y_k = np.random.randint(Ls,Le,(batch_size,K,1))+np.random.rand(batch_size,K,1)

    # print(a[100])
    # print(x_k[100].transpose())
    # print(y_k[100].transpose())
    k_random_grid_points_i = np.zeros([batch_size,K])

    for k in range(K):
        calc_dist = np.sqrt((grid_all_points_bs[:,:, 0] - x_k[:,k]) ** 2 \
                            + (grid_all_points_bs[:,:, 1] - y_k[:,k]) ** 2)
        k_random_grid_points_i[:,k] = calc_dist.argmin(axis=1)
    # Time delays
    for k in range(K):
        for m in range(M):
            for n in range(N):
                tk[:, k, m, n] = np.sqrt((x_k[:,k] - x_t_bs[:, m]) ** 2 + (y_k[:,k] - y_t_bs[:, m]) ** 2)
                rk[:, k, m, n] = np.sqrt((x_k[:,k] - x_r_bs[:, n]) ** 2 + (y_k[:,k] - y_r_bs[:, n]) ** 2)
                tau[:, k, m, n] = (tk[:, k, m, n] + rk[:, k, m, n]) / c

    r_glob = np.zeros([batch_size, size_grid_x * size_grid_y * M * N]) + 1j * np.zeros([batch_size, size_grid_x * size_grid_y * M * N])
    k_random_grid_points = np.array(k_random_grid_points_i,copy=True)
    # print(k_random_grid_points[100])
    # print(grid_all_points_bs[100,k_random_grid_points[100].astype(int)].transpose())
    for bs in range(batch_size):
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    r_glob[bs, k_random_grid_points_i[bs,k].astype(int)] = DB[k] * h[bs, k, m, n] * \
                                                                           np.sqrt(200000000000) * (1 / tk[bs, k, m, n]) * (1 / rk[bs, k, m, n])
                k_random_grid_points_i[bs,:] = k_random_grid_points_i[bs,:] + size_grid_x * size_grid_y


    # np.put(r, k_random_grid_points_i.astype(int), 1)

    l = np.floor(tau / dt).astype(int)
    for bs in range(batch_size):
        for k in range(K):
            for n in range(N):
                for m in range(M):
                    x[bs, n, l[bs,k,m,n].item(): l[bs,k,m,n].item() + L] = x[bs, n, l[bs,k,m,n].item(): l[bs,k,m,n].item() + L] + DB[k] * s[m, :] * h[bs, k, m, n] * \
                                             const_sqrt_200000000000 * (1 / tk[bs, k, m, n]) * (1 / rk[bs, k, m, n])

    x_flat = x[:, 0, :];
    for n in range(1, N):
        x_flat = np.concatenate([x_flat, x[:, n, :]], axis=1)

    return x, k_random_grid_points, r_glob

def gen_batch(batch_size, M, N, K, NOISE, H, R):
#     NOISE = 1
#     H = 1
    SNR_dB = np.random.rand(3)
    y, label, numb = gen_mimo_samples(batch_size, SNR_dB, M, N, K, NOISE, H, R)
    batch_data = torch.zeros(batch_size, 2*y.shape[0])
#     batch_label = torch.zeros(batch_size, 2*label.shape[0]).to(device)
    batch_label = torch.zeros(batch_size, label[range(num_nonz)].shape[0])
    r1 = 40
    r2 = 20
    # for i in range(batch_size):
        # SNR_dB = ((r1 - r2) * torch.rand((1,)) + r2).item()
    for k in range(K):
        SNR_dB[k] = 20  # ((r1 - r2) * np.random.rand(1) + r2)
    y, label, numb = gen_mimo_samples(batch_size, SNR_dB, M, N, K, NOISE, H, R)
    batch_data = torch.cat([torch.from_numpy(y.real),torch.from_numpy(y.imag)], dim=2)
#         batch_data[i] = torch.cat([torch.from_numpy(np.abs(y))]).to(device)
#         batch_label[i] = torch.cat([torch.from_numpy(label),torch.from_numpy(label+M*N*36)]).to(device)
    batch_label = torch.cat([torch.from_numpy(label)])


    return batch_label.type(torch.LongTensor).to(device), batch_data.type(torch.FloatTensor).to(device)


print("building validation set")
for i in range(0, valid_size, batch_size):
    #     mat_A = torch.rand(output_size, input_size).to(device)
    batch_label, batch_data = gen_batch(batch_size, M, N, K, NOISE, H, R)
    # print(batch_label.shape)
    # print("batch_data shape = " + str(batch_data.shape))
    # print("valid_data shape = " + str(valid_data.shape))
    # print(range(i,i+batch_size-1))
    valid_data[range(i, i + batch_size), :] = batch_data
    valid_label[range(i, i + batch_size), :] = batch_label
print('done')

print('done')

best_valid_accs = 0
base_epoch = lr_decay_startpoint
base_lr = lr
optimState = {'learningRate': 0.01, 'weigthDecay': 0.0001}

net = GetConv1d(N, num_unroll, input_size, output_size, rnn_size, output_size, batch_size)
# print(net)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
# summary(net,[(num_layers,input_size),(num_layers,rnn_size * num_layers * 2)])
# summary(net,[(batch_size, input_size),(batch_size, num_layers * rnn_size * 2)])

# create a stochastic gradient descent optimizer
# optimizer = optim.RMSprop(params=net.parameters(), lr=0.001, alpha=0.9, eps=1e-04, weight_decay=0.0001, momentum=0, centered=False)
# create a loss function
LOSS = MultiClassNLLCriterion()
optimizer = optim.SGD(params=net.parameters(), lr=optimState['learningRate'], \
                      momentum=0.9, dampening=0, weight_decay=optimState['weigthDecay'], nesterov=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.25)
# checkpoint = torch.load( "/content/gdrive/My Drive/model_l_2t_17_rnn_800_3.pth")
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# epoch = checkpoint['epoch'] + 1
# loss = checkpoint['loss']
epoch = 0
print(net)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
# mat_A = torch.rand(output_size, input_size).to(device)
for epoch in range(epoch, num_epochs):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    # learing rate self - adjustment
    # if(epoch > 250):
    #     optimState['learningRate'] = base_lr / (1 + 0.06 * (epoch - base_epoch))
    #     if(epoch % 50 == 0): base_epoch = epoch; base_lr= base_lr * 0.25

    logger = open(logger_file, 'a')
    # train
    train_accs = 0
    train_accl = 0
    train_accm = 0
    train_err = 0
    nbatch = 0

    net.train()
    start = time.time()
    for i in range(0, train_size, batch_size):
        batch_label, batch_data = gen_batch(batch_size, M, N, K, 0, 0, 1) #NOISE, H, R)
        batch_label.to(device)
        optimizer.zero_grad()
        pred_prob = net(batch_data).to(device)  # 0 or 1?!
        err = LOSS(pred_prob, batch_label.to(device))
        err.backward()
        # with torch.no_grad():
        #     for name, param in net.named_parameters():
        #         # print(name)
        #         #print(param.grad.data)
        #         param.grad.clamp_(-4.0, 4.0)
        #         gnorm = param.grad.norm()
        #         if (gnorm > max_grad_norm):
        #             param.grad.mul_(max_grad_norm / gnorm)
        optimizer.step()
        #         print(pred_prob.get_device())
        #         print(batch_label.get_device())
        batch_accs = AccS(batch_label[:, range(0, num_nonz)], pred_prob.to(device).float())
        batch_accl = AccL(batch_label[:, range(0, num_nonz)], pred_prob.to(device).float())
        batch_accm = AccM(batch_label[:, range(0, num_nonz)], pred_prob.to(device).float())
        train_accs = train_accs + batch_accs.item()
        train_accl = train_accl + batch_accl.item()
        train_accm = train_accm + batch_accm
        train_err = train_err + err.item()
        nbatch = nbatch + 1
        if (torch.cuda.is_available() and HOME == 0):
            if (nbatch) % 255 == 1:
                print("Epoch " + str(epoch) + " Batch " + str(nbatch) + " {:.4} {:.4} {:.4} loss = {:.4}".format(
                    batch_accs,
                    batch_accl,
                    batch_accm,
                    err.item()))
        else:
            print("Epoch " + str(epoch) + " Batch " + str(nbatch) + " {:.4} {:.4} {:.4} loss = {:.4}".format(batch_accs,
                                                                                                             batch_accl,
                                                                                                             batch_accm,
                                                                                                             err.item()))
    end = time.time()
    print("Train [{}] Time {} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}".format(epoch, end - start, \
                                                                                    train_accs / nbatch,
                                                                                    train_accl / nbatch, \
                                                                                    train_accm / nbatch,
                                                                                    train_err / nbatch))
    # logger.write("Train [{}] Time {:.4} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}\n".format(epoch, end - start, \
    #                                                                                             train_accs / nbatch,
    #                                                                                             train_accl / nbatch, \
    #                                                                                             train_accm / nbatch,
    #                                                                                             train_err / nbatch))

    # eval
    nbatch = 0
    valid_accs = 0
    valid_accl = 0
    valid_accm = 0
    valid_err = 0
    start = time.time()
    net.eval()
    for i in range(0, valid_size, batch_size):
        batch_data = valid_data[range(i, i + batch_size), :]
        batch_label[:, range(0, num_nonz)] = valid_label[range(i, i + batch_size), :]
        pred_prob = net(batch_data)
        err = LOSS(pred_prob, batch_label)
        batch_accs = AccS(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accl = AccL(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accm = AccM(batch_label[:, range(0, num_nonz)], pred_prob.float())
        valid_accs = valid_accs + batch_accs.item()
        valid_accl = valid_accl + batch_accl.item()
        valid_accm = valid_accm + batch_accm
        valid_err = valid_err + err.item()
        nbatch = nbatch + 1
    #         if (nbatch+99) % 100 == 0:
    #             print("Eval Epoch " + str(epoch) + " Batch " + str(nbatch) + " {:.4} {:.4} {:.4} loss = {:.4}".format(batch_accs, batch_accl,
    #                                                                                             batch_accm, err.item()))
    end = time.time()
    print("Valid [{}] Time {} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}".format(epoch, end - start, \
                                                                                    valid_accs / nbatch,
                                                                                    valid_accl / nbatch, \
                                                                                    valid_accm / nbatch,
                                                                                    valid_err / nbatch))
    # logger.write("Valid [{}] Time {} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}\n".format(epoch, end - start, \
    #                                                                                          valid_accs / nbatch,
    #                                                                                          valid_accl / nbatch, \
    #                                                                                          train_accm / nbatch,
    #                                                                                          valid_err / nbatch))
    scheduler.step()
    # if(valid_accs > best_valid_accs):
    #     best_valid_accs = valid_accs
    #     print("saving model")
    #     logger.write('saving model\n')
    #     checkpoint = {'epoch': epoch,
    #                   'model_state_dict': net.state_dict(),
    #                   'optimizer_state_dict': optimizer.state_dict(),
    #                   'loss': err.item()}
    #     # torch.save(checkpoint, 'checkpoint.pth')
    #     torch.save(checkpoint, "./checkpoints/"+model_all+"_"+str(num_nonz)+".pth") #or torch.save(net, PATH)
    #     #net.load_state_dict(torch.load(PATH)) # or the_model = torch.load(PATH)

    # if(epoch % 2 == 0):
    print("saving model")
    logger.write('saving model\n')
    checkpoint = {'epoch': epoch, \
                  'model_state_dict': net.state_dict(), \
                  'optimizer_state_dict': optimizer.state_dict(), \
                  'scheduler_state_dict': scheduler.state_dict(), \
                  'loss': err.item()}
    # if torch.cuda.is_available() and HOME == 0:
    #     torch.save(checkpoint, "/content/gdrive/My Drive/" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    # else:
    #     torch.save(checkpoint, "./" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    # logger.close()



