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
HOME = 0
if torch.cuda.is_available() and HOME == 0:
    from google.colab import drive
    drive.mount("/content/gdrive", force_remount=True)


c = 3 * 10 ** 8
dt = 10 ** (-7)
Ts = 1.6000e-06
L = int(Ts / dt)
T = 400
NOISE = 1
H = 0
R = 1

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
            self.now_h = self.buildGRUstack[i](x[:, i, :], self.now_hs[i])
            self.now_hs.append(self.now_h)
            self.outputs.append(self.now_hs[i + 1][-1])
            # for L in range(self.num_layers):
            #     setattr(self, 'hid_%d_%d' %(i, L), self.now_hs[i][L])
            #     setattr(self, 'cell_%d_%d' %(i, L), self.now_cs[i][L])
        for i in range(1, self.num_unroll):
            for j in range(self.num_layers):
                self.buildGRUstack[i].l_i2h[j].weight.data = self.buildGRUstack[0].l_i2h[j].weight.data
                self.buildGRUstack[i].l_h2h[j].weight.data = self.buildGRUstack[0].l_h2h[j].weight.data
                self.buildGRUstack[i].l_i2h[j].bias.data = self.buildGRUstack[0].l_i2h[j].bias.data
                self.buildGRUstack[i].l_h2h[j].bias.data = self.buildGRUstack[0].l_h2h[j].bias.data
                self.buildGRUstack[i].l_i2h[j].weight.grad = self.buildGRUstack[0].l_i2h[j].weight.grad
                self.buildGRUstack[i].l_h2h[j].weight.grad = self.buildGRUstack[0].l_h2h[j].weight.grad
                self.buildGRUstack[i].l_i2h[j].bias.grad = self.buildGRUstack[0].l_i2h[j].bias.grad
                self.buildGRUstack[i].l_h2h[j].bias.grad = self.buildGRUstack[0].l_h2h[j].bias.grad
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


###########Usage#######################################

# input_size = 20
# output_size = 50
# rnn_size = 10
# num_layers = 2
# num_unroll = 3
# # graph of net
# x = torch.rand(3, input_size)
# z = torch.zeros(3, rnn_size * num_layers * 2)


# model = BuildGRUStack(input_size, rnn_size, num_layers)
# init_hs = []
# init_cs = []
# init_states = z.reshape((z.size(0),num_layers * 2, rnn_size))
# init_states_lst = list(init_states.chunk(num_layers * 2,1))
# for i in range(num_layers):
#     init_hs.append(init_states_lst[2*i].reshape(num_layers,rnn_size))
#     init_cs.append(init_states_lst[2*i+1].reshape(num_layers,rnn_size))
# now_hs, now_cs = model(x, init_hs, init_cs)
# temp = make_dot((now_hs[2], now_cs[2]), params=dict(list(model.named_parameters())))
# s = Source(temp, filename="BuildGRUStack.gv", format="png")
# s.view()
#
# model = BuildGRUUnrollNet(num_unroll, num_layers, rnn_size, input_size)
# out = model(x, z)
# temp = make_dot(out, params=dict(list(model.named_parameters())+ [('x', x)]+ [('z', z)]))
# s = Source(temp, filename="BuildGRUUnrollNet.gv", format="png")
# s.view()
#
# model = GetGRUNet(num_unroll, num_layers, rnn_size, output_size, input_size)
# output = model(x,z)
# for i in range(1, num_unroll):
#     for j in range(num_layers):
#         model.GRUnet.buildGRUstack[i].l_i2h[j].weight = model.GRUnet.buildGRUstack[0].l_i2h[j].weight
#         model.GRUnet.buildGRUstack[i].l_h2h[j].weight = model.GRUnet.buildGRUstack[0].l_h2h[j].weight
#         model.GRUnet.buildGRUstack[i].l_i2h[j].bias = model.GRUnet.buildGRUstack[0].l_i2h[j].bias
#         model.GRUnet.buildGRUstack[i].l_h2h[j].bias = model.GRUnet.buildGRUstack[0].l_h2h[j].bias
# print(model)
# temp = make_dot(output, params=dict(list(model.named_parameters())+ [('x', x)]+ [('z', z)]))
# s = Source(temp, filename="test.gv", format="png")
# s.view()

# modell = nn.Sequential()
# modell.add_module('W0', nn.Linear(8, 16))
# modell.add_module('tanh', nn.Tanh())
# modell.add_module('W1', nn.Linear(16, 1))
#
# x = torch.randn(1,8)
#
# temp = make_dot(modell(x), params=dict(modell.named_parameters()))
#
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
    batch_size = 5  # 600000  #
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
input_size = T*2  # dimension of observation vector y
output_size = 11*11  # dimension of sparse vector x

# model hyper parameters
rnn_size = 200  # number of units in RNN cell
num_layers = 2  # number of stacked RNN layers
num_unroll = N  # number of RNN unrolled time steps

# torch.set_num_threads(16)
# manualSeed = torch.randint(1,10000,(1,))
# print("Random seed " + str(manualSeed.item()))
# torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() and HOME == 0 else "cpu")
print(device)

if torch.cuda.is_available():
    train_size = int(batch_size*800)  #
    valid_size = int(batch_size*200)  #  #
else:
    train_size = 100  # 600000  #
    valid_size = 10  # 100000  #
print(train_size)
print(valid_size)
print(batch_size)
valid_data = torch.zeros(valid_size, N, input_size).to(device)
valid_label = torch.zeros(valid_size, num_nonz).type(torch.LongTensor).to(device)
batch_data = torch.zeros(batch_size, N, input_size).to(device)
batch_label = torch.zeros(batch_size, num_nonz).to(device)  # for MultiClassNLLCriterion LOSS
batch_zero_states = torch.zeros(batch_size, num_layers * rnn_size * 2).to(device)  # init_states for GRU

# AccM, AccL, Accs = 0, 0, 0


err = 0

model_all = "model_l_" + str(num_layers) + "t_" + str(num_unroll) + '_gru_mimo_' + str(rnn_size)
logger_file = model_all + str(dataset) + "_" + str(num_nonz) + '.log'
if torch.cuda.is_available():
    logger_file = "/content/gdrive/My Drive/" + logger_file  # or torch.save(net, PATH)
else:
    logger_file = "./" + logger_file
logger = open(logger_file, 'w')
# for k,v in pairs(opt) do logger:write(k .. ' ' .. v ..'\n') end
# logger:write('network have ' .. paras:size(1) .. ' parameters' .. '\n')
# logger:close()

# torch.manual_seed(10)
# if torch.cuda.is_available():
#     mat_A = torch.load("/content/gdrive/My Drive/mat_A.pt").to(device)
# else:
#     mat_A = torch.load("./mat_A.pt").to(device)

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

def gen_batch(batch_size, N, M, K, NOISE, H, R):
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
    batch_label, batch_data = gen_batch(batch_size, N, M, K, NOISE, H, R)
    # print(batch_label.shape)
    # print("batch_data shape = " + str(batch_data.shape))
    # print("valid_data shape = " + str(valid_data.shape))
    # print(range(i,i+batch_size-1))
    valid_data[range(i, i + batch_size), :] = batch_data
    valid_label[range(i, i + batch_size), :] = batch_label
print('done')

best_valid_accs = 0
base_epoch = lr_decay_startpoint
base_lr = lr
optimState = {'learningRate': 0.01, 'weigthDecay': 0.001}

net = GetGRUNet(num_unroll, num_layers, rnn_size, output_size, input_size)

# print(net)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# summary(net,[(num_layers,input_size),(num_layers,rnn_size * num_layers * 2)])
# summary(net,[(batch_size, input_size),(batch_size, num_layers * rnn_size * 2)])

# create a stochastic gradient descent optimizer
# optimizer = optim.RMSprop(params=net.parameters(), lr=0.001, alpha=0.9, eps=1e-04, weight_decay=0.0001, momentum=0, centered=False)
# create a loss function
LOSS = MultiClassNLLCriterion()
optimizer = optim.RMSprop(params=net.parameters(), lr=optimState['learningRate'], \
                          alpha=0.9, eps=1e-05, weight_decay=optimState['weigthDecay'], momentum=0.0, centered=False)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,9,12,15], gamma=0.25)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4,
#                                                        verbose=True, threshold=0.0001, threshold_mode='rel', \
#                                                        cooldown=0, min_lr=0, eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
# if torch.cuda.is_available():
#     checkpoint = torch.load("/content/gdrive/My Drive/" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
# else:
#     checkpoint = torch.load("./" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# epoch = checkpoint['epoch'] + 1
# loss = checkpoint['loss']
epoch = 0
print(net)
print(device)
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
        batch_label, batch_data = gen_batch(batch_size, N, M, K, NOISE, H, R)
        batch_label.to(device)
        optimizer.zero_grad()
        pred_prob = net(batch_data, batch_zero_states).to(device)  # 0 or 1?!
        err = LOSS(pred_prob, batch_label.to(device))
        err.backward()
        with torch.no_grad():
            for name, param in net.named_parameters():
                # print(name)
                # print(param.grad.data)
                param.grad.clamp_(-4.0, 4.0)
                gnorm = param.grad.norm()
                if (gnorm > max_grad_norm):
                    param.grad.mul_(max_grad_norm / gnorm)
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
        if (nbatch) % 255 == 1:
            print("Epoch " + str(epoch) + " Batch " + str(nbatch) + " {:.4} {:.4} {:.4} loss = {:.4}".format(batch_accs,
                                                                                                             batch_accl,
                                                                                                             batch_accm,
                                                                                                             err.item()))
            # for param_group in optimizer.param_groups:
            #     print(param_group['lr'])
    end = time.time()
    print("Train [{}] Time {} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}".format(epoch, end - start, \
                                                                                    train_accs / nbatch,
                                                                                    train_accl / nbatch, \
                                                                                    train_accm / nbatch,
                                                                                    train_err / nbatch))
    logger.write("Train [{}] Time {:.4} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}\n".format(epoch, end - start, \
                                                                                                train_accs / nbatch,
                                                                                                train_accl / nbatch, \
                                                                                                train_accm / nbatch,
                                                                                                train_err / nbatch))

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
        pred_prob = net(batch_data, batch_zero_states)
        err = LOSS(pred_prob, batch_label)
        batch_accs = AccS(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accl = AccL(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accm = AccM(batch_label[:, range(0, num_nonz)], pred_prob.float())
        valid_accs = valid_accs + batch_accs.item()
        valid_accl = valid_accl + batch_accl.item()
        valid_accm = valid_accm + batch_accm
        valid_err = valid_err + err.item()
        nbatch = nbatch + 1
#     scheduler.step(valid_err / nbatch)
    scheduler.step()
    #         if (nbatch+99) % 100 == 0:
    #             print("Eval Epoch " + str(epoch) + " Batch " + str(nbatch) + " {:.4} {:.4} {:.4} loss = {:.4}".format(batch_accs, batch_accl,
    #                                                                                             batch_accm, err.item()))
    end = time.time()
    print("Valid [{}] Time {} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}".format(epoch, end - start, \
                                                                                    valid_accs / nbatch,
                                                                                    valid_accl / nbatch, \
                                                                                    valid_accm / nbatch,
                                                                                    valid_err / nbatch))
    logger.write("Valid [{}] Time {} s-acc {:.4} l-acc {:.4} m-acc {:.4} err {:.4}\n".format(epoch, end - start, \
                                                                                             valid_accs / nbatch,
                                                                                             valid_accl / nbatch, \
                                                                                             train_accm / nbatch,
                                                                                             valid_err / nbatch))
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
    if torch.cuda.is_available():
        torch.save(checkpoint, "/content/gdrive/My Drive/" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    else:
        torch.save(checkpoint, "./" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    logger.close()



