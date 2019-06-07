import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from mat4py import loadmat
# #from torchsummary import summary
from graphviz import Digraph
from torchviz import make_dot
from graphviz import Source

import time
HOME = 0
if torch.cuda.is_available() and HOME == 0:
    from google.colab import drive
    drive.mount("/content/gdrive", force_remount=True)




class BuildResNetStack(nn.Module):

    def __init__(self, input_size):
        super(BuildResNetStack, self).__init__()
        self.input_size = input_size
        self.fc_in = nn.Linear(self.input_size,self.input_size)
        self.bn = nn.BatchNorm1d(self.input_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(self.input_size,self.input_size)

    def forward(self, x):
        self.x = x
        self.fc_in_x = self.fc_in(self.x)
        self.bn_x = self.bn(self.fc_in_x)
        self.relu_x = self.relu(self.bn_x)
        self.skip = self.x + self.fc_out(self.relu_x)
        self.relu_skip = self.relu(self.skip)
        return self.relu_skip

class BuildResNetStackInterm(nn.Module):

    def __init__(self, input_size, fc_size):
        super(BuildResNetStackInterm, self).__init__()
        self.input_size = input_size
        self.fc_size = fc_size
        self.fc_in = nn.Linear(self.input_size,self.input_size)
        self.bn = nn.BatchNorm1d(self.input_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.ModuleList([nn.Linear(self.input_size,self.fc_size), nn.Linear(self.input_size,self.fc_size)])

    def forward(self, x):
        self.x = x
        self.fc_in_x = self.fc_in(self.x)
        self.bn_x = self.bn(self.fc_in_x)
        self.relu_x = self.relu(self.bn_x)
        self.skip = self.fc_out[1](self.x) + self.fc_out[0](self.relu_x)
        self.relu_skip = self.relu(self.skip)
        return self.relu_skip


class BuildResNetUnrollNet(nn.Module):

    def __init__(self, num_unroll, input_size, fc_size):
        super(BuildResNetUnrollNet, self).__init__()
        self.num_unroll = num_unroll
        self.fc_size = fc_size
        self.input_size = input_size
        self.buildResNetstack_lst_in = [BuildResNetStack(self.input_size)] * self.num_unroll
        self.buildResNetstack_lst_out = [BuildResNetStack(self.fc_size)] * self.num_unroll
        self.l_ResNets_in = nn.ModuleList(self.buildResNetstack_lst_in)
        self.l_ResNets_imd = BuildResNetStackInterm(self.input_size, self.fc_size)
        self.l_ResNets_out = nn.ModuleList(self.buildResNetstack_lst_out)

    def forward(self, x):
        self.x = x
        # self.res_in = [self.l_ResNets_in[0](self.x)]
        self.x = self.l_ResNets_in[0](self.x)
        self.x = self.l_ResNets_in[1](self.x)
        self.x = self.l_ResNets_in[2](self.x)
        # for L in range(0, self.num_unroll-1):
        #     self.res_in.append(self.l_ResNets_in[L+1](self.res_in[L]))
        # self.res_int = self.l_ResNets_imd(self.res_in[-1])
        self.res_int = self.l_ResNets_imd(self.x)
        self.res_out = self.l_ResNets_out[0](self.res_int)
        self.res_out = self.l_ResNets_out[1](self.res_out)
        self.res_out = self.l_ResNets_out[2](self.res_out)
        # self.res_out = [self.l_ResNets_out[0](self.res_int)]
        # for L in range(0, self.num_unroll-1):
        #     self.res_out.append(self.l_ResNets_out[L+1](self.res_out[L]))

        return self.res_out
        # return self.res_out[-1]


class GetResNet(nn.Module):

    def __init__(self, num_unroll, input_size, fc_size):
        super(GetResNet, self).__init__()
        self.num_unroll, self.input_size, self.fc_size = num_unroll, input_size, fc_size
        self.l_fc_in = nn.Linear(self.input_size,self.input_size)
        self.l_bn_in = nn.BatchNorm1d(self.input_size)
        self.l_ResNet = BuildResNetUnrollNet(self.num_unroll, self.input_size, self.fc_size)
        self.l_fc_out = nn.Linear(self.fc_size, self.fc_size)

    def forward(self, x):
        self.x = x
        self.fc_in = self.l_fc_in(self.x)
        self.bn_x = self.l_bn_in(self.fc_in)
        self.resnet = self.l_ResNet(self.bn_x)
        self.pred = self.l_fc_out(self.resnet)
        return self.pred


###########Usage#######################################

input_size = 20
output_size = 100
rnn_size = 10
num_layers = 2
num_unroll = 3
model = GetResNet(num_unroll, input_size, output_size)
# graph of net
x = torch.rand(3, input_size)
out = model(x)
print(model)
temp = make_dot(out, params=dict(list(model.named_parameters())+ [('x', x)]))
s = Source(temp, filename="test.gv", format="png")
s.view()


# model = BuildResNetStack(input_size, rnn_size, num_layers)
# init_hs = []
# init_cs = []
# init_states = z.reshape((z.size(0),num_layers * 2, rnn_size))
# init_states_lst = list(init_states.chunk(num_layers * 2,1))
# for i in range(num_layers):
#     init_hs.append(init_states_lst[2*i].reshape(num_layers,rnn_size))
#     init_cs.append(init_states_lst[2*i+1].reshape(num_layers,rnn_size))
# now_hs, now_cs = model(x, init_hs, init_cs)
# temp = make_dot((now_hs[2], now_cs[2]), params=dict(list(model.named_parameters())))
# s = Source(temp, filename="BuildResNetStack.gv", format="png")
# s.view()
#
# model = BuildResNetUnrollNet(num_unroll, num_layers, rnn_size, input_size)
# out = model(x, z)
# temp = make_dot(out, params=dict(list(model.named_parameters())+ [('x', x)]+ [('z', z)]))
# s = Source(temp, filename="BuildResNetUnrollNet.gv", format="png")
# s.view()
#
# model = GetResNet(num_unroll, num_layers, rnn_size, output_size, input_size)
# output = model(x,z)
# for i in range(1, num_unroll):
#     for j in range(num_layers):
#         model.ResNetnet.buildResNetstack[i].l_i2h[j].weight = model.ResNetnet.buildResNetstack[0].l_i2h[j].weight
#         model.ResNetnet.buildResNetstack[i].l_h2h[j].weight = model.ResNetnet.buildResNetstack[0].l_h2h[j].weight
#         model.ResNetnet.buildResNetstack[i].l_i2h[j].bias = model.ResNetnet.buildResNetstack[0].l_i2h[j].bias
#         model.ResNetnet.buildResNetstack[i].l_h2h[j].bias = model.ResNetnet.buildResNetstack[0].l_h2h[j].bias
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
# if HOME == 0:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# else:
#     device = "cpu"


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
num_epochs = 400  # total training epochs
max_grad_norm = 5.0
clip_gradient = 4.0
N = 3  # the number of receivers
M = 3  # the number of transmitters
K = 3  # the number of targets

# task related parameters
# task: y = Ax, given A recovery sparse x from y
dataset = 'uniform'  # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
# num_nonz = K*N*M*2  # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
num_nonz = K  # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
input_size = N*200*2  # dimension of observation vector y
output_size = 12*12  # dimension of sparse vector x
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    train_size = int(256*750*6/2)  #
    valid_size = int(256*750/2)  #
else:
    train_size = 100  # 600000  #
    valid_size = 10  # 100000  #
print(device)
valid_data = torch.zeros(valid_size, input_size).to(device)
valid_label = torch.zeros(valid_size, num_nonz).type(torch.LongTensor).to(device)
batch_data = torch.zeros(batch_size, input_size).to(device)
batch_label = torch.zeros(batch_size, num_nonz).to(device)  # for MultiClassNLLCriterion LOSS
batch_zero_states = torch.zeros(batch_size, num_layers * rnn_size * 2).to(device)  # init_states for GRU


# AccM, AccL, Accs = 0, 0, 0


err = 0

model_all = "model_l_" + str(num_layers) + "t_" + str(num_unroll) + '_ResNet_' + str(rnn_size)
logger_file = model_all + str(dataset) + "_" + str(num_nonz) + '.log'
if torch.cuda.is_available() and HOME == 0:
     logger_file = "/content/gdrive/My Drive/" + logger_file  # or torch.save(net, PATH)
else:
    logger_file = "./" + logger_file
logger = open(logger_file, 'w')

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
def gen_mimo_samples(SNR_dB, M, N, K, NOISE, H):
    c = 3 * 10 ** 8
    dt = 10 ** (-6)
    Ts = 1.6000e-06
    L = int(Ts / dt)
    T = 200
    DB = 10. ** (0.1 * SNR_dB)

    # N = 8  # the number of receivers
    # M = 1  # the number of transmitters

    # K = 1  # the number of targets
    # np.random.seed(15)
    # Position of receivers
    x_r = np.array([1000, 2000, 2500, 2500, 2000, 1000, 500, 500])+550#*np.random.rand(1)# + 500 * (np.random.rand(N) - 0.5))  # \
    # 1500,3000,500,2500,1000,1500,500,3000,\
    # 2500,3500,1000,3500,2000,4000,3000,3000]+500*(np.random.rand(N)-0.5))
    y_r = np.array([500, 500, 1000, 2000, 2500, 2500, 2000, 1500])+550#*np.random.rand(1)# + 500 * (np.random.rand(N) - 0.5))  # \
    # 3500,3500,500,4000,4000,2500,3000,500,\
    # 3500,3000,2000,1000,2000,500,4000,1500]+500*(np.random.rand(N)-0.5))

    # Position of transmitters
    x_t = np.array([0, 4000, 4000, 0, 1500, 0, 4000, 2000])+550#+500*np.random.rand(1)
    y_t = np.array([0, 0, 4000, 4000, 4000, 1500, 1500, 0])+550#+500*np.random.rand(1)

    # NOISE = 1  # on/off noise
    # H = 1  # on/off êîýôôèöèåíòû îòðàæåíèÿ
    rk = np.zeros([K, M, N]);
    tk = np.zeros([K, M, N]);
    tau = np.zeros([K, M, N]);
    if H == 0:
        h = np.ones([K, M, N])
    else:
        h = (np.random.randn(K, M, N) + 1j * np.random.randn(K, M, N)) / np.sqrt(2)

    s = np.zeros([M, L]) + 1j * np.zeros([M, L])
    for m in range(M):
        s[m] = np.exp(1j * 2 * np.pi * (m) * np.arange(L) / M) / np.sqrt(L);
        # sqrt(0.5)*(randn(1,L)+1i*randn(1,L))/sqrt(L);
#     Ls = 875
#     Le = Ls + 125 * 6
#     dx = 125
    Ls = 0
    Le = Ls + 250 * 12
    dx = 250
    dy = dx
    dy = dx
    x_grid = np.arange(Ls, Le, dx)
    y_grid = np.arange(Ls, Le, dy)
    size_grid_x = len(x_grid)
    size_grid_y = len(y_grid)
    grid_all_points = [[i, j] for i in x_grid for j in y_grid]
    r = np.zeros(size_grid_x * size_grid_y * M * N)
    k_random_grid_points_i = np.random.permutation(size_grid_x * size_grid_y)[range(K)]
    k_random_grid_points = np.array([])
    # Position of targets
    x_k = np.zeros([K])
    y_k = np.zeros([K])
    for kk in range(K):
        x_k[kk] = grid_all_points[k_random_grid_points_i.item(kk)][0]
        y_k[kk] = grid_all_points[k_random_grid_points_i.item(kk)][1]


    # Time delays
    for k in range(K):
        for m in range(M):
            for n in range(N):
                tk[k, m, n] = np.sqrt((x_k[k] - x_t[m]) ** 2 + (y_k[k] - y_t[m]) ** 2)
                rk[k, m, n] = np.sqrt((x_k[k] - x_r[n]) ** 2 + (y_k[k] - y_r[n]) ** 2)
                tau[k, m, n] = (tk[k, m, n] + rk[k, m, n]) / c

    r_glob = np.zeros([size_grid_x * size_grid_y * M * N]) + 1j * np.zeros([size_grid_x * size_grid_y * M * N])
    for m in range(M):
        for n in range(N):
            for k in range(K):
                r_glob[k_random_grid_points_i[k]] = DB[k] * h[k, m, n] * \
                                                  np.sqrt(200000000000) * (1 / tk[k, m, n]) * (1 / rk[k, m, n])
            k_random_grid_points = np.append(k_random_grid_points,k_random_grid_points_i)
            k_random_grid_points_i = k_random_grid_points_i + size_grid_x * size_grid_y

    # for m in range(M):
    #     for n in range(N):
    #         k_random_grid_points = np.append(k_random_grid_points,k_random_grid_points[-1] + size_grid_x * size_grid_y)

    r[k_random_grid_points.astype(int)] = 1
    if NOISE == 0:
        x = np.zeros([N, T]) + 1j * np.zeros([N, T])
    else:
        x = (np.random.randn(N, T) + 1j * np.random.randn(N, T)) / np.sqrt(2)

    for k in range(K):
        for m in range(M):
            for n in range(N):
                l = np.floor(tau[k, m, n] / dt)
                l = l.astype(int)
                x[n, range(l, l + L)] = x[n, range(l, l + L)] + DB[k] * s[m, :] * h[k, m, n] * \
                                        np.sqrt(200000000000) * (1 / tk[k, m, n]) * (1 / rk[k, m, n])

    x_flat = x[0, :].transpose();
    for n in range(1, N):
        x_flat = np.concatenate([x_flat, x[n, :].transpose()], axis=0)

    return x_flat, r, r_glob, k_random_grid_points
def gen_batch(batch_size, num_nonz, N, M, K, NOISE, H):
#     NOISE = 1
#     H = 1
#     SNR_dB = torch.randint(30,60,(1,)).item()
    SNR_dB = np.random.rand(3)
    y, rr, rr_glob, label = gen_mimo_samples(SNR_dB, M, N, K, NOISE, H)
    batch_data = torch.zeros(batch_size, 2*y.shape[0]).to(device)
#     batch_label = torch.zeros(batch_size, 2*label.shape[0]).to(device)
    batch_label = torch.zeros(batch_size, label[range(num_nonz)].shape[0]).to(device)
    r1 = 40
    r2 = 10
    for i in range(batch_size):
#         SNR_dB = ((r1 - r2) * torch.rand((1,)) + r2).item()
        for k in range(K):
            SNR_dB[k] = ((r1 - r2) * np.random.rand(1) + r2)
        y, rr, rr_glob, label = gen_mimo_samples(SNR_dB, M, N, K, NOISE, H)
        batch_data[i] = torch.cat([torch.from_numpy(y.real),torch.from_numpy(y.imag)]).to(device)
#         batch_data[i] = torch.cat([torch.from_numpy(np.abs(y))]).to(device)
#         batch_label[i] = torch.cat([torch.from_numpy(label),torch.from_numpy(label+M*N*36)]).to(device)
        batch_label[i] = torch.cat([torch.from_numpy(label[range(num_nonz)])]).to(device)


    return batch_label.type(torch.LongTensor).to(device), batch_data

print("building validation set")
for i in range(0, valid_size, batch_size):
    #     mat_A = torch.rand(output_size, input_size).to(device)
    # batch_label, batch_data = gen_batch(batch_size, num_nonz, mat_A)
    batch_label, batch_data = gen_batch(batch_size, num_nonz, N, M, K, 1, 1)
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
optimState = {'learningRate': 0.01, 'weigthDecay': 0.0001}

net = GetResNet(num_unroll, input_size, output_size)
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
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,9,12,15], gamma=0.1)
# checkpoint = torch.load( "/content/gdrive/My Drive/model_l_2t_17_rnn_800_3.pth")
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch'] + 1
# loss = checkpoint['loss']
epoch = 0
print(net)
# mat_A = torch.rand(output_size, input_size).to(device)
for epoch in range(epoch, num_epochs):

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
        batch_label, batch_data = gen_batch(batch_size, num_nonz, N, M, K, 1, 1)
        batch_label.to(device)
        optimizer.zero_grad()
        pred_prob = net(batch_data).to(device)  # 0 or 1?!
        err = LOSS(pred_prob, batch_label.to(device))
        err.backward()
        with torch.no_grad():
            for name, param in net.named_parameters():
                print(name)
                #print(param.grad.data)
                param.grad.clamp_(-4.0, 4.0)
                gnorm = param.grad.norm()
                if (gnorm > max_grad_norm):
                    param.grad.mul_(max_grad_norm / gnorm)
        optimizer.step()
        scheduler.step()
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
                  'loss': err.item()}
    if torch.cuda.is_available():
        torch.save(checkpoint, "/content/gdrive/My Drive/" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    else:
        torch.save(checkpoint, "./" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    logger.close()



