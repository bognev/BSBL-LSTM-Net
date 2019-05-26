import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
# from mat4py import loadmat
# #from torchsummary import summary
# from graphviz import Digraph
# from torchviz import make_dot
# from graphviz import Source

import time

# from google.colab import drive
# 
# drive.mount("/content/gdrive", force_remount=True)




class BuildGFGRUStack(nn.Module):

    def __init__(self, input_size, rnn_size, num_layers):
        super(BuildGFGRUStack, self).__init__()

        self.input_size = input_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        l_i2h_lst = [nn.Linear(self.input_size, 2 * self.rnn_size)]
        l_h2h_lst = [nn.Linear(self.rnn_size, 2 * self.rnn_size)]  * self.num_layers
        l_wj1j_lst = [nn.Linear(self.input_size, self.rnn_size)]
        # l_bn_lst = [nn.BatchNorm1d(2 * self.rnn_size)]
        # self.l_do = nn.Dropout(0.25)
        l_wg_lst = [[nn.Linear(self.rnn_size, 1)] * self.num_layers] * (self.num_layers-1)
        l_ug_lst = [[nn.Linear(self.num_layers * self.rnn_size, 1)] * self.num_layers] * self.num_layers
        l_wj1j_lst = [nn.Linear(self.rnn_size, self.rnn_size)] * self.num_layers
        for L in range(1, self.num_layers):
            l_i2h_lst.append(nn.Linear(self.rnn_size, 2 * self.rnn_size))
            l_h2h_lst.append(nn.Linear(self.rnn_size, 2 * self.rnn_size))
            l_wj1j_lst.append(nn.Linear(self.rnn_size, self.rnn_size))
            # l_bn_lst.append(nn.BatchNorm1d(2 * self.rnn_size))
        self.l_i2h = nn.ModuleList(l_i2h_lst)
        self.l_h2h = nn.ModuleList(l_h2h_lst)
        self.l_wj1j = nn.ModuleList(l_wj1j_lst)
        # self.l_bn = nn.ModuleList(l_bn_lst)
        self.l_wg = [nn.ModuleList([nn.Linear(self.input_size, 1)] * self.num_layers)]
        self.l_ug = [nn.ModuleList(l_ug_lst[L])]
        for L in range(0, self.num_layers-1):
            self.l_wg.append(nn.ModuleList(l_wg_lst[L]))
            self.l_ug.append(nn.ModuleList(l_ug_lst[L]))




    def forward(self, x, prev_hs):
        self.x_size = []
        self.prev_h = 0
        self.next_hs = []
        self.i2h = []
        self.h2h = []
        self.g = torch.zeros(self.num_layers, self.num_layers, x.shape[0])
        # for gg in range(1, self.num_layers):
        h_stacked = prev_hs[0]
        for L in range(1,self.num_layers):
            h_stacked = torch.cat((h_stacked, prev_hs[L]),1)
        for L in range(self.num_layers):
            self.prev_h = prev_hs[L]
            g_l_acc = 0
            if L == 0:
                self.x = x
            else:
                self.x = self.next_hs[L - 1]
            for gg in range(self.num_layers):
                self.g[L,gg] = torch.squeeze(torch.tanh(self.l_wg[L][gg](self.x) + self.l_ug[L][gg](h_stacked)))
                g_l_acc = g_l_acc + self.g[L,gg]*self.l_uij[L][gg](prev_hs[gg])
            self.i2h.append((self.l_i2h[L](self.x)))
            self.h2h.append((self.l_h2h[L](self.prev_h)))
            Wx1, Wx2 = self.i2h[L].chunk(2, dim=1) # it should return 4 tensors self.rnn_size
            Uh1, Uh2 = self.h2h[L].chunk(2, dim=1)
            zt = torch.sigmoid(Wx1 + Uh1)
            rt = torch.sigmoid(Wx2 + Uh2)
            h_candidate = torch.tanh(self.l_wj1j[L]*self.x + rt * g_l_acc)

            h_candidate = 1
            ht = (1-zt) * self.prev_h + zt * h_candidate
            self.next_hs.append(ht)
        return torch.stack(self.next_hs)


class BuildGFGRUUnrollNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, input_size):
        super(BuildGFGRUUnrollNet, self).__init__()
        self.num_unroll = num_unroll
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.input_size = input_size
        self.outputs = []
        self.output = []
        self.now_h = []
        self.buildGFGRUstack_lst = []
        for i in range(0, self.num_unroll):
            self.buildGFGRUstack_lst.append(BuildGFGRUStack(self.input_size, self.rnn_size, self.num_layers))
        self.buildGFGRUstack = nn.ModuleList(self.buildGFGRUstack_lst)

    def forward(self, x, init_states_input):

        self.init_hs = []
        self.now_hs = []
        self.outputs = []

        init_states = init_states_input.reshape((init_states_input.size(0), self.num_layers, self.rnn_size))
        init_states_lst = list(init_states.chunk(self.num_layers, 1))

        for i in range(self.num_layers):
            self.init_hs.append(init_states_lst[i].reshape(init_states_input.size(0), self.rnn_size))

        self.now_hs.append(torch.stack(self.init_hs))

        for i in range(self.num_unroll):
            self.now_h = self.buildGFGRUstack[i](x, self.now_hs[i])
            self.now_hs.append(self.now_h)
            self.outputs.append(self.now_hs[i + 1][-1])
            # for L in range(self.num_layers):
            #     setattr(self, 'hid_%d_%d' %(i, L), self.now_hs[i][L])
            #     setattr(self, 'cell_%d_%d' %(i, L), self.now_cs[i][L])
        # for i in range(1, self.num_unroll):
        #     for j in range(self.num_layers):
        #         self.buildGFGRUstack[i].l_i2h[j].weight.data = self.buildGFGRUstack[0].l_i2h[j].weight.data
        #         self.buildGFGRUstack[i].l_h2h[j].weight.data = self.buildGFGRUstack[0].l_h2h[j].weight.data
        #         self.buildGFGRUstack[i].l_i2h[j].bias.data = self.buildGFGRUstack[0].l_i2h[j].bias.data
        #         self.buildGFGRUstack[i].l_h2h[j].bias.data = self.buildGFGRUstack[0].l_h2h[j].bias.data
        #         self.buildGFGRUstack[i].l_i2h[j].weight.grad = self.buildGFGRUstack[0].l_i2h[j].weight.grad
        #         self.buildGFGRUstack[i].l_h2h[j].weight.grad = self.buildGFGRUstack[0].l_h2h[j].weight.grad
        #         self.buildGFGRUstack[i].l_i2h[j].bias.grad = self.buildGFGRUstack[0].l_i2h[j].bias.grad
        #         self.buildGFGRUstack[i].l_h2h[j].bias.grad = self.buildGFGRUstack[0].l_h2h[j].bias.grad
        self.output = self.outputs[0]
        for i in range(1, self.num_unroll):
            self.output = torch.cat((self.output, self.outputs[i]), 1)

        return self.output


class GetGFGRUNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, output_size, input_size):
        super(GetGFGRUNet, self).__init__()
        self.num_unroll, self.num_layers, self.rnn_size, self.output_size, self.input_size = num_unroll, num_layers, rnn_size, output_size, input_size
        self.l_pred_l = nn.Linear(self.num_unroll * self.rnn_size, self.output_size)
        self.GFGRUnet = BuildGFGRUUnrollNet(self.num_unroll, self.num_layers, self.rnn_size, self.input_size)
        self.l_pred_bn = nn.BatchNorm1d(self.output_size)
        # setattr(self, 'GFGRUNetLinear', self.l_pred_l)

    def forward(self, x, init_states_input):
        self.GFGRU_output = self.GFGRUnet(x, init_states_input)
        self.pred = self.l_pred_bn(self.l_pred_l(self.GFGRU_output))
        return self.pred


###########Usage#######################################

input_size = 20
output_size = 50
rnn_size = 10
num_layers = 2
num_unroll = 3
# graph of net
x = torch.rand(3, input_size)
z = torch.zeros(3, rnn_size * num_layers * 2)


# model = BuildGFGRUStack(input_size, rnn_size, num_layers)
# init_hs = []
# init_cs = []
# init_states = z.reshape((z.size(0),num_layers * 2, rnn_size))
# init_states_lst = list(init_states.chunk(num_layers * 2,1))
# for i in range(num_layers):
#     init_hs.append(init_states_lst[2*i].reshape(num_layers,rnn_size))
#     init_cs.append(init_states_lst[2*i+1].reshape(num_layers,rnn_size))
# now_hs, now_cs = model(x, init_hs, init_cs)
# temp = make_dot((now_hs[2], now_cs[2]), params=dict(list(model.named_parameters())))
# s = Source(temp, filename="BuildGFGRUStack.gv", format="png")
# s.view()
#
# model = BuildGFGRUUnrollNet(num_unroll, num_layers, rnn_size, input_size)
# out = model(x, z)
# temp = make_dot(out, params=dict(list(model.named_parameters())+ [('x', x)]+ [('z', z)]))
# s = Source(temp, filename="BuildGFGRUUnrollNet.gv", format="png")
# s.view()
#
# model = GetGFGRUNet(num_unroll, num_layers, rnn_size, output_size, input_size)
# output = model(x,z)
# for i in range(1, num_unroll):
#     for j in range(num_layers):
#         model.GFGRUnet.buildGFGRUstack[i].l_i2h[j].weight = model.GFGRUnet.buildGFGRUstack[0].l_i2h[j].weight
#         model.GFGRUnet.buildGFGRUstack[i].l_h2h[j].weight = model.GFGRUnet.buildGFGRUstack[0].l_h2h[j].weight
#         model.GFGRUnet.buildGFGRUstack[i].l_i2h[j].bias = model.GFGRUnet.buildGFGRUstack[0].l_i2h[j].bias
#         model.GFGRUnet.buildGFGRUstack[i].l_h2h[j].bias = model.GFGRUnet.buildGFGRUstack[0].l_h2h[j].bias
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

# if torch.cuda.is_available():
#     batch_size = 250  # 10# training batch size
# else:
batch_size = 5  # 600000  #
lr = 0.002  # basic learning rate
lr_decay_startpoint = 250  # learning rate from which epoch
num_epochs = 400  # total training epochs
max_grad_norm = 5.0
clip_gradient = 4.0

# task related parameters
# task: y = Ax, given A recovery sparse x from y
dataset = 'uniform'  # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
num_nonz = 3  # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
input_size = 20  # dimension of observation vector y
output_size = 100  # dimension of sparse vector x

# model hyper parameters
rnn_size = 200  # number of units in RNN cell
num_layers = 3  # number of stacked RNN layers
num_unroll = 5  # number of RNN unrolled time steps

# torch.set_num_threads(16)
# manualSeed = torch.randint(1,10000,(1,))
# print("Random seed " + str(manualSeed.item()))
torch.set_default_tensor_type(torch.FloatTensor)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# if torch.cuda.is_available():
#     train_size = 600000  #
#     valid_size = 100000  #
# else:
train_size = 100  # 600000  #
valid_size = 10  # 100000  #
valid_data = torch.zeros(valid_size, input_size).to(device)
valid_label = torch.zeros(valid_size, num_nonz).type(torch.LongTensor).to(device)
batch_data = torch.zeros(batch_size, input_size).to(device)
batch_label = torch.zeros(batch_size, num_nonz).to(device)  # for MultiClassNLLCriterion LOSS
batch_zero_states = torch.zeros(batch_size, num_layers * rnn_size).to(device)  # init_states for GFGRU

# AccM, AccL, Accs = 0, 0, 0


err = 0

model_all = "model_l_" + str(num_layers) + "t_" + str(num_unroll) + '_gru_' + str(rnn_size)
logger_file = model_all + str(dataset) + "_" + str(num_nonz) + '.log'
# if torch.cuda.is_available():
#     logger_file = "/content/gdrive/My Drive/" + logger_file  # or torch.save(net, PATH)
# else:
logger_file = "./" + logger_file
logger = open(logger_file, 'w')
# for k,v in pairs(opt) do logger:write(k .. ' ' .. v ..'\n') end
# logger:write('network have ' .. paras:size(1) .. ' parameters' .. '\n')
# logger:close()

# torch.manual_seed(10)
# mat_A = torch.rand(output_size,input_size)
# if torch.cuda.is_available():
#     mat_A = torch.load("/content/gdrive/My Drive/mat_A.pt").to(device)
# else:
mat_A = torch.load("./mat_A.pt").to(device)
# mat_A = torch.load("/content/gdrive/My Drive/mat_A.pt").to(device)



def gen_batch(batch_size, num_nonz, mat_A):
    # mat_A = loadmat('matrix_corr_unit_20_100.mat')
    # mat_A = torch.FloatTensor(mat_A['A']).t()
    # print(mat_A.shape)
    # mat_A = torch.rand(output_size, input_size)
    batch_X = torch.Tensor(batch_size, 100).to(device)
    batch_n = torch.Tensor(batch_size, num_nonz).to(device)
    bs = batch_size
    len = int(100 / num_nonz * num_nonz)
    perm = torch.randperm(100)[range(len)].to(device)
    #     batch_label = torch.zeros(batch_size, num_nonz).type(torch.LongTensor).to(device)  # for MultiClassNLLCriterion LOSS
    for i in range(int(bs * num_nonz / len)):
        perm = torch.cat((perm, torch.randperm(100)[range(len)].to(device)))
    batch_label = perm[range(bs * num_nonz)].reshape([bs, num_nonz]).type(torch.LongTensor).to(device)
    batch_X.zero_()
    if dataset == 'uniform':
        batch_n.uniform_(-0.5, 0.5)
        batch_n[batch_n.gt(0)] = batch_n[batch_n.gt(0)] + 0.1
        batch_n[batch_n.le(0)] = batch_n[batch_n.le(0)] - 0.1
    #
    # print(batch_X.shape)
    #     print(batch_X.get_device())
    #     print(mat_A.get_device())
    #     print(batch_n.get_device())
    for i in range(bs):
        for j in range(num_nonz):
            batch_X[i][batch_label[i][j]] = batch_n[i][j]
    batch_data = torch.mm(batch_X, mat_A)  # +0.001*torch.randn(batch_size,input_size).to(device)
    # print(batch_label.shape)
    # print(batch_data.shape)
    return batch_label, batch_data


print("building validation set")
for i in range(0, valid_size, batch_size):
    #     mat_A = torch.rand(output_size, input_size).to(device)
    batch_label, batch_data = gen_batch(batch_size, num_nonz, mat_A)
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
optimState = {'learningRate': 0.001, 'weigthDecay': 0.0000}

net = GetGFGRUNet(num_unroll, num_layers, rnn_size, output_size, input_size)
# print(net)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
net.to(device)
# summary(net,[(num_layers,input_size),(num_layers,rnn_size * num_layers * 2)])
# summary(net,[(batch_size, input_size),(batch_size, num_layers * rnn_size * 2)])

# create a stochastic gradient descent optimizer
# optimizer = optim.RMSprop(params=net.parameters(), lr=0.001, alpha=0.9, eps=1e-04, weight_decay=0.0001, momentum=0, centered=False)
# create a loss function
LOSS = MultiClassNLLCriterion()
optimizer = optim.RMSprop(params=net.parameters(), lr=optimState['learningRate'], \
                          alpha=0.99, eps=1e-05, weight_decay=optimState['weigthDecay'], momentum=0.0, centered=False)

# checkpoint = torch.load( "/content/gdrive/My Drive/model_l_2t_17_rnn_800_3.pth")
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch'] + 1
# loss = checkpoint['loss']
epoch = 0

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
        batch_label, batch_data = gen_batch(batch_size, num_nonz, mat_A)
        batch_label.to(device)
        optimizer.zero_grad()
        pred_prob = net(batch_data, batch_zero_states).to(device)  # 0 or 1?!
        err = LOSS(pred_prob, batch_label.to(device))
        err.backward()
        with torch.no_grad():
            for name, param in net.named_parameters():
#                 print(name)
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
        if (nbatch) % 512 == 1:
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
    # if torch.cuda.is_available():
    #     torch.save(checkpoint, "/content/gdrive/My Drive/" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    # else:
    torch.save(checkpoint, "./" + model_all + "_" + str(num_nonz) + ".pth")  # or torch.save(net, PATH)
    logger.close()



