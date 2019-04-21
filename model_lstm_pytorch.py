import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from mat4py import loadmat
#from torchsummary import summary
from graphviz import Digraph
from torchviz import make_dot
from graphviz import Source

import time

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

        # self.l_nb = nn.ModuleList(l_bn_lst)
        # for L in range(num_layers):
        #     setattr(self, 'layer_i2h_%d' % L, self.l_i2h)
        #     setattr(self, 'layer_h2h_%d' % L, self.l_h2h)
        #     # setattr(self, 'layer_bnorm_%d' % L, self.l_bn)
        self.next_hs = []
        self.next_cs = []
        sigmoid_lst = [torch.nn.Sigmoid()]
        tanh_lst = [torch.nn.Tanh()]
        for L in range(1,self.num_layers):
            sigmoid_lst.append(torch.nn.Sigmoid())
            tanh_lst.append(torch.nn.Tanh())

        self.sigmoid = nn.ModuleList(sigmoid_lst)
        self.tanh = nn.ModuleList(tanh_lst)
        self.i2h = []
        self.h2h = []

        l_i2h_lst = [nn.Linear(self.input_size, 4 * self.rnn_size)]
        l_h2h_lst = [nn.Linear(self.rnn_size, 4 * self.rnn_size)]
        # l_bn_lst = [nn.BatchNorm1d(4 * self.rnn_size)]
        for L in range(1, self.num_layers):
            l_i2h_lst.append(nn.Linear(self.rnn_size, 4 * self.rnn_size))
            l_h2h_lst.append(nn.Linear(self.rnn_size, 4 * self.rnn_size))

            # l_bn_lst.append(nn.BatchNorm1d(4 * self.rnn_size))
        self.l_i2h = nn.ModuleList(l_i2h_lst)
        self.l_h2h = nn.ModuleList(l_h2h_lst)

    def forward(self, x, prev_hs, prev_cs):
        self.next_hs = []
        self.next_cs = []
        self.i2h = []
        self.h2h = []
        for L in range(self.num_layers):
            self.prev_c = prev_cs[L]
            self.prev_h = prev_hs[L]
            if L == 0:
                self.x = x
            else:
                self.x = self.next_hs[L - 1]
            self.i2h.append(self.l_i2h[L](self.x))
            self.h2h.append(self.l_h2h[L](self.prev_h))
            all_sums = self.i2h[L] + self.h2h[L]
            (n1, n2, n3, n4) = all_sums.chunk(4, dim=1)  # it should return 4 tensors self.rnn_size
            in_gate = self.sigmoid[L](n1)
            forget_gate = self.sigmoid[L](n2)
            out_gate = self.sigmoid[L](n3)
            in_transform = self.tanh[L](n4)
            next_c = forget_gate*self.prev_c + in_gate*in_transform
            next_h = out_gate * self.tanh[L](next_c)

            self.next_hs.append(next_h)
            self.next_cs.append(next_c)
        return torch.stack(self.next_hs), torch.stack(self.next_cs)#, i2h, h2h


class BuildLstmUnrollNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, input_size):
        super(BuildLstmUnrollNet, self).__init__()

        self.num_unroll = num_unroll
        self.num_layers = num_layers
        self.rnn_size = rnn_size
        self.input_size = input_size

        self.init_hs = []
        self.init_cs = []

        self.out = []
        self.out_states_lst = []
        self.outputs = []
        self.out_states = []
        self.output = []


        self.now_h, self.now_c = [], []
        self.init_hs = []
        self.init_cs = []
        self.outputs = []
        self.out_states_lst = []
        self.i2h_lst = []
        self.h2h_lst = []
        self.now_hs = []
        self.now_cs = []
        self.buildlstmstack_lst = []
        for i in range(0, self.num_unroll):
            self.buildlstmstack_lst.append(BuildLstmStack(self.input_size, self.rnn_size, self.num_layers))
        self.buildlstmstack = nn.ModuleList(self.buildlstmstack_lst)

    def forward(self, x, init_states_input):

        self.init_hs = []
        self.init_cs = []
        self.now_hs = []
        self.now_cs = []
        self.outputs = []

        init_states = init_states_input.reshape((init_states_input.size(0),self.num_layers * 2, self.rnn_size))
        init_states_lst = list(init_states.chunk(self.num_layers * 2,1))

        for i in range(self.num_layers):
            self.init_hs.append(init_states_lst[2*i].reshape(init_states_input.size(0),self.rnn_size))
            self.init_cs.append(init_states_lst[2*i+1].reshape(init_states_input.size(0),self.rnn_size))

        self.now_hs.append(torch.stack(self.init_hs))
        self.now_cs.append(torch.stack(self.init_cs))

        for i in range(self.num_unroll):
            self.now_h, self.now_c = self.buildlstmstack[i](x, self.now_hs[i], self.now_cs[i])
            self.now_hs.append(self.now_h)
            self.now_cs.append(self.now_c)
            #self.outputs.append(torch.cat(self.now_hs[-1],1))
            self.outputs.append(self.now_hs[i][-1])#.reshape((self.num_layers, init_states_input.size(0), self.rnn_size)))
            # for L in range(self.num_layers):
            #     setattr(self, 'hid_%d_%d' %(i, L), self.now_hs[i][L])
            #     setattr(self, 'cell_%d_%d' %(i, L), self.now_cs[i][L])

        # for i in range(self.num_layers):
        #     self.out_states_lst.append(self.now_hs[i])
        #     self.out_states_lst.append(self.now_cs[i])

        # for i in range(1,self.num_unroll):
        #     for j in range(self.num_layers):
        #         self.buildlstmstack_lst[i].l_i2h[j] = self.buildlstmstack_lst[0].l_i2h[j]
        #         self.buildlstmstack_lst[i].l_h2h[j] = self.buildlstmstack_lst[0].l_h2h[j]

        #self.output = torch.cat(self.outputs, 2)
        self.output = self.outputs[0]
        for i in range(1, self.num_unroll):
            self.output = torch.cat((self.output, self.outputs[i]),1)
        #print(self.output.shape)

        return self.output


class GetLstmNet(nn.Module):

    def __init__(self, num_unroll, num_layers, rnn_size, output_size, input_size):
        super(GetLstmNet,self).__init__()

        self.lstm_input = []
        self.lstm_output = []
        self.init_states = []
        self.out_states = []
        self.num_unroll, self.num_layers, self.rnn_size, self.output_size, self.input_size  = num_unroll, num_layers, rnn_size, output_size, input_size
        self.l_pred_l = nn.Linear(self.num_unroll * self.rnn_size, self.output_size)
        #self.l_pred_bn = nn.BatchNorm1d(self.output_size)
        self.pred = []
        self.lstmnet = BuildLstmUnrollNet(self.num_unroll, self.num_layers, self.rnn_size, self.input_size)
        # setattr(self, 'LstmUnrollNet', self.lstmnet)
        # setattr(self, 'LstmNetLinear', self.l_pred_l)

    def forward(self, x, init_states_input):
        # print(x.shape)
        # print(init_states_input.shape)
        self.lstm_output = self.lstmnet(x, init_states_input)
        # print(self.lstm_output.shape)
        self.pred = self.l_pred_l(self.lstm_output)
        # print(self.pred.shape)
        return self.pred

###########Usage#######################################

input_size = 20
output_size = 100
rnn_size = 100
num_layers = 3
num_unroll = 5
#graph of net
# x = torch.rand(3,20)
# z = torch.zeros(3,rnn_size * num_layers * 2)
# model = BuildLstmStack(input_size, rnn_size, num_layers)
# init_hs = []
# init_cs = []
# init_states = z.reshape((z.size(0),num_layers * 2, rnn_size))
# init_states_lst = list(init_states.chunk(num_layers * 2,1))
# for i in range(num_layers):
#     init_hs.append(init_states_lst[2*i].reshape(num_layers,rnn_size))
#     init_cs.append(init_states_lst[2*i+1].reshape(num_layers,rnn_size))
# now_hs, now_cs = model(x, init_hs, init_cs)
# temp = make_dot((now_hs[2], now_cs[2]), params=dict(list(model.named_parameters())))
# s = Source(temp, filename="BuildLstmStack.gv", format="png")
# s.view()
#
# model = BuildLstmUnrollNet(num_unroll, num_layers, rnn_size, input_size)
# out = model(x, z)
# temp = make_dot(out, params=dict(list(model.named_parameters())+ [('x', x)]+ [('z', z)]))
# s = Source(temp, filename="BuildLstmUnrollNet.gv", format="png")
# s.view()
#
# model = GetLstmNet(num_unroll, num_layers, rnn_size, output_size, input_size)
# output = model(x,z)
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
        self.lsm = nn.LogSoftmax()
        self.nll = nn.NLLLoss()
        self.output = 0
        self.outputs = 0

    def forward(self, inputs, target):
        self.output = self.lsm(inputs)
        shape = target.shape
        self.outputs = 0
        # print(self.output.shape)
        # print(target.shape)
        for i in range(0,shape[1]):
            self.outputs = self.outputs + self.nll(self.output,target[:,i].squeeze())
        return self.outputs/shape[1]


#match number
def AccS(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(num_nonz) #?!
    pred = pred.float()
    t_score = torch.zeros(label.shape)
    for i in range(0, num_nonz):
        for j in range(0, num_nonz):
            t_score[:,i].add(label[:,i].float().eq(pred[:,j]).float())
    return t_score.mean()
#loose match
def AccL(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(20) #?!
    pred = pred.float()
    t_score = torch.zeros(label.shape)
    for i in range(0, num_nonz):
        for j in range(0, 20):
            t_score[:,i].add(label[:,i].float().eq(pred[:,j]).float())#t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.mean()
#sctrict match
def AccM(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(num_nonz) #?!
    pred = pred.float()
    t_score = torch.zeros(label.shape)
    for i in range(0, num_nonz):
        for j in range(0, num_nonz):
            t_score[:,i].add(label[:,i].float().eq(pred[:,j]).float())#t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.sum(1).eq(num_nonz).sum() * 1./ pred.shape[0]

gpu = 1 # gpu id
batch_size = 10#250 # training batch size
lr = 0.001 # basic learning rate
lr_decay_startpoint = 250 #learning rate from which epoch
num_epochs = 2#00 # total training epochs
max_grad_norm = 5.0
clip_gradient = 4.0

# task related parameters
# task: y = Ax, given A recovery sparse x from y
dataset = 'uniform' # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
num_nonz = 3 # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
input_size = 20 # dimension of observation vector y
output_size = 100 # dimension of sparse vector x

# model hyper parameters
rnn_size = 425 # number of units in RNN cell
num_layers = 2 # number of stacked RNN layers
num_unroll = 11 # number of RNN unrolled time steps

torch.set_num_threads(2)
manualSeed = torch.randint(1,10000,(1,))
print("Random seed " + str(manualSeed.item()))
torch.set_default_tensor_type(torch.FloatTensor)

train_size = 100#600000
valid_size = 10#100000
valid_data = torch.zeros(valid_size, input_size)
valid_label = torch.zeros(valid_size, num_nonz)
batch_data = torch.zeros(batch_size, input_size)
batch_label = torch.zeros(batch_size, num_nonz) # for MultiClassNLLCriterion LOSS
batch_zero_states = torch.zeros(batch_size, num_layers * rnn_size * 2) #init_states for lstm

#AccM, AccL, Accs = 0, 0, 0


err = 0



model_all = "model_l_"+str(num_layers)+"t_"+str(num_unroll)+'_rnn_'+ str(rnn_size)
logger_file = model_all+str(dataset)+"_"+str(num_nonz)+'.log'
logger = open(logger_file, 'w')
#for k,v in pairs(opt) do logger:write(k .. ' ' .. v ..'\n') end
#logger:write('network have ' .. paras:size(1) .. ' parameters' .. '\n')
#logger:close()




def gen_batch(batch_size, num_nonz):
    mat_A = loadmat('matrix_corr_unit_20_100.mat')
    mat_A = torch.FloatTensor(mat_A['A']).t()
    #print(mat_A.shape)
    batch_X = torch.Tensor(batch_size, 100)
    batch_n = torch.Tensor(batch_size, num_nonz)
    bs = batch_size
    len = int(100 / num_nonz*num_nonz)
    perm = torch.randperm(100)[range(len)]
    batch_label = torch.zeros(batch_size, num_nonz)  # for MultiClassNLLCriterion LOSS
    for i in range(int(bs*num_nonz/len)):
        perm = torch.cat((perm, torch.randperm(100)[range(len)]))
    batch_label.copy_(perm[range(bs*num_nonz)].reshape([bs, num_nonz]))
    batch_label = batch_label.type(torch.LongTensor)
    batch_X.zero_()
    if dataset == 'uniform':
        batch_n.uniform_(-1,1)
        batch_n[batch_n.gt(0)] = 1
        batch_n[batch_n.le(0)] = -1
    #
    #print(batch_X.shape)
    for i in range(bs):
        for j in range(num_nonz):
            batch_X[i][batch_label[i][j]] = batch_n[i][j]
    batch_data.copy_(torch.mm(batch_X, mat_A))
    # print(batch_label.shape)
    # print(batch_data.shape)
    return batch_label, batch_data

print("building validation set")
for i in range(0, valid_size, batch_size):
    batch_label, batch_data = gen_batch(batch_size, num_nonz)
    # print(batch_label.shape)
    # print("batch_data shape = " + str(batch_data.shape))
    # print("valid_data shape = " + str(valid_data.shape))
    # print(range(i,i+batch_size-1))
    valid_data[range(i,i+batch_size), :].copy_(batch_data)
    valid_label[range(i, i + batch_size)][:].copy_(batch_label)
print('done')

best_valid_accs = 0
base_epoch = lr_decay_startpoint
base_lr = lr
optimState = {'learningRate' : 0.001, 'weigthDecay' : 0.001}

net = GetLstmNet(num_unroll, num_layers, rnn_size, output_size, input_size)
print(net)
device = torch.device('cpu')
net.to(device)
#summary(net,[(num_layers,input_size),(num_layers,rnn_size * num_layers * 2)])
# summary(net,[(batch_size, input_size),(batch_size, num_layers * rnn_size * 2)])

# create a stochastic gradient descent optimizer
optimizer = optim.RMSprop(net.parameters(), lr=lr)
# create a loss function
LOSS = MultiClassNLLCriterion()


for epoch in range(1,num_epochs):
    #learing rate self - adjustment
    if(epoch > 250):
        optimState['learningRate'] = base_lr / (1 + 0.06 * (epoch - base_epoch))
        if(epoch % 50 == 0): base_epoch = epoch; base_lr= base_lr * 0.25
    logger = open(logger_file, 'a')
    #train
    train_accs = 0
    train_accl = 0
    train_accm = 0
    train_err = 0
    nbatch = 0
    net.train()
    for i in range(0,train_size,batch_size):
        start = time.time()
        batch_label, batch_data = gen_batch(batch_size, num_nonz)

        optimizer.zero_grad()
        pred_prob = net(batch_data, batch_zero_states) #0 or 1?!

        err = LOSS(pred_prob, batch_label)
        print("loss = "+str(err.item()))
        df_dpred = err.backward()#retain_graph=True)
        # with torch.no_grad():
        #     for param in net.parameters(True):
        #         param.clamp_(-4,4)
        #         gnorm = param.norm()
        #         if(gnorm > max_grad_norm):
        #             param.mul_(max_grad_norm/gnorm)
        #net.l_pred_l.weight.data.clamp_(-0.01,0.01)
        # for param in self.net.parameters():
        #     param.grad.data.clamp(-1, 1)
        optimizer.step()
        batch_accs = AccS(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accl = AccL(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accm = AccM(batch_label[:, range(0, num_nonz)], pred_prob.float())
        train_accs = train_accs + batch_accs
        train_accl = train_accl + batch_accl
        train_accm = train_accm + batch_accm
        train_err = train_err + err
        nbatch = nbatch + 1
        if nbatch % 512 == 1:
            print("{} {} {} err {}\n".format(batch_accs, batch_accl, batch_accm, err))
    end = time.time()
    print("Train {} Time {} s-acc {} l-acc {} m-acc {} err {}\n".format(epoch, end - start, \
                                                                        train_accs / nbatch, train_accl / nbatch,\
                                                                        train_accm / nbatch, train_err / nbatch))
    logger.write("Train {} Time {} s-acc {} l-acc {} m-acc {} err {}\n".format(epoch, end - start, \
                                                                        train_accs / nbatch, train_accl / nbatch,\
                                                                        train_accm / nbatch, train_err / nbatch))

    #eval
    nbatch = 0
    valid_accs = 0
    valid_accl = 0
    valid_accm = 0
    valid_err = 0
    for i in range(0,valid_size,batch_size):
        batch_data.copy_(valid_data[range(i,i+batch_size),:])
        batch_label[:,range(0, num_nonz)].copy_(valid_label[range(i, i + batch_size), :])
        net.eval()
        pred_prob = net(batch_data,batch_zero_states).float()
        err = LOSS(pred_prob, batch_label)
        batch_accs = AccS(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accl = AccL(batch_label[:, range(0, num_nonz)], pred_prob.float())
        batch_accm = AccM(batch_label[:, range(0, num_nonz)], pred_prob.float())
        valid_accs = valid_accs + batch_accs
        valid_accl = valid_accl + batch_accl
        valid_accm = valid_accm + batch_accm
        valid_err = valid_err + err
        nbatch = nbatch + 1
    print("Valid {} Time {} s-acc {} l-acc {} m-acc {} err {}\n".format(epoch, end - start, \
                                                                        valid_accs / nbatch, valid_accl / nbatch,\
                                                                        valid_accm / nbatch, valid_err / nbatch))
    logger.write("Valid {} Time {} s-acc {} l-acc {} m-acc {} err {}\n".format(epoch, end - start, \
                                                                        valid_accs / nbatch, valid_accl / nbatch,\
                                                                        train_accm / nbatch, valid_err / nbatch))
    if(valid_accs > best_valid_accs):
        best_valid_accs = valid_accs
        print("saving model")
        logger.write('saving model\n')
        torch.save(net.state_dict(), "./checkpoints/"+model_all+"."+str(num_nonz)+".pt") #or torch.save(net, PATH)
        #net.load_state_dict(torch.load(PATH)) # or the_model = torch.load(PATH)

    if(epoch % 100 == 0):
        print("saving model")
        torch.save(net.state_dict(), "./checkpoints/" + model_all + "." + str(num_nonz) + "."+str(epoch)+".pt")  # or torch.save(net, PATH)

    logger.close()
    if epoch == lr_decay_startpoint:
        optimState["learningRate"] = 0.001
        optimState["weightDecay"] = 0.001


#print(end - start)
















# # run the main training loop
# for epoch in range(epochs):
#     for batch_idx, (data, target) in enumerate(train_loader):
#     data, target = data, target
#     # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
#     data = data.view(-1, 28*28)
#     optimizer.zero_grad()
#     net_out = model(data)
#     loss = criterion(net_out, target)
#     loss.backward()
#     optimizer.step()
#     if batch_idx % log_interval == 0:
#         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         epoch, batch_idx * len(data), len(train_loader.dataset),
#         100. * batch_idx / len(train_loader), loss.data[0]))











