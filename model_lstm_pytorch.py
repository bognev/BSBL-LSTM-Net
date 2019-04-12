import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from mat4py import loadmat
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

# input_size = 20
# output_size = 100
# rnn_size = 100
# num_layers = 3
# num_unroll = 5
# model = GetLstmNet(num_unroll, num_layers, rnn_size, output_size, input_size)
# x = torch.rand(3,20)
# z = torch.zeros(3,rnn_size * num_layers * 2)
# #y = net.train()
# output = model(x,z)
# print(model)


class MultiClassNLLCriterion(torch.nn.Module):

    def __init__(self):
        super(MultiClassNLLCriterion, self).__init__()
        self.lsm = nn.LogSoftmax()
        self.nll = nn.NLLLoss()
        output = 0

    def forward(self, inputs, target):
        outputs = self.lsm(inputs)
        shape = target.shape
        for i in range(1,shape[1]):
            outputs += self.nll(self.output,target)

        return outputs/shape[1]

#match number
def AccS(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(num_nonz,1,True) #?!
    pred = pred.float()
    t_score = torch.zeros(label.shape)
    for i in range(1, num_nonz):
        for j in range(1, num_nonz):
            t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.mean()
#loose match
def AccL(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(20,1,True) #?!
    pred = pred.float()
    t_score = torch.zeros(label.shape)
    for i in range(1, num_nonz):
        for j in range(1, 20):
            t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.mean()
#sctrict match
def AccM(label, pred_prob):
    num_nonz = label.shape[1]
    _, pred = pred_prob.topk(num_nonz,1,True) #?!
    pred = pred.float()
    t_score = torch.zeros(label.shape)
    for i in range(1, num_nonz):
        for j in range(1, 20):
            t_score[:,i].add(label[:,i].eq(pred[:,j])).float()
    return t_score.sum(2).eq(num_nonz).sum() * 1./ pred.shape[1]









gpu = 1 # gpu id
batch_size = 250 # training batch size
lr = 0.001 # basic learning rate
lr_decay_startpoint = 250 #learning rate from which epoch
num_epochs = 400 # total training epochs
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

torch.set_num_threads(4)
manualSeed = torch.randint(1,10000,(1,))
print("Random seed " + str(manualSeed.item()))
torch.set_default_tensor_type(torch.FloatTensor)

train_size = 600000
valid_size = 100000
valid_data = torch.zeros(valid_size, input_size)
valid_label = torch.zeros(valid_size, num_nonz)
batch_data = torch.zeros(batch_size, input_size)
batch_label = torch.zeros(batch_size, num_nonz) # for MultiClassNLLCriterion LOSS
batch_zero_states = torch.zeros(batch_size, num_layers * rnn_size * 2) #init_states for lstm

AccM, AccL, Accs = 0, 0, 0


err = 0



model_all = "model_l_"+str(num_layers)+"t_"+str(num_unroll)+'_rnn_'+ str(rnn_size)
logger_file = model_all+str(dataset)+"_"+str(num_nonz)+'.log'
logger = open(logger_file, 'w')
#for k,v in pairs(opt) do logger:write(k .. ' ' .. v ..'\n') end
#logger:write('network have ' .. paras:size(1) .. ' parameters' .. '\n')
#logger:close()


mat_A = loadmat('matrix_corr_unit_20_100.mat')
batch_X = torch.Tensor(batch_size, 100)
batch_n = torch.Tensor(batch_size, num_nonz)

def gen_batch():
    bs = batch_size
    len = int(100 / num_nonz*num_nonz)
    print(len)
    perm = torch.randperm(100)[range(1,len)]
    print(bs*num_nonz/len)
    for i in range(1, int(bs*num_nonz/len)):
        perm = torch.cat((perm, torch.randperm(100)[range(1,len)]))
    batch_label.copy_(perm[range(1, bs*num_nonz)].reshape([bs, num_nonz]))
    batch_X.zero_()
    if dataset == 'uniform':
        batch_n.uniform_(-1,1)
        batch_n[batch_n.gt(0)] = 1
        batch_n[batch_n.le(0)] = -1
    for i in range(bs):
        for j in range(num_nonz):
            batch_X[i][batch_label[i][j]] = batch_n[i][j]
    batch_data.copy(batch_X * mat_A)

print("building validation set")
for i in range(1, valid_size, batch_size):
    gen_batch()
    valid_data[range(i,i+batch_size-1)][:].copy(batch_data)
    valid_label[range(i, i + batch_size - 1)][:].copy(batch_label)
print('done')

best_valid_accs = 0
base_epoch = lr_decay_startpoint
base_lr = lr
optimRate = {'learningRate' : 0.001, 'weigthDecay' : 0.001}

net = GetLstmNet(num_unroll, num_layers, rnn_size, output_size, input_size)
print(net)
# create a stochastic gradient descent optimizer
optimizer = optim.RMSprop(net.parameters(), lr=lr)
# create a loss function
LOSS = MultiClassNLLCriterion()


for epoch in range(1,num_epochs):
    #learing rate self - adjustment
    if(epoch > 250):
        optimRate['learningRate'] = base_lr / (1 + 0.06 * (epoch - base_epoch))
        if(epoch % 50 == 0): base_epoch = epoch; base_lr= base_lr * 0.25
    logger = open(logger_file, 'a')
    #train
    train_accs = 0
    train_accl = 0
    rain_accm = 0
    train_err = 0
    nbatch = 0
    start = time.time()
    for i in range(1,train_size,batch_size):
        gen_batch()
        net.train()
        optimizer.zero_grad()
        pred_prob = net(batch_data, batch_zero_states)
        err = LOSS(pred_prob, batch_label)
        print("loss = ", err.item())
        df_dpred = err.backward()
        optimizer.step()
        batch_accs = AccS(batch_label[:, range(1, num_nonz)], net.output[1].float())
        batch_accl = AccL(batch_label[:, range(1, num_nonz)], net.output[1].float())
        batch_accm = AccM(batch_label[:, range(1, num_nonz)], net.output[1].float())
        train_accs = train_accs + batch_accs
        train_accl = train_accl + batch_accl
        train_accm = train_accm + batch_accm
        train_err = train_err + err
        nbatch = nbatch + 1
        if nbatch % 512 == 1:
            print("%.4f %.4f %.4f err %.4f", batch_accs, batch_accl, batch_accm, err)
    print("Train [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f",epoch, time.time().real,\
        train_accs / nbatch, train_accl / nbatch, train_accm / nbatch, train_err / nbatch)
    logger.write("Train [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f\n", epoch, time.time().real,\
        train_accs / nbatch, train_accl / nbatch, train_accm / nbatch, train_err / nbatch)
    end = time.time()
    #eval
    nbatch = 0
    valid_accs = 0
    valid_accl = 0
    valid_accm = 0
    valid_err = 0
    for i in range(1,valid_size,batch_size):
        batch_data.copy(valid_data[range(i,i+batch_size-1),:])
        net.eval()
        pred_prob = net(batch_data,batch_zero_states).float()
        err = LOSS(pred_prob, batch_label)
        batch_accs = AccS(batch_label[:, range(1, num_nonz)], net.output[1].float())
        batch_accl = AccL(batch_label[:, range(1, num_nonz)], net.output[1].float())
        batch_accm = AccM(batch_label[:, range(1, num_nonz)], net.output[1].float())
        valid_accs = valid_accs + batch_accs
        valid_accl = valid_accl + batch_accl
        valid_accm = valid_accm + batch_accm
        valid_err = valid_err + err
        nbatch = nbatch + 1
    print("Valid [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f", epoch, time.time().real, \
          valid_accs / nbatch, valid_accl / nbatch, valid_accm / nbatch, valid_err / nbatch)
    logger.write("Valid [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f\n", epoch, time.time().real, \
                 valid_accs / nbatch, valid_accl / nbatch, valid_accm / nbatch, valid_err / nbatch)
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











