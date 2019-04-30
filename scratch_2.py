# The 1-norm regularized least-squares example of section 8.7 (Exploiting
# structure).

from cvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed
from cvxopt import blas, lapack, solvers
import math
import matplotlib.pyplot as plt
from l1regls import l1regls

import torch
from mat4py import loadmat
from cvxpy import *
import numpy as np
import cvxopt
from multiprocessing import Pool
plt.close('all')
dataset = 'uniform' # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
num_nonz = 3 # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
input_size = 20 # dimension of observation vector y
output_size = 100 # dimension of sparse vector x
gpu = 1 # gpu id
batch_size = 1#250 # training batch size
lr = 0.001 # basic learning rate
lr_decay_startpoint = 250 #learning rate from which epoch
num_epochs = 2#00 # total training epochs
max_grad_norm = 5.0
clip_gradient = 4.0
batch_data = torch.zeros(batch_size, input_size)
batch_label = torch.zeros(batch_size, num_nonz) # for MultiClassNLLCriterion LOSS
# torch.manual_seed(10)
def gen_batch(batch_size, num_nonz):
    # mat_A = loadmat('matrix_corr_unit_20_100.mat')
    # mat_A = torch.Tensor(mat_A['A']).t()
    mat_A = torch.rand(output_size,input_size)
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
        batch_n.uniform_(-0.4,0.4)
        batch_n[batch_n.gt(0)] = batch_n[batch_n.gt(0)] + 0.1
        batch_n[batch_n.le(0)] = batch_n[batch_n.le(0)] - 0.1
    #
    #print(batch_X.shape)
    for i in range(bs):
        for j in range(num_nonz):
            batch_X[i][batch_label[i][j]] = batch_n[i][j]
    batch_data.copy_(batch_X@mat_A)
    print(batch_X.shape)
    print(mat_A.shape)
    return batch_label, batch_data, batch_X, mat_A


batch_label, batch_data, batch_X, mat_A = gen_batch(batch_size, num_nonz)

mat_A_np = mat_A.numpy()
batch_data_np = batch_data.numpy().reshape(input_size)
batch_X_np = batch_X.numpy()

print(mat_A_np.shape)
print(batch_data_np.shape)
print(batch_X_np.shape)

# mat_A_np = np.random.rand(input_size, output_size)
# batch_X_np = np.multiply((np.random.rand(output_size) > 0.8).astype(float),
#                      np.random.randn(output_size)) / np.sqrt(output_size)
# batch_data_np = mat_A_np.dot(batch_X_np)

# batch_data_np = mat_A_np.dot(batch_X_np).reshape(input_size,)
gammas = np.linspace(0.01, 0.1, 10)
# Define problem
batch_X_cvx = Variable(output_size)
gamma = Parameter(nonneg=True)
objective = 0.5*norm(batch_data_np-batch_X_cvx@mat_A_np,2)**2 + gamma*norm(batch_X_cvx,1)
# constr = [sum(batch_X_cvx) == 0, norm(batch_X_cvx,"inf") <= 1]
prob = Problem(Minimize(objective))#, constr)
i=0
# for gamma_val in gammas:
i=i+1
gamma.value = gammas[0]#gamma_val
prob.solve()
plt.figure(i)
plt.subplot(211)
plt.plot(batch_X_np[0], lw=2)
plt.grid(True)
plt.subplot(212)
plt.plot(batch_X_cvx.value, lw=2)
plt.grid(True)
plt.tight_layout()

plt.show()



# plt.figure(1)
# plt.subplot(211)
# plt.plot(batch_X_np[0], lw=2)
# plt.grid(True)
# plt.subplot(212)
# plt.plot(batch_X_cvx.value, lw=2)
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()
