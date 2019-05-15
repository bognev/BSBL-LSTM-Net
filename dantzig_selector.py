# The 1-norm regularized least-squares example of section 8.7 (Exploiting
# structure).

from cvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed
from cvxopt import blas, lapack, solvers
import math
import matplotlib.pyplot as plt
from l1regls import l1regls

import torch
from mat4py import loadmat
import cvxpy as cp
import numpy as np
import cvxopt
from multiprocessing import Pool
plt.close('all')
dataset = 'uniform' # type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
num_nonz = 3 # number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
input_size = 20 # dimension of observation vector y
output_size = 300 # dimension of sparse vector x
gpu = 1 # gpu id
batch_size = 1#250 # training batch size
num_groups = 4
max_grad_norm = 5.0
clip_gradient = 4.0
batch_data = torch.zeros(batch_size, input_size)
batch_label = torch.zeros(batch_size, num_nonz) # for MultiClassNLLCriterion LOSS
lmbd = 100
F_m = int(num_groups*output_size/4)
print(F_m)
# torch.manual_seed(10)
def gen_groups(F, num_groups, output_size, input_size, num_nonz):
    # np.random.seed(100)
    mat_A = np.random.randn(num_groups,output_size,input_size)
    # A = np.random.randn(num_groups*output_size, num_groups*input_size)
    A = np.zeros((num_groups * output_size, num_groups * input_size))
    perm = np.random.permutation(range(input_size))
    label = perm[range(num_nonz)]
    for i in range(1,num_groups):
        label = np.concatenate((label, perm[range(num_nonz)]+i*input_size))
    x=np.zeros((num_groups*input_size))
    x[label]=1*np.ones(num_groups*num_nonz)#*np.random.randn(num_groups*num_nonz)
    for i in range(num_groups):
        A[output_size*i:output_size*(i+1),i*input_size:(i+1)*input_size] = mat_A[i]
    n = np.random.randn(int(num_groups * output_size))
    y = (A@x + n)
    y_f = F@y
    return y,A,x,y_f, label

F = np.random.rand(int(F_m),output_size*num_groups)
y, A, x, y_f, label = gen_groups(F, num_groups, output_size, input_size, num_nonz)

#meazurement group lasso
lambdas = cp.Parameter(nonneg=True)
lambdas.value = lmbd
# Define problem
x_v = cp.Variable(input_size*num_groups)
p = cp.Variable(1)
q = cp.Variable(1)
objective = 0.5*p**2+lambdas*q
a = []
for ii in range(input_size):
    a.append(cp.norm(x_v[ii:input_size*num_groups:input_size],2))
constr = [cp.norm(y_f-F@A@x_v,2) <= p, sum(a) <= q]
prob = cp.Problem(cp.Minimize(objective), constr)
prob.solve()

#y group lasso
lambdas = cp.Parameter(nonneg=True)
lambdas.value = lmbd
# Define problem
x_y = cp.Variable(input_size*num_groups)
p = cp.Variable(1)
q = cp.Variable(1)
objective = 0.5*p**2+lambdas*q

a = []
for ii in range(input_size):
    a.append(cp.norm(x_y[ii:input_size*num_groups:input_size],2))

constr = [cp.norm(y-A@x_y,2) <= p, sum(a) <= q]
prob = cp.Problem(cp.Minimize(objective), constr)
prob.solve()

#meazurement group dantzig selector as lasso
lambdas_f = cp.Parameter(nonneg=True)
lambdas_f.value = lmbd
# Define problem
x_v_f = cp.Variable(input_size*num_groups)
p_f = cp.Variable(1)
q_f = cp.Variable(1)
G = np.zeros((input_size,num_groups,F_m))
a_f = []
d_f = []
for ii in range(input_size):
    G[ii,:,:] = (F@A[:,ii:input_size*num_groups:input_size]).transpose()
    a_f.append(cp.norm(x_v_f[ii:input_size*num_groups:input_size],2))
    d_f.append(G[ii]@(y_f-F@A@x_v_f))
d_f_max = cp.norm(cp.abs(d_f[0]),2)
for ii in range(1,input_size):
    d_f_max = cp.maximum(cp.norm(cp.abs(d_f[ii]),2), d_f_max)
objective_f = 0.5*p_f**2+lambdas_f*q_f
constr_f = [d_f_max <= p_f, cp.sum(a_f) <= q_f]
prob_f = cp.Problem(cp.Minimize(objective_f), constr_f)
prob_f.solve()

#meazurement group dantzig selector (4) from article
lambdas_f = cp.Parameter(nonneg=True)
lambdas_f.value = lmbd
# Define problem
x_v_e = cp.Variable(input_size*num_groups)
G = np.zeros((input_size,num_groups,F_m))
a_f = []
d_f = []

for ii in range(input_size):
    G[ii,:,:] = (F@A[:,ii:input_size*num_groups:input_size]).transpose()
    a_f.append(cp.norm(x_v_e[ii:input_size*num_groups:input_size],2))
    d_f.append(G[ii]@(y_f-F@A@x_v_e))
d_f_max = cp.norm((d_f[0]),2)
for ii in range(1,input_size):
    d_f_max = cp.maximum(cp.norm(d_f[ii],1), d_f_max)
objective_f = cp.sum(a_f)
constr_f = [d_f_max <= lambdas_f]
prob_f = cp.Problem(cp.Minimize(objective_f), constr_f)
prob_f.solve()

#y group dantzig selector (4) from article
lambdas_f = cp.Parameter(nonneg=True)
lambdas_f.value = lmbd
# Define problem
x_v_y = cp.Variable(input_size*num_groups)
G = np.zeros((input_size,num_groups,output_size*num_groups))
a_f = []
d_f = []
for ii in range(input_size):
    G[ii,:,:] = (A[:,ii:input_size*num_groups:input_size]).transpose()
    a_f.append(cp.norm(x_v_y[ii:input_size*num_groups:input_size],2))
    d_f.append(G[ii]@(y-A@x_v_y))
d_f_max = cp.norm(cp.abs(d_f[0]),2)
for ii in range(1,input_size):
    d_f_max = cp.maximum(cp.norm(cp.abs(d_f[ii]),2), d_f_max)
objective_f = cp.sum(a_f)
constr_f = [d_f_max <= lambdas_f]
prob_f = cp.Problem(cp.Minimize(objective_f), constr_f)
prob_f.solve()

#y elastic group lasso
lambdas = cp.Parameter(nonneg=True)
mu = cp.Parameter(nonneg=True)
lambdas.value = lmbd
mu.value = 0.5
# Define problem
x_el = cp.Variable(input_size*num_groups)
p = cp.Variable(1)
q = cp.Variable(1)
r = cp.Variable(1)
objective = 0.5*p**2+lambdas*q+mu*r

a, b = [], []
for ii in range(input_size):
    a.append(cp.norm(x_el[ii:input_size*num_groups:input_size],2))
    # b.append()

constr = [cp.norm(y-A@x_el,2) <= p, cp.sum(a) <= q, cp.norm(x_el,2)**2 <= r]
prob = cp.Problem(cp.Minimize(objective), constr)
prob.solve()

#y sparse group lasso
lambdas = cp.Parameter(nonneg=True)
# alpha = cp.Parameter(nonneg=True)
lambdas.value = lmbd
alpha = 0.3
# Define problem
x_sgl = cp.Variable(input_size*num_groups)
p = cp.Variable(1)
q = cp.Variable(1)
# r = cp.Variable(1)
objective = 0.5*p**2+lambdas*q

a, b = [], []
for ii in range(input_size):
    a.append((1-alpha)*cp.norm(x_sgl[ii:input_size*num_groups:input_size],2)\
             +alpha*cp.norm(x_sgl[ii:input_size*num_groups:input_size],1))
    # b.append()

constr = [cp.norm(y-A@x_sgl,2) <= p, cp.sum(a) <= q]
prob = cp.Problem(cp.Minimize(objective), constr)
prob.solve()

#y group lasso
lambdas = cp.Parameter(nonneg=True)
lambdas.value = lmbd
# Define problem
x_y2 = cp.Variable(input_size*num_groups)
xy = np.zeros(input_size*num_groups)
xy[label[0]:input_size*num_groups:input_size] = 0.1
xy[label[1]:input_size*num_groups:input_size] = 0.25
xy[label[2]:input_size*num_groups:input_size] = 0.5
p = cp.Variable(1)
q = cp.Variable(1)
objective = 0.5*p**2+lambdas*q

a = []
for ii in range(input_size):
    a.append(cp.norm(x_y[ii:input_size*num_groups:input_size],2))

constr = [cp.norm(y-A@(x_y2+xy),2) <= p, sum(a) <= q]
prob = cp.Problem(cp.Minimize(objective), constr)
prob.solve()

plt.figure(1)
plt.subplot(911)
plt.plot(x, lw=2)
plt.grid(True)
plt.subplot(912)
plt.plot(x_v.value, lw=2)
plt.grid(True)
plt.subplot(913)
plt.plot(x_y.value, lw=2)
plt.grid(True)
plt.subplot(914)
plt.plot(x_v_f.value, lw=2)
plt.grid(True)
plt.subplot(915)
plt.plot(x_v_e.value, lw=2)
plt.grid(True)
plt.subplot(916)
plt.plot(x_v_y.value, lw=2)
plt.grid(True)
plt.subplot(917)
plt.plot(x_el.value, lw=2)
plt.grid(True)
plt.subplot(918)
plt.plot(x_sgl.value, lw=2)
plt.grid(True)

plt.tight_layout()
#
plt.figure(2)
plt.subplot(111)
plt.plot(x_y2.value, lw=2)
plt.grid(True)
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
