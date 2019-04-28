from cvxpy import *
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt

# Generate problem data
sp.random.seed(1)
n = 10
m = 1000
A = np.random.rand(m, n)
x_true = np.multiply((np.random.rand(n) > 0.8).astype(float),
                     np.random.randn(n)) / np.sqrt(n)
#b = A.dot(x_true) + 0.5*np.random.randn(m)
b = A.dot(x_true)
gammas = np.linspace(1, 10, 11)

# Define problem
x = Variable(n)
gamma = Parameter(nonneg=True)
objective = 0.5*sum_squares(A*x - b) + gamma*norm1(x)
prob = Problem(Minimize(objective))

# Solve problem for different values of gamma parameter
for gamma_val in gammas:
    gamma.value = gamma_val
    prob.solve(solver=OSQP, warm_start=True)


plt.figure(1)
plt.subplot(211)
plt.plot(x_true, lw=2)
plt.grid(True)
plt.subplot(212)
plt.plot(x.value, lw=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# from cvxpy import *
# import numpy as np
# from multiprocessing import Pool
#
# # Problem data.
# m = 100
# n = 75
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m, 1)
#
# def prox(args):
#     f, v = args
#     f += (rho/2)*sum_squares(x - np.array(v).reshape(n,))
#     Problem(Minimize(f)).solve()
#     return x.value
#
# # Setup problem.
# rho = 1.0
# gamma = 1
# x = Variable(n)
# funcs = [sum_squares(A*x - np.array(b).reshape(m,)),
#          gamma*norm(x, 1)]
# ui = [np.zeros((n, 1)) for func in funcs]
# xbar = np.zeros((n, 1))
# pool = Pool(4)
# # ADMM loop.
# for i in range(50):
#     prox_args = [xbar - u for u in ui]
#     xi = pool.map(prox, zip(funcs, prox_args))
#     xbar = sum(xi)/len(xi)
#     ui = [u + x_ - xbar for x_, u in zip(xi, ui)]
#
# # Compare ADMM with standard solver.
# prob = Problem(Minimize(sum(funcs)))
# result = prob.solve()
# print("ADMM best", (sum_squares(np.dot(A, xbar) - b) + gamma*norm(xbar, 1)).value)
# print("ECOS best", result)