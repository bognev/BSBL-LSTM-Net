from cvxpy import *
import numpy as np
import cvxopt
from multiprocessing import Pool


# Problem data.
n = 100
m = 10
A = cvxopt.normal(n,m)
b = cvxopt.normal(n)
gamma = Parameter(nonneg=True)

# Construct the problem.
x = Variable(m)
# objective = Minimize(sum_squares(A*x - b) + gamma*norm(x, 1))
objective = Minimize(sum_squares(A*x - np.array(b).reshape(n,)) + gamma*norm(x, 1))
p = Problem(objective)

# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

gammas = np.logspace(-1, 2, num=100)
# Serial computation.
x_values = [get_x(value) for value in gammas]

# Parallel computation.
pool = Pool(processes = 4)
par_x = pool.map(get_x, gammas)

for v1,v2 in zip(x_values, par_x):
    if np.linalg.norm(v1 - v2) > 1e-5:
        print("error")