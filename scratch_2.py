# The 1-norm regularized least-squares example of section 8.7 (Exploiting
# structure).

from cvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed
from cvxopt import blas, lapack, solvers
import math
import matplotlib.pyplot as plt
from l1regls import l1regls



m, n = 100, 10
setseed()
A = normal(m, n)
x1 = matrix([0,0,0,1,0,0,0,1,0,0])
y = A*x1
x = l1regls(A, y)


plt.plot(x)
plt.ylabel('x')
plt.show()