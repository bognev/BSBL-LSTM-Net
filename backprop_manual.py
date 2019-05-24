import numpy as np

x = 3
y = -4
sigy = 1/(1+np.exp(-y))
sigx = 1/(1+np.exp(-x))
xpy = x+y
xpysqr = xpy**2
den = sigx + xpysqr
invden = 1/den
num = x + sigy
f = num*den
print(f)

dnum = 1*invden
dinvden = 1*dnum
dx = 1*dnum
dsigy = 1*dnum
dy = dsigy*(np.exp(-y)/(1+np.exp(-y))**2)
dinvden = 1*dnum
dden = dinvden*(-1)/(den**2)
dsigx = 1*dden
dx += dsigx*(np.exp(-x)/(1+np.exp(-x))**2)
dxpysqr = 1*dden
dxpy = dxpysqr*2*xpy
dx += dxpy
dy += + dxpy


W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)