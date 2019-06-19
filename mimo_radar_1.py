import numpy as np
import cvxpy as cp
import torch
import time
import matplotlib.pyplot as plt
import time

c=3*10**8
dt=10**(-7)
Ts=1.6000e-06
L=int(Ts/dt)
T=400
SNR_dB=15
DB=10.**(0.1*SNR_dB)

N=8 #the number of receivers
M=3 #the number of transmitters

K=2 #the number of targets
# np.random.seed(15)
#Position of receivers
x_r=np.array([1000,2000,2500,2500,2000,1000,500,500]) + 128#+500*(np.random.rand(N)-0.5))#\
    # 1500,3000,500,2500,1000,1500,500,3000,\
    # 2500,3500,1000,3500,2000,4000,3000,3000]+500*(np.random.rand(N)-0.5))
y_r=np.array([500,500,1000,2000,2500,2500,2000,1500]) + 128#+500*(np.random.rand(N)-0.5))#\
     # 3500,3500,500,4000,4000,2500,3000,500,\
     # 3500,3000,2000,1000,2000,500,4000,1500]+500*(np.random.rand(N)-0.5))

#Position of transmitters
x_t=np.array([0,4000,4000,0,1500,0,4000,2000]) + 128
y_t=np.array([0,0,4000,4000,4000,1500,1500,0]) + 128

NOISE = 1  #on/off noise
H = 1      #on/off êîýôôèöèåíòû îòðàæåíèÿ
rk = np.zeros([K,M,N]);
tk = np.zeros([K,M,N]);
tau = np.zeros([K,M,N]);
if H == 0:
    h=np.ones([K,M,N])
else:
    h=(np.random.randn(K,M,N)+1j*np.random.randn(K,M,N))/np.sqrt(2)

s=np.zeros([M,L])+1j*np.zeros([M,L])
for m in range(M):
    s[m]=np.exp(1j*2*np.pi*(m)*np.arange(L)/M)/np.sqrt(L);#sqrt(0.5)*(randn(1,L)+1i*randn(1,L))/sqrt(L);
Ls = 0
Le = Ls + 4000
dx = 225
dy = dx
x_grid = np.arange(Ls,Le,dx)
y_grid = np.arange(Ls,Le,dy)
size_grid_x  = len(x_grid)
size_grid_y  = len(y_grid)
grid_all_points = [[i, j] for i in x_grid for j in y_grid]
r=np.zeros(size_grid_x*size_grid_y)
k_random_grid_points = np.random.permutation(size_grid_x*size_grid_y)[range(K)]
#Position of targets
x_k=np.zeros([K])
y_k=np.zeros([K])
a=[10,20]
for kk in range(K):
    # x_k[kk]=grid_all_points[k_random_grid_points.item(kk)][0]
    # y_k[kk]=grid_all_points[k_random_grid_points.item(kk)][1]
    x_k[kk] = grid_all_points[a[kk]][0]
    y_k[kk] = grid_all_points[a[kk]][1]
r[k_random_grid_points] = 1

#Time delays
for k in range(K):
    for m in range(M):
        for n in range(N):
            tk[k,m,n]=np.sqrt((x_k[k]-x_t[m])**2+(y_k[k]-y_t[m])**2)
            rk[k,m,n]=np.sqrt((x_k[k]-x_r[n])**2+(y_k[k]-y_r[n])**2)
            tau[k,m,n]=(tk[k,m,n]+rk[k,m,n])/c

r_glob = np.zeros([size_grid_x*size_grid_y*M*N]) + 1j*np.zeros([size_grid_x*size_grid_y*M*N])
for m in range(M):
    for n in range(N):
        for k in range(K):
            r_glob[k_random_grid_points[k]] = DB*h[k,m,n]*\
                                              np.sqrt(200000000000)*(1/tk[k,m,n])*(1/rk[k,m,n])
        k_random_grid_points = k_random_grid_points + size_grid_x * size_grid_y

tau_grid_t = np.zeros([size_grid_x,size_grid_y,M])
for m in np.arange(M):
    for xx in np.arange(size_grid_x):
        for yy in np.arange(size_grid_y):
            tau_grid_t[xx,yy,m] = np.sqrt((x_grid[xx]-x_t[m])**2+(y_grid[yy]-y_t[m])**2)

tau_grid_r = np.zeros([size_grid_x,size_grid_y,N])
for n in np.arange(N):
    for xx in np.arange(size_grid_x):
        for yy in np.arange(size_grid_y):
            tau_grid_r[xx,yy,n] = np.sqrt((x_grid[xx]-x_r[n])**2+(y_grid[yy]-y_r[n])**2)

tau_grid_c = np.zeros([size_grid_x,size_grid_y,N,M])
for n in np.arange(N):
    for m in np.arange(M):
        tau_grid_c[:,:,n,m] = (tau_grid_r[:,:,n]+tau_grid_t[:,:,m])/c

if NOISE == 0:
    x=np.zeros([N,T]) + 1j*np.zeros([N,T])
else:
    x=np.random.randn(N,T)+1j*np.random.randn(N,T)/np.sqrt(2)

for k in range(K):
    for m in range(M):
        for n in range(N):
            l=np.floor(tau[k,m,n]/dt)
            l=l.astype(int)
            x[n,range(l,l+L)]= x[n,range(l,l+L)] + DB*s[m,:]*h[k,m,n]*\
                                                   np.sqrt(200000000000)*(1/tk[k,m,n])*(1/rk[k,m,n])

x_flat = x[0,:].transpose();
for n in range(1,N):
    x_flat = np.concatenate([x_flat,x[n,:].transpose()],axis=0)


dictionary=(np.zeros([M,N,T,size_grid_x,size_grid_y])+ \
    1j*np.zeros([M,N,T,size_grid_x,size_grid_y]))/np.sqrt(2);

ll = [];

for xx in np.arange(size_grid_x):
    for yy in np.arange(size_grid_y):
        for m in np.arange(M):
            for n in np.arange(N):
                l=np.floor(tau_grid_c[xx,yy,n,m]/dt)
                dictionary[m,n,np.arange(l,l+L,dtype = np.integer),xx,yy] = s[m,:].transpose()*\
                                                                                np.sqrt(200000000000)*\
                                                                                (1/tk[k,m,n])*(1/rk[k,m,n])

D_flat = np.zeros([N*T,N*size_grid_x*size_grid_y*M]) + 1j*np.zeros([N*T,N*size_grid_x*size_grid_y*M])
i=0
for m in range(M):
    for n in range(N):
        for xx in range(size_grid_x):
            for yy in range(size_grid_y):
                D_flat[range(n*T,(n+1)*T),i]= np.squeeze(dictionary[m,n,range(T),xx,yy])
                i += 1
from gen_mimo_samples import gen_mimo_samples
start_time = time.time()
y, label, rr_glob  = gen_mimo_samples(256, 20*np.ones(3), M, N, K, 0, H, 1)
print("--- %s seconds ---" % (time.time() - start_time))
print(label[100])

#group lasso
lambdas = cp.Parameter(nonneg=True)
lambdas.value = 1
# Define problem
x = cp.Variable(size_grid_x*size_grid_y*M*N,complex=True)
p = cp.Variable(1)
q = cp.Variable(1)
objective = 0.5*p**2+lambdas*q

a = []
for ii in range(size_grid_x*size_grid_y):
    a.append(cp.norm(x[range(ii,size_grid_x*size_grid_y*M*N,size_grid_x*size_grid_y)],2))

constr = [cp.norm(y[4,:]-D_flat@x,2) <= p, sum(a) <= q]
prob = cp.Problem(cp.Minimize(objective), constr)
prob.solve()

lambdas1 = cp.Parameter(nonneg=True)
lambdas1.value = 1

x1 = cp.Variable(size_grid_x*size_grid_y*M*N,complex=True)
p1 = cp.Variable(1)
q1 = cp.Variable(1)
objective1 = 0.5*p1**2+lambdas1*q1

a1 = []
for ii in range(size_grid_x*size_grid_y):
    a1.append(cp.norm(x1[range(ii,size_grid_x*size_grid_y*M*N,size_grid_x*size_grid_y)],2))

constr1 = [cp.norm(y[1,:]-D_flat@x1,2) <= p1, sum(a1) <= q1]
prob1 = cp.Problem(cp.Minimize(objective1), constr1)
prob1.solve()

plt.figure(2)
plt.subplot(311)
plt.plot(np.abs(rr_glob[4,:].transpose()), lw=2)
plt.grid(True)
plt.subplot(312)
plt.plot(np.abs(x.value), lw=2)
plt.grid(True)
plt.subplot(313)
plt.plot(np.real(y[4,:]), lw=2)
plt.grid(True)
# plt.show()

plt.figure(3)
plt.subplot(311)
plt.plot(np.abs(rr_glob[1,:].transpose()), lw=2)
plt.grid(True)
plt.subplot(312)
plt.plot(np.abs(x1.value), lw=2)
plt.grid(True)
plt.subplot(313)
plt.plot(np.real(y[1,:]), lw=2)
plt.grid(True)
plt.show()

print("done")
