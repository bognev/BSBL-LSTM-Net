import numpy as np
c = 3 * 10 ** 8
dt = 10 ** (-7)
Ts = 1.6000e-06
L = int(Ts / dt)
T = 400
def gen_mimo_samples(batch_size, SNR_dB, M, N, K, NOISE, H):
    x_r = np.array(
        [1000, 2000, 2500, 2500, 2000, 1000, 500, 500])  +11# + 500 * (np.random.rand(N) - 0.5))  # \
    y_r = np.array(
        [500, 500, 1000, 2000, 2500, 2500, 2000, 1500])  +11# + 500 * (np.random.rand(N) - 0.5))  # \
    # Position of transmitters
    x_t = np.array([0, 4000, 4000, 0, 1500, 0, 4000, 2000])  +11
    y_t = np.array([0, 0, 4000, 4000, 4000, 1500, 1500, 0])  +11
    x_r = x_r.reshape(8, 1)
    y_r = y_r.reshape(8, 1)
    x_t = x_t.reshape(8, 1)
    y_t = y_t.reshape(8, 1)
    # 1500,3000,500,2500,1000,1500,500,3000,\
    # 2500,3500,1000,3500,2000,4000,3000,3000]+500*(np.random.rand(N)-0.5))
    # 3500,3500,500,4000,4000,2500,3000,500,\
    # 3500,3000,2000,1000,2000,500,4000,1500]+500*(np.random.rand(N)-0.5))

    s = np.zeros([M, L]) + 1j * np.zeros([M, L])
    for m in range(M):
        s[m] = np.exp(1j * 2 * np.pi * (m) * np.arange(L) / M) / np.sqrt(L);  # np.sqrt(0.5)*(np.random.randn(1,L)+1j*np.random.randn(1,L))/np.sqrt(L);#
    Ls = 10
    Le = Ls + 6*250
    dx = 250
    dy = dx
    dy = dx
    x_grid = np.arange(Ls, Le, dx)
    y_grid = np.arange(Ls, Le, dy)
    size_grid_x = len(x_grid)
    size_grid_y = len(y_grid)
    grid_all_points = [[i, j] for i in x_grid for j in y_grid]
    grid_all_points_a = np.array(grid_all_points)

    const_sqrt_200000000000 = np.sqrt(200000000000)
    grid_all_points_bs = np.repeat(grid_all_points_a[np.newaxis, ...], batch_size, axis=0)
    x_r_bs = np.repeat(x_r[np.newaxis, ...], batch_size, axis=0)
    y_r_bs = np.repeat(y_r[np.newaxis, ...], batch_size, axis=0)
    x_t_bs = np.repeat(x_t[np.newaxis, ...], batch_size, axis=0)
    y_t_bs = np.repeat(y_t[np.newaxis, ...], batch_size, axis=0)
    rk = np.zeros([batch_size, K, M, N, 1]);
    tk = np.zeros([batch_size, K, M, N, 1]);
    tau = np.zeros([batch_size, K, M, N, 1]);
    r = np.zeros([batch_size, size_grid_x * size_grid_y])
    DB = 10. ** (0.1 * SNR_dB)
    # NOISE = 1  # on/off noise
    # H = 1  # on/off êîýôôèöèåíòû îòðàæåíèÿ
    if NOISE == 0:
        x = np.zeros([batch_size, N, T]) + 1j * np.zeros([batch_size, N, T])
    else:
        x = (np.random.randn(batch_size, N, T) + 1j * np.random.randn(batch_size, N, T)) / np.sqrt(2)
    if H == 0:
        h = np.ones([batch_size, K, M, N])
    else:
        h = (np.random.randn(batch_size, K, M, N) + 1j * np.random.randn(batch_size, K, M, N)) / np.sqrt(2)


    k_random_grid_points = np.array([])
    # Position of targets
    a=np.random.randint(0,35,K)
    x_k = grid_all_points_a[a][:,0]#np.random.randint(Ls,Le,(batch_size,K,1))+np.random.rand(batch_size,K,1)
    y_k = grid_all_points_a[a][:,1]#np.random.randint(Ls,Le,(batch_size,K,1))+np.random.rand(batch_size,K,1)
    x_k[0] = grid_all_points[10][0]
    y_k[0] = grid_all_points[10][1]
    x_k[1] = grid_all_points[20][0]
    y_k[1] = grid_all_points[20][1]
    k_random_grid_points_i = np.zeros([batch_size,K])
    # k_random_grid_points = np.array([])
    for k in range(K):
        calc_dist = np.sqrt((grid_all_points_bs[:,range(size_grid_x * size_grid_y), 0] - x_k[k]) ** 2 \
                            + (grid_all_points_bs[:,range(size_grid_x * size_grid_y), 1] - y_k[k]) ** 2)
        # grid_all_points_a[calc_dist.argmin()]
        k_random_grid_points_i[:,k] = calc_dist.argmin(axis=1)

    # Time delays
    for k in range(K):
        for m in range(M):
            for n in range(N):
                print(x_k[k])
                print(y_k[k])
                print(x_t_bs[:, m])
                print(y_t_bs[:, m])
                tk[:, k, m, n] = np.sqrt((x_k[k] - x_t_bs[:, m]) ** 2 + (y_k[k] - y_t_bs[:, m]) ** 2)
                rk[:, k, m, n] = np.sqrt((x_k[k] - x_r_bs[:, n]) ** 2 + (y_k[k] - y_r_bs[:, n]) ** 2)
                tau[:, k, m, n] = (tk[:, k, m, n] + rk[:, k, m, n]) / c
    print(tau)
    r_glob = np.zeros([batch_size, size_grid_x * size_grid_y * M * N]) + 1j * np.zeros([batch_size, size_grid_x * size_grid_y * M * N])
    for m in range(M):
        for n in range(N):
            for k in range(K):
                r_glob[:, k_random_grid_points_i[:,k].astype(int)] = DB[k] * h[:, k, m, n] * \
                                                  np.sqrt(200000000000) * (1 / tk[:, k, m, n]) * (1 / rk[:, k, m, n])
            k_random_grid_points = np.append(k_random_grid_points,k_random_grid_points_i)
            k_random_grid_points_i = k_random_grid_points_i + size_grid_x * size_grid_y

    # for m in range(M):
    #     for n in range(N):
    #         k_random_grid_points = np.append(k_random_grid_points,k_random_grid_points[-1] + size_grid_x * size_grid_y)


    # np.put(r, k_random_grid_points_i.astype(int), 1)

    l = np.floor(tau / dt).astype(int)
    for bs in range(batch_size):
        for k in range(K):
            for n in range(N):
                for m in range(M):
                    x[:, n, l[bs,k,m,n].item(): l[bs,k,m,n].item() + L] = x[:, n, l[bs,k,m,n].item(): l[bs,k,m,n].item() + L] + DB[k] * s[m, :] * h[bs, k, m, n] * \
                                             const_sqrt_200000000000 * (1 / tk[bs, k, m, n]) * (1 / rk[bs, k, m, n])

    x_flat = x[:, 0, :];
    for n in range(1, N):
        x_flat = np.concatenate([x_flat, x[:, n, :]], axis=1)

    return x_flat, k_random_grid_points_i, r_glob
