def gen_mimo_samples(SNR_dB, M, N, K, NOISE, H):
    import numpy as np

    c = 3 * 10 ** 8
    dt = 10 ** (-7)
    Ts = 1.6000e-06
    L = int(Ts / dt)
    T = 400
    DB = 10. ** (0.1 * SNR_dB)

    # N = 8  # the number of receivers
    # M = 1  # the number of transmitters

    # K = 1  # the number of targets
    # np.random.seed(15)
    # Position of receivers
    x_r = np.array([1000, 2000, 2500, 2500, 2000, 1000, 500, 500])# + 500 * (np.random.rand(N) - 0.5))  # \
    # 1500,3000,500,2500,1000,1500,500,3000,\
    # 2500,3500,1000,3500,2000,4000,3000,3000]+500*(np.random.rand(N)-0.5))
    y_r = np.array([500, 500, 1000, 2000, 2500, 2500, 2000, 1500])# + 500 * (np.random.rand(N) - 0.5))  # \
    # 3500,3500,500,4000,4000,2500,3000,500,\
    # 3500,3000,2000,1000,2000,500,4000,1500]+500*(np.random.rand(N)-0.5))

    # Position of transmitters
    x_t = np.array([0, 4000, 4000, 0, 1500, 0, 4000, 2000])
    y_t = np.array([0, 0, 4000, 4000, 4000, 1500, 1500, 0])

    # NOISE = 1  # on/off noise
    # H = 1  # on/off êîýôôèöèåíòû îòðàæåíèÿ
    rk = np.zeros([K, M, N]);
    tk = np.zeros([K, M, N]);
    tau = np.zeros([K, M, N]);
    if H == 0:
        h = np.ones([K, M, N])
    else:
        h = (np.random.randn(K, M, N) + 1j * np.random.randn(K, M, N)) / np.sqrt(2)

    s = np.zeros([M, L]) + 1j * np.zeros([M, L])
    for m in range(M):
        s[m] = np.exp(1j * 2 * np.pi * (m) * np.arange(L) / M) / np.sqrt(L);
        # sqrt(0.5)*(randn(1,L)+1i*randn(1,L))/sqrt(L);
    Ls = 875
    Le = Ls + 125 * 6
    dx = 125
    dy = dx
    x_grid = np.arange(Ls, Le, dx)
    y_grid = np.arange(Ls, Le, dy)
    size_grid_x = len(x_grid)
    size_grid_y = len(y_grid)
    grid_all_points = [[i, j] for i in x_grid for j in y_grid]
    r = np.zeros(size_grid_x * size_grid_y * M * N)
    k_random_grid_points_i = np.random.permutation(size_grid_x * size_grid_y)[range(K)]
    k_random_grid_points = np.array([])
    # Position of targets
    x_k = np.zeros([K])
    y_k = np.zeros([K])
    for kk in range(K):
        x_k[kk] = grid_all_points[k_random_grid_points_i.item(kk)][0]
        y_k[kk] = grid_all_points[k_random_grid_points_i.item(kk)][1]


    # Time delays
    for k in range(K):
        for m in range(M):
            for n in range(N):
                tk[k, m, n] = np.sqrt((x_k[k] - x_t[m]) ** 2 + (y_k[k] - y_t[m]) ** 2)
                rk[k, m, n] = np.sqrt((x_k[k] - x_r[n]) ** 2 + (y_k[k] - y_r[n]) ** 2)
                tau[k, m, n] = (tk[k, m, n] + rk[k, m, n]) / c

    r_glob = np.zeros([size_grid_x * size_grid_y * M * N]) + 1j * np.zeros([size_grid_x * size_grid_y * M * N])
    for m in range(M):
        for n in range(N):
            for k in range(K):
                r_glob[k_random_grid_points_i[k]] = DB * h[k, m, n] * \
                                                  np.sqrt(200000000000) * (1 / tk[k, m, n]) * (1 / rk[k, m, n])
            k_random_grid_points = np.append(k_random_grid_points,k_random_grid_points_i)
            k_random_grid_points_i = k_random_grid_points_i + size_grid_x * size_grid_y

    # for m in range(M):
    #     for n in range(N):
    #         k_random_grid_points = np.append(k_random_grid_points,k_random_grid_points[-1] + size_grid_x * size_grid_y)

    r[k_random_grid_points.astype(int)] = 1
    if NOISE == 0:
        x = np.zeros([N, T]) + 1j * np.zeros([N, T])
    else:
        x = np.random.randn(N, T) + 1j * np.random.randn(N, T) / np.sqrt(2)

    for k in range(K):
        for m in range(M):
            for n in range(N):
                l = np.floor(tau[k, m, n] / dt)
                l = l.astype(int)
                x[n, range(l, l + L)] = x[n, range(l, l + L)] + DB * s[m, :] * h[k, m, n] * \
                                        np.sqrt(200000000000) * (1 / tk[k, m, n]) * (1 / rk[k, m, n])

    x_flat = x[0, :].transpose();
    for n in range(1, N):
        x_flat = np.concatenate([x_flat, x[n, :].transpose()], axis=0)

    return x_flat, r, r_glob, k_random_grid_points
