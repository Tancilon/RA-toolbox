import h5py
import numpy as np
import scipy.io

def run_dibra():
    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        data = f['workerlist_sim'][:].T
    # Parameters
    TopK = 10

    # Get the size of the data matrix
    rankernum = data.shape[0]      # Equivalent to size(data, 1)
    querynum = data.shape[1]       # Equivalent to size(data, 2)
    gallerynum = data.shape[2]      # Equivalent to size(data, 3)

    converged = np.zeros((querynum, rankernum))
    new_w = np.zeros((querynum, rankernum))
    w0 = np.ones((querynum, rankernum)) / rankernum

    L = np.zeros((querynum, gallerynum))
    for i in range(querynum):
        for j in range(rankernum):
            L[i, :] += (data[j, i, :] * w0[i, j])

    origin_ranklist = np.argsort(-L, axis=1)
    origin_rank = np.argsort(origin_ranklist, axis=1)


    for q in range(querynum):
        now_L = origin_ranklist[q, :]
        now_L_rank = origin_rank[q, :]
        i = 0
        allconverged = 0
        pre_w = w0[q, :]
        now_w = np.zeros(rankernum)

        while allconverged == 0:
            i += 1
            allconverged = 1
            for r in range(rankernum):
                if converged[q, r] == 0:
                    distance = 0
                    V_ranklist = np.argsort(-data[r, q, :])  # Sort for V_ranklist
                    V_ranklist = V_ranklist.reshape(1, gallerynum)
                    V_rank = np.argsort(V_ranklist, axis=1)

                    for j in range(TopK):
                        idx_V = V_ranklist[0, j]
                        distance += abs((j + 1) / TopK - (now_L_rank[idx_V] / gallerynum))

                    distance /= (TopK / 2)
                    now_w[r] = pre_w[r] + np.exp(-i * distance)
                    if (now_w[r] - pre_w[r]) > 0.001:
                        allconverged = 0
                    else:
                        converged[q, r] = 1
                else:
                    now_w[r] = pre_w[r]

            new_L = np.zeros(gallerynum)
            for r in range(rankernum):
                new_L += (data[r, q, :] * now_w[r])
                pre_w[r] = now_w[r]

            now_L = np.argsort(-new_L)
            now_L_rank = np.argsort(now_L)

        new_w[q, :] = now_w

    new_L = np.zeros((querynum, gallerynum))
    for j in range(querynum):
        for r in range(rankernum):
            new_L[j, :] += (data[r, j, :] * new_w[j, r])

    total_ranklist = np.argsort(-new_L, axis=1)
    result = np.argsort(total_ranklist, axis=1)

    # Save the result to a MAT file
    scipy.io.savemat('D:/LocalGit/RA-toolbox/py.mat', {'result': result})

run_dibra()
