import h5py
import numpy as np
from scipy.io import savemat


def run_dibra():
    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        data = f['workerlist_sim'][:].T
    # Parameters
    topk = 10

    # Get the size of the data matrix
    rankernum = data.shape[0]  # Equivalent to size(data, 1)
    querynum = data.shape[1]  # Equivalent to size(data, 2)
    gallerynum = data.shape[2]  # Equivalent to size(data, 3)

    converged = np.zeros((querynum, rankernum))
    new_w = np.zeros((querynum, rankernum))
    w0 = np.ones((querynum, rankernum)) / rankernum

    l_list = np.zeros((querynum, gallerynum))
    for i in range(querynum):
        for j in range(rankernum):
            l_list[i, :] += (data[j, i, :] * w0[i, j])

    origin_ranklist = np.argsort(-l_list, axis=1)
    origin_rank = np.argsort(origin_ranklist, axis=1)

    for q in range(querynum):
        now_l_rank = origin_rank[q, :]
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
                    v_ranklist = np.argsort(-data[r, q, :])  # Sort for v_ranklist
                    v_ranklist = v_ranklist.reshape(1, gallerynum)

                    for j in range(topk):
                        idx_v = v_ranklist[0, j]
                        distance += abs((j + 1) / topk - (now_l_rank[idx_v] / gallerynum))

                    distance /= (topk / 2)
                    now_w[r] = pre_w[r] + np.exp(-i * distance)
                    if (now_w[r] - pre_w[r]) > 0.001:
                        allconverged = 0
                    else:
                        converged[q, r] = 1
                else:
                    now_w[r] = pre_w[r]

            new_l = np.zeros(gallerynum)
            for r in range(rankernum):
                new_l += (data[r, q, :] * now_w[r])
                pre_w[r] = now_w[r]

            now_l = np.argsort(-new_l)
            now_l_rank = np.argsort(now_l)

        new_w[q, :] = now_w

    new_l = np.zeros((querynum, gallerynum))
    for j in range(querynum):
        for r in range(rankernum):
            new_l[j, :] += (data[r, j, :] * new_w[j, r])

    total_ranklist = np.argsort(-new_l, axis=1)
    result = np.argsort(total_ranklist, axis=1)

    # Save the result to a MAT file
    savemat('D:/LocalGit/RA-toolbox/py.mat', {'result': result})


run_dibra()
