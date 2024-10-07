import h5py
import numpy as np
import scipy.io

def run_postndcg():
    # 使用 with 语句打开并读取 HDF5 文件
    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    item_num = sim.shape[2]

    # 排序
    ranklist = np.argsort(-sim, axis=2)
    rank = np.argsort(ranklist, axis=2)

    result = np.zeros((querynum, item_num))
    ndcglist = np.zeros((rankernum, rankernum))

    for i in range(querynum):
        for j in range(rankernum - 1):
            for k in range(j + 1, rankernum):
                ndcg = 0
                ranklist1 = ranklist[j, i, :]
                ranklist2 = ranklist[k, i, :]

                for m in range(item_num):
                    if ranklist1[m] == ranklist2[m]:
                        ndcg += np.log(2) / np.log(m + 2)  # m + 2 处理对数

                ndcglist[j, k] = ndcg
                ndcglist[k, j] = ndcg

        ndcgrank = np.sum(ndcglist, axis=1)
        ndcgrank = np.argsort(-ndcgrank)
        result[i, :] = rank[ndcgrank[0], i, :]

    # 保存结果到 .mat 文件
    scipy.io.savemat('D:\LocalGit\RA-toolbox\py.mat', {'result': result})

run_postndcg()
