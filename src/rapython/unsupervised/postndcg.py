"""
PostNDCG Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised ensemble of ranking models for news comments using pseudo answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.

Authors:
    fsw, tancilon
Date:
    2024-9-18
"""
import h5py
import numpy as np
from scipy.io import savemat


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
    savemat(r'D:\LocalGit\RA-toolbox\py.mat', {'result': result})


run_postndcg()
