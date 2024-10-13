"""
HPA Algorithm

This implementation is based on the following reference:

Reference:
-----------
- Fujita, S., Kobayashi, H., & Okumura, M. (2020). Unsupervised ensemble of ranking models for news comments using pseudo answers. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42 (pp. 133-140). Springer International Publishing.

Authors:
    tancilon
Date:
    2024-10-13
"""
import h5py
import numpy as np
from scipy.io import savemat


def hpa_func(sim, topk):
    # 获取画廊数量和排名者数量
    gallerynum = sim.shape[0]
    rankernum = sim.shape[1]

    # 计算平均排名
    averagerank = np.sum(sim, axis=1)
    averagerank = averagerank / np.max(averagerank)  # 防止除以零

    # 获取排名列表
    ranklist = np.argsort(-sim, axis=0)

    # 初始化 ndcg
    ndcg = np.zeros(rankernum)

    # 计算 ndcg 值
    for i in range(rankernum):
        for j in range(topk):
            ndcg[i] += averagerank[ranklist[j, i]] * np.log(2) / np.log(i + 2)

    # 获取 ndcg 排名
    ndcgrank = np.argsort(-ndcg)

    # 初始化 finalrank
    finalrank = np.zeros(gallerynum)

    # 计算最终排名
    for i in range(rankernum):
        finalrank += ndcg[ndcgrank[i]] * sim[:, ndcgrank[i]]

    # 对最终排名进行排序
    finalrank = np.argsort(-finalrank)
    finalrank = np.argsort(finalrank)

    return finalrank


def run_hpa():
    # 使用 with 语句打开并读取 HDF5 文件
    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    item_num = sim.shape[2]

    topk = item_num
    result = np.zeros((querynum, item_num))

    for i in range(querynum):
        finalrank = hpa_func(sim[:, i, :].reshape(rankernum, item_num).T, topk)
        result[i, :] = finalrank.flatten()  # 确保行向量形式

    # 保存 .mat 文件
    savemat(r'D:\LocalGit\RA-toolbox\py.mat', {'res': result})


run_hpa()
