import h5py
import numpy as np
import scipy.io
import time



def hpa_func(sim, topK):
    # 获取画廊数量和排名者数量
    galleryNum = sim.shape[0]
    rankerNum = sim.shape[1]

    # 计算平均排名
    averageRank = np.sum(sim, axis=1)
    averageRank = averageRank / np.max(averageRank)  # 防止除以零

    # 获取伪排名列表
    pseudoRanklist = np.argsort(averageRank)
    pseudoRank = np.argsort(pseudoRanklist)

    # 获取排名列表
    ranklist = np.argsort(-sim, axis=0)

    # 初始化 NDCG
    NDCG = np.zeros(rankerNum)

    # 计算 NDCG 值
    for i in range(rankerNum):
        for j in range(topK):
                NDCG[i] += averageRank[ranklist[j, i]] * np.log(2) / np.log(i + 2)

    # 获取 NDCG 排名
    NDCGrank = np.argsort(-NDCG)

    # 初始化 finalRank
    finalRank = np.zeros(galleryNum)

    # 计算最终排名
    for i in range(rankerNum):
        finalRank += NDCG[NDCGrank[i]] * sim[:, NDCGrank[i]]

    # 对最终排名进行排序
    finalRank = np.argsort(-finalRank)
    finalRank = np.argsort(finalRank)

    return finalRank


def run_hpa():

    # 使用 with 语句打开并读取 HDF5 文件
    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    item_num = sim.shape[2]

    # 排序
    rank = np.argsort(-sim, axis=2)
    rank = np.argsort(rank, axis=2)

    topK = item_num
    result = np.zeros((querynum, item_num))

    for i in range(querynum):
        finalRank = hpa_func(sim[:, i, :].reshape(rankernum, item_num).T, topK)
        result[i, :] = finalRank.flatten()  # 确保行向量形式

    # 保存 .mat 文件
    scipy.io.savemat(r'D:\LocalGit\RA-toolbox\py.mat',{'res': result})



run_hpa()


