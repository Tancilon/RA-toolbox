import h5py
import numpy as np
from scipy.io import loadmat, savemat


def run_irank():
    """
    sim: voter * query * item
    """
    # 加载数据
    # sim = loadmat(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat")['workerlist_sim']
    # 使用 with 语句打开并读取 HDF5 文件
    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    itemnum = sim.shape[2]

    print(sim.shape)
    # 排序并得到ranklist和rank
    ranklist = np.argsort(-sim, axis=2)
    rank = np.argsort(ranklist, axis=2)

    topk = itemnum
    result = np.zeros((querynum, itemnum))
    superviserank = np.zeros((rankernum - 1, querynum, itemnum))
    newsim = np.zeros((rankernum, querynum, itemnum))

    # 迭代三次
    for iteration in range(3):
        newsim = sim * 0.9
        for i in range(rankernum):
            if i == 0:
                superviserank = rank[1:rankernum + 1, :, :]
            elif i == rankernum - 1:
                superviserank = rank[0:rankernum, :, :]
            else:
                superviserank[:i, :, :] = rank[:i, :, :]
                superviserank[i:rankernum, :, :] = rank[i + 1:rankernum + 1, :, :]

            Dscore = 1.0 / superviserank
            print(Dscore.shape)
            DscoreTotal = np.sum(Dscore, axis=0)
            print(DscoreTotal.shape)
            Sresultlist = np.argsort(-DscoreTotal, axis=1)
            # Sresult = np.argsort(Sresultlist, axis=2)
            # Sresult = Sresult.reshape(querynum, itemnum)
            Sresultlist = Sresultlist.reshape(querynum, itemnum)

            for k in range(querynum):
                for l in range(topk):
                    newsim[i, k, Sresultlist[k, l]] += 0.1

        sim = newsim
        ranklist = np.argsort(-sim, axis=2)
        rank = np.argsort(ranklist, axis=2)

    finalsim = np.sum(sim, axis=0)
    print(finalsim.shape)
    finalsim = finalsim.reshape(querynum, itemnum)
    result = np.argsort(-finalsim, axis=1)
    result = np.argsort(result, axis=1)
    # 保存排名结果
    savemat(r'D:\LocalGit\RA-toolbox\py.mat', {'result': result})




run_irank()


