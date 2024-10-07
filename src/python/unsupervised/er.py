import h5py
import scipy.io
import numpy as np
from scipy.stats import norm

def ensemble_ranking(R, tol=1e-10, max_iter=1000):
    alt_no, method_no = R.shape
    i = 0
    R_star = np.zeros((alt_no, 1))

    alpha = np.zeros((1,method_no))  # half-quadratic auxiliary variable
    lambda_weights = alpha  # Lambda plays the role of weights in Ensemble Ranking

    sigma = 0.0
    while i < max_iter:
        Error = R - np.tile(R_star, (1, method_no))
        Error_norm = np.sqrt(np.sum(Error**2, axis=0))

        if np.sum(Error_norm) == 0:
            break

        sigma = (np.linalg.norm(Error_norm, 2)**2 / (2 * len(Error_norm)**2))

        alpha_old = alpha.copy()
        alpha = delta(Error, sigma)

        R_star_old = R_star.copy()
        lambda_weights = alpha / np.sum(alpha)
        R_star = R @ lambda_weights.T
        R_star = R_star.reshape(-1, 1)
        # Convergence conditions
        if np.linalg.norm(alpha - alpha_old, 2) < tol and np.linalg.norm(R_star - R_star_old) < tol:
            break

        i += 1

    # Computing confidence index and trust level
    confMat = norm.pdf(R - np.tile(R_star, (1, R.shape[1])), 0, sigma) / norm.pdf(0, 0, sigma)
    consensusIndex = np.sum(confMat) / (method_no * alt_no)
    trustMat = confMat @ np.diag(lambda_weights)
    trustLevel = np.sum(trustMat) / alt_no

    # Computing final rankings
    sorted_indices = np.argsort(R_star.flatten())
    finalRanking = np.zeros(len(sorted_indices))
    for idx, rank in enumerate(sorted_indices):
        finalRanking[rank] = idx + 1

    return R_star, finalRanking, lambda_weights, consensusIndex, trustLevel, sigma

# The minizer function delta() of the Welsch M-estimator
def delta(v, sigma):
    _, rnkrs = v.shape
    E = np.zeros(rnkrs)
    for i in range(rnkrs):
        E[i] = np.exp(-(np.linalg.norm(v[:, i]))**2 / (2 * sigma**2))
    return E

def run_er():

    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T
    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    item_num = sim.shape[2]

    # 排序
    rank = np.argsort(-sim, axis=2)
    rank = np.argsort(rank, axis=2)

    res = np.zeros((querynum, item_num))

    for i in range(querynum):
        _, finalRanking, *_ = ensemble_ranking(rank[:, i, :].reshape(rankernum, item_num).T)  # 转置为 (item_num, rankernum)
        res[i, :] = finalRanking # 赋值

    # 保存结果到 .mat 文件
    scipy.io.savemat(r'D:\LocalGit\RA-toolbox\py.mat',{'res': res})


run_er()