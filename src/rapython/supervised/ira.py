from enum import Enum, auto

import h5py
import numpy as np
from scipy.io import loadmat, savemat


class MethodType(Enum):
    RANK = auto()
    SCORE = auto()


def rank_based_ira(datasetname, inputfileaddress, K, iteration):
    function_name = f"IRA_R({K},{iteration})"

    with h5py.File(inputfileaddress, 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    print(f"{function_name} Running {datasetname}")

    query_label, gallery_label, cam_gallery, cam_query = get_eval_file(datasetname)

    # 加载 .mat 文件
    query_label_data = loadmat(query_label)  # 加载 query_label 的 .mat 文件
    gallery_label_data = loadmat(gallery_label)  # 加载 gallery_label 的 .mat 文件

    # 假设 mat 文件中存储的变量名为 'query_label' 和 'gallery_label'
    query_label = np.array(query_label_data['query_label']).squeeze()  # 提取并转换为 numpy 数组
    gallery_label = np.array(gallery_label_data['gallery_label']).squeeze()  # 提取并转换为 numpy 数组

    error_rate = 0.02  # Interaction error rate

    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    gallerynum = sim.shape[2]

    ranklist = np.argsort(-sim, axis=2)

    feedtrue_g = np.zeros((querynum, gallerynum))
    feeded_g = np.zeros((querynum, gallerynum))

    weight = np.ones((querynum, rankernum))

    # Get origin rank
    origin_sim = np.zeros((querynum, gallerynum))
    for i in range(querynum):
        for j in range(rankernum):
            origin_sim[i, :] += sim[j, i, :] * weight[i, j]

    origin_ranklist = np.argsort(-origin_sim, axis=1)
    total_ranklist = origin_ranklist

    for i in range(iteration):
        new_weight = np.zeros((querynum, rankernum))

        for q in range(querynum):
            qlabel = query_label[q]
            sed = 0
            now_num = 1
            rt = []
            while sed < K:
                if feeded_g[q, total_ranklist[q, now_num - 1]] == 0:
                    sed += 1
                    rt.append(total_ranklist[q, now_num - 1])
                    feeded_g[q, total_ranklist[q, now_num - 1]] = 1
                now_num += 1

            rt_label = gallery_label[rt]
            scored_g = np.where(rt_label == qlabel)[0]
            for j in range(K):
                if j in scored_g:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = 10
                    else:
                        feedtrue_g[q, rt[j]] = -10
                else:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = -10
                    else:
                        feedtrue_g[q, rt[j]] = 10

            scored_g = np.where(feedtrue_g[q, :] == 10)[0]
            for j in range(rankernum):
                ranker_rt = ranklist[j, q, :]
                for k in scored_g:
                    x = np.where(ranker_rt == k)[0][0]
                    score = np.ceil(x / K)
                    if score == 0:
                        continue
                    new_weight[q, j] += 1 / score

        weight = weight * 0.1 + new_weight * 0.9
        for j in range(querynum):
            weight[j, :] /= np.max(weight[j, :])

        new_sim = np.zeros((querynum, gallerynum))
        for j in range(querynum):
            for k in range(rankernum):
                new_sim[j, :] += sim[k, j, :] * weight[j, k]

        new_sim += feedtrue_g

        total_ranklist = np.argsort(-new_sim, axis=1)
        total_rank = np.argsort(total_ranklist, axis=1)

        res = total_rank.T
    return function_name, total_rank, res


def score_based_ira(datasetname, inputfile_address, K, iteration):
    function_name = f"IRA_S({K},{iteration})"

    with h5py.File(inputfile_address, 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    query_label, gallery_label, cam_gallery, cam_query = get_eval_file(datasetname)

    # 加载 .mat 文件
    query_label_data = loadmat(query_label)  # 加载 query_label 的 .mat 文件
    gallery_label_data = loadmat(gallery_label)  # 加载 gallery_label 的 .mat 文件

    # 假设 mat 文件中存储的变量名为 'query_label' 和 'gallery_label'
    query_label = np.array(query_label_data['query_label']).squeeze()  # 提取并转换为 numpy 数组
    gallery_label = np.array(gallery_label_data['gallery_label']).squeeze()  # 提取并转换为 numpy 数组

    error_rate = 0.02  # Interaction error rate
    print(f'Running {function_name}')

    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    gallerynum = sim.shape[2]

    feedtrue_g = np.zeros((querynum, gallerynum))
    feeded_g = np.zeros((querynum, gallerynum))
    weight = np.ones((querynum, rankernum))

    # get origin rank
    origin_sim = np.zeros((querynum, gallerynum))
    for i in range(querynum):
        for j in range(rankernum):
            origin_sim[i, :] += sim[j, i, :] * weight[i, j]

    origin_ranklist = np.argsort(-origin_sim, axis=1)
    total_ranklist = origin_ranklist

    for i in range(iteration):
        new_weight = np.zeros((querynum, rankernum))

        for q in range(querynum):
            qlabel = query_label[q]
            sed = 0
            now_num = 0
            rt = []
            while sed < K:
                if feeded_g[q, total_ranklist[q, now_num]] == 0:
                    sed += 1
                    rt.append(total_ranklist[q, now_num])
                    feeded_g[q, total_ranklist[q, now_num]] = 1
                now_num += 1

            rt_label = gallery_label[rt]
            scored_g = np.where(rt_label == qlabel)[0]

            for j in range(K):
                if j in scored_g:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = 10
                    else:
                        feedtrue_g[q, rt[j]] = -10
                else:
                    if np.random.rand() > error_rate:
                        feedtrue_g[q, rt[j]] = -10
                    else:
                        feedtrue_g[q, rt[j]] = 10

            scored_g = np.where(feedtrue_g[q, :] == 10)[0]
            if scored_g.size > 1:
                anno_g = sim[:, q, scored_g].reshape(rankernum, scored_g.size)
                std_w = np.std(anno_g, axis=1)
                max_std = np.max(std_w)
                std_w = std_w / max_std
                new_weight[q, :] += 1.0 / std_w
                total_weight = np.max(new_weight[q, :])
                new_weight[q, :] /= total_weight

        weight = weight * 0.1 + new_weight * 0.9
        weight /= np.max(weight, axis=1, keepdims=True)

        new_sim = np.zeros((querynum, gallerynum))
        for j in range(querynum):
            for k in range(rankernum):
                new_sim[j, :] += sim[k, j, :] * weight[j, k]

        new_sim += feedtrue_g
        total_ranklist = np.argsort(-new_sim, axis=1)
        total_rank = np.argsort(total_ranklist, axis=1)
        res = total_rank.T
    return function_name, total_rank, res


def run_ira(mode=MethodType.RANK):
    inputfile_address = r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat"
    if mode == MethodType.RANK:
        _, total_rank, _ = rank_based_ira("cuhk03detected", inputfile_address, 3, 1)
        savemat(r'D:\LocalGit\RA-toolbox\py.mat', {'res': total_rank})
    elif mode == MethodType.SCORE:
        _, total_rank, _ = score_based_ira("cuhk03detected", inputfile_address, 3, 1)
        savemat(r'D:\LocalGit\RA-toolbox\py.mat', {'res': total_rank})


def get_eval_file(datasetname):
    if datasetname == "cuhk03detected":
        query_label_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03detected\bdb-cuhk03detected-query_id-.mat"
        gallery_label_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03detected\bdb-cuhk03detected-gallery_idtest-.mat"
        cam_gallery_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03detected\bdb-cuhk03detected-gallery_camidstest-.mat"
        cam_query_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03detected\bdb-cuhk03detected-query_camids-.mat"

    elif datasetname == "cuhk03labeled":
        query_label_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03labeled\bdb-cuhk03labeled-query_id-.mat"
        gallery_label_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03labeled\bdb-cuhk03labeled-gallery_idtest-.mat"
        cam_gallery_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03labeled\bdb-cuhk03labeled-gallery_camidstest-.mat"
        cam_query_path = r"D:\RA_ReID\Person-ReID\label&cam\cuhk03labeled\bdb-cuhk03labeled-query_camids-.mat"

    elif datasetname == "DukeMTMC_VideoReID":
        query_label_path = r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID-query_id-.mat"
        gallery_label_path = (r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID"
                              r"-gallery_idtest-.mat")
        cam_gallery_path = (r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID"
                            r"-gallery_camidstest-.mat")
        cam_query_path = (r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID-query_camids"
                          r"-.mat")

    elif datasetname == "dukemtmcreid":
        query_label_path = r"D:\RA_ReID\Person-ReID\label&cam\dukemtmcreid\bdb-dukemtmcreid-query_id-.mat"
        gallery_label_path = r"D:\RA_ReID\Person-ReID\label&cam\dukemtmcreid\bdb-dukemtmcreid-gallery_idtest-.mat"
        cam_gallery_path = r"D:\RA_ReID\Person-ReID\label&cam\dukemtmcreid\bdb-dukemtmcreid-gallery_camidstest-.mat"
        cam_query_path = r"D:\RA_ReID\Person-ReID\label&cam\dukemtmcreid\bdb-dukemtmcreid-query_camids-.mat"

    elif datasetname == "market1501":
        query_label_path = r"D:\RA_ReID\Person-ReID\label&cam\market1501\bdb-market1501-query_id-.mat"
        gallery_label_path = r"D:\RA_ReID\Person-ReID\label&cam\market1501\bdb-market1501-gallery_idtest-.mat"
        cam_gallery_path = r"D:\RA_ReID\Person-ReID\label&cam\market1501\bdb-market1501-gallery_camidstest-.mat"
        cam_query_path = r"D:\RA_ReID\Person-ReID\label&cam\market1501\bdb-market1501-query_camids-.mat"

    elif datasetname == "MARS":
        query_label_path = r"D:\RA_ReID\Person-ReID\label&cam\MARS\AGRL-MARS-query_id-.mat"
        gallery_label_path = r"D:\RA_ReID\Person-ReID\label&cam\MARS\AGRL-MARS-gallery_idtest-.mat"
        cam_gallery_path = r"D:\RA_ReID\Person-ReID\label&cam\MARS\AGRL-MARS-gallery_camidstest-.mat"
        cam_query_path = r"D:\RA_ReID\Person-ReID\label&cam\MARS\AGRL-MARS-query_camids-.mat"

    else:
        query_label_path = ""
        gallery_label_path = ""
        cam_gallery_path = ""
        cam_query_path = ""

    return query_label_path, gallery_label_path, cam_gallery_path, cam_query_path


run_ira(MethodType.SCORE)
