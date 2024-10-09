import h5py
import numpy as np
from scipy.io import loadmat


def run_qi_ira(datasetname, inputfileaddress, inter_k, iteration):
    function_name = "qi-ira"
    error_rate = 0.02  # Interaction error rate

    with h5py.File(inputfileaddress, 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    query_label, gallery_label, cam_gallery, cam_query = get_eval_file(datasetname)

    # 加载 .mat 文件
    query_label_data = loadmat(query_label)  # 加载 query_label 的 .mat 文件
    gallery_label_data = loadmat(gallery_label)  # 加载 gallery_label 的 .mat 文件

    # 假设 mat 文件中存储的变量名为 'query_label' 和 'gallery_label'
    query_label = np.array(query_label_data['query_label']).squeeze()  # 提取并转换为 numpy 数组
    gallery_label = np.array(gallery_label_data['gallery_label']).squeeze()  # 提取并转换为 numpy 数组

    rankernum = sim.shape[0]
    querynum = sim.shape[1]
    gallerynum = sim.shape[2]

    feedtrue = np.zeros((querynum, gallerynum))
    feeded = np.zeros((querynum, gallerynum))
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
            now_num = 0
            rt = []

            while sed < inter_k:
                if feeded[q, total_ranklist[q, now_num]] == 0:
                    sed += 1
                    rt.append(total_ranklist[q, now_num])
                    feeded[q, total_ranklist[q, now_num]] = 1
                now_num += 1

            rt_label = gallery_label[rt]
            feedback_p = np.where(rt_label == qlabel)[0]

            for j in range(inter_k):
                if j in feedback_p:
                    if np.random.rand() > error_rate:
                        feedtrue[q, rt[j]] = 10
                else:
                    if np.random.rand() > error_rate:
                        feedtrue[q, rt[j]] = -10

            feedback_p = np.where(feedtrue[q, :] == 10)[0]
            feedback_n = np.where(feedtrue[q, :] == -10)[0]

            if feedback_p.size > 0:
                score_p = sim[:, q, feedback_p]
                score_n = sim[:, q, feedback_n]
                score_p = np.reshape(score_p, (rankernum, feedback_p.size))
                score_n = np.reshape(score_n, (rankernum, feedback_n.size))

                s_p = np.sum(score_p, axis=1) / feedback_p.size
                s_n = np.sum(score_n, axis=1) / feedback_n.size if feedback_n.size > 0 else np.zeros(rankernum)

                s = s_p - s_n if feedback_n.size > 0 else s_p

                new_weight[q, :] += s

        weight = weight * 0.1 + new_weight * 0.9
        for j in range(querynum):
            weight[j, :] /= np.max(weight[j, :])

        new_sim = np.zeros((querynum, gallerynum))
        for j in range(querynum):
            for k in range(rankernum):
                new_sim[j, :] += sim[k, j, :] * weight[j, k]

        new_sim += feedtrue
        total_ranklist = np.argsort(-new_sim, axis=1)
        total_rank = np.argsort(total_ranklist, axis=1)

        res = total_rank.T  # Transpose for final result
    return function_name, total_rank, res


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
