import h5py
import numpy as np
import scipy
from scipy.io import loadmat


def run_qi_ira(datasetname, inputfileAddress, K, iteration):
    functionName = f"QI-IRA({K},{iteration})"

    print(f'QT_IRA Running {datasetname}')

    error_rate = 0.02  # Interaction error rate


    with h5py.File(r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat", 'r') as f:
        # 读取数据集
        sim = f['workerlist_sim'][:].T

    print(f"{functionName} Running {datasetname}")

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
            Qlabel = query_label[q]
            sed = 0
            now_num = 0
            RT = []

            while sed < K:
                if feeded[q, total_ranklist[q, now_num]] == 0:
                    sed += 1
                    RT.append(total_ranklist[q, now_num])
                    feeded[q, total_ranklist[q, now_num]] = 1
                now_num += 1

            RT_label = gallery_label[RT]
            feedback_P = np.where(RT_label == Qlabel)[0]

            for j in range(K):
                if j in feedback_P:
                    if np.random.rand() > error_rate:
                        feedtrue[q, RT[j]] = 10
                else:
                    if np.random.rand() > error_rate:
                        feedtrue[q, RT[j]] = -10

            feedback_P = np.where(feedtrue[q, :] == 10)[0]
            feedback_N = np.where(feedtrue[q, :] == -10)[0]

            if feedback_P.size > 0:
                score_P = sim[:, q, feedback_P]
                score_N = sim[:, q, feedback_N]
                score_P = np.reshape(score_P, (rankernum, feedback_P.size))
                score_N = np.reshape(score_N, (rankernum, feedback_N.size))

                S_P = np.sum(score_P, axis=1) / feedback_P.size
                S_N = np.sum(score_N, axis=1) / feedback_N.size if feedback_N.size > 0 else np.zeros(rankernum)

                S = S_P - S_N if feedback_N.size > 0 else S_P

                new_weight[q, :] += S

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

    return functionName, total_rank, res
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
        gallery_label_path = r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID-gallery_idtest-.mat"
        cam_gallery_path = r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID-gallery_camidstest-.mat"
        cam_query_path = r"D:\RA_ReID\Person-ReID\label&cam\DukeMTMC_VideoReID\AGRL-DukeMTMC_VideoReID-query_camids-.mat"

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

inputfileAddress = r"D:\RA_ReID\Person-ReID\test\cuhk03detected_6workers.mat"
_, total_rank, _ = run_qi_ira("cuhk03detected", inputfileAddress, 3, 1)
scipy.io.savemat(r'D:\LocalGit\RA-toolbox\py.mat', {'res': total_rank})