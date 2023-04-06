import numpy as np
from utils.plotDecBoundaries import plotDecBoundaries
from utils.datasets import data_info, norm_fea, rm_cal, proj_fea
import math
import matplotlib.pyplot as plt

def class_mean(num_class, each_class_list, fea_num):
    cl_mean = np.zeros([num_class, fea_num])
    for cla in range(num_class):
        fea_class = np.array(each_class_list[cla])
        for fea in range(fea_num):
            cl_mean[cla, fea] = np.mean(fea_class[:, fea])
    return cl_mean

def validate(fea_map, labels, cl_mean, labels_type, fea_num):
    misclassified = 0
    eval = np.zeros(labels.shape[0])
    fea_map = fea_map.reshape((fea_map.shape[0], fea_num))
    for n in range(fea_map.shape[0]):
        dis = math.inf
        cl_idx = 0
        for j in range(cl_mean.shape[0]):
            dis_cls = 0
            #print(fea_map[n].shape)
            for i in range(fea_map[n].shape[0]):
                dis_cls += (fea_map[n][i] - cl_mean[j][i])**2
            if dis_cls < dis:
                dis = dis_cls
                cl_idx = j
        eval[n] = labels_type[cl_idx]
        if labels_type[cl_idx] != labels[n]:
            misclassified += 1
    err_rate = misclassified / labels.shape[0]
    return err_rate, eval

def get_best_m(m_map, norm_orig_data):
    fea_map = norm_orig_data[:, :-1]
    labels = norm_orig_data[:, -1]
    errs = np.zeros(m_map.shape)
    best_err = math.inf
    best_m = 0
    for i in range(m_map.shape[0]):
        m = int(m_map[i])
        #print("m", type(int(m)))
        rm = rm_cal(m)
        pro_fea, oned_fea = proj_fea(fea_map, rm)
        oned_data = np.concatenate((oned_fea.reshape((oned_fea.shape[0], 1)), np.array([labels]).T), axis=1)
        #print(oned_data.shape)
        _, _, _, labels_type, fea_num, label_num, _, _, oned_cl_data_list = data_info(oned_data)
        cl_mean_oned = class_mean(label_num, oned_cl_data_list, fea_num)
        #print(cl_mean_oned)
        err_proj, _ = validate(oned_fea, labels, cl_mean_oned, labels_type, 1)
        errs[i] = err_proj
        if err_proj < best_err:
            best_err = err_proj
            best_m = m
            best_rm = rm
            best_cl_mean = cl_mean_oned
            best_pro_fea = pro_fea
            pro_data = np.concatenate((pro_fea, np.array([labels]).T), axis=1)
            _, _, _, _, _, _, _, _, pro_cl_data_list = data_info(pro_data)
            best_cl_meam_fea = class_mean(label_num, pro_cl_data_list, norm_orig_data.shape[1]-1)
    plt.plot(m_map, errs)
    plt.title('training error rate vs. m')
    plt.xlabel('m')
    plt.ylabel('training error')
    plt.show()
    return best_m, best_rm, best_err, best_cl_mean, best_pro_fea, best_cl_meam_fea



if __name__ == '__main__':
    test1_path = './HW1_datasets/dataset1_test.csv'
    train1_path = './HW1_datasets/dataset1_train.csv'
    test2_path = './HW1_datasets/dataset2_test.csv'
    train2_path = './HW1_datasets/dataset2_train.csv'
    test3_path = './HW1_datasets/dataset3_test.csv'
    train3_path = './HW1_datasets/dataset3_train.csv'

    #change train_path and test_path
    orig_data, fea_map, labels, labels_type, fea_num, label_num, data_num, num_data_per_class, classes_data_list = data_info(train3_path)
    orig_data_test, fea_map_test, labels_test, labels_type_test, fea_num_test, label_num_test, data_num_test, num_data_per_class_test, classes_data_list_test = data_info(test3_path)

    #Q(a)
    cl_mean = class_mean(label_num, classes_data_list, fea_num)
    print(cl_mean)
    plotDecBoundaries(fea_map, labels, cl_mean)
    plotDecBoundaries(fea_map_test, labels_test, cl_mean)
    err_rate, eval = validate(fea_map, labels, cl_mean, labels_type, fea_num)
    err_rate_test, eval_test = validate(fea_map_test, labels_test, cl_mean, labels_type_test, fea_num_test)
    print('Error rate for training set:', err_rate)
    print('Error rate for testing set:', err_rate_test)

    #Q(c)
    mean = np.zeros(fea_num)
    std = np.zeros(fea_num)
    for i in range(fea_num):
        mean[i] = np.mean(fea_map[:, i])
        std[i] = np.std(fea_map[:, i])
    print('fea_mean', mean)
    print('fea_std', std)
    norm_fea_map = norm_fea(fea_map, fea_num, mean, std)
    norm_fea_map_test = norm_fea(fea_map_test, fea_num_test, mean, std)
    #print(norm_fea_map.shape, np.array([labels]).T.shape)
    norm_data = np.concatenate((norm_fea_map, np.array([labels]).T), axis=1)
    #print(norm_data)
    #norm_data_test = np.concatenate((norm_fea_map_test, np.array([labels_test]).T), axis=1)

    _, _, _, _, _, _, _, _, norm_cl_data_list = data_info(norm_data)
    #_, _, _, _, _, _, _, _, norm_cl_data_list_test = data_info(norm_data_test)

    cl_mean_norm = class_mean(label_num, norm_cl_data_list, fea_num)
    #cl_mean_norm = class_mean(label_num_test, norm_cl_data_list_test, fea_num_test)
    print(cl_mean_norm)
    plotDecBoundaries(norm_fea_map, labels, cl_mean_norm)
    plotDecBoundaries(norm_fea_map_test, labels_test, cl_mean_norm)
    err_rate_norm, eval_norm = validate(norm_fea_map, labels, cl_mean_norm, labels_type, fea_num)
    err_rate_norm_test, eval_norm_test = validate(norm_fea_map_test, labels_test, cl_mean_norm, labels_type_test, fea_num_test)
    print('Error rate for normalized training set:', err_rate_norm)
    print('Error rate for normalized testing set:', err_rate_norm_test)

    #Q(e)
    m_map = np.arange(0, 40)
    best_m, best_rm, best_err, cl_mean_oned, pro_fea_train, best_cl_mean_fea = get_best_m(m_map, norm_data)
    print("best_m", best_m)
    print("best_rm", best_rm)
    print(best_cl_mean_fea)
    plotDecBoundaries(pro_fea_train, labels, best_cl_mean_fea)
    _, oned_fea_train = proj_fea(norm_fea_map, best_rm)
    pro_fea_test, oned_fea_test = proj_fea(norm_fea_map_test, best_rm)
    plotDecBoundaries(pro_fea_test, labels_test, best_cl_mean_fea)
    err_proj, _ = validate(oned_fea_train, labels, cl_mean_oned, labels_type, 1)
    err_proj_test, _ = validate(oned_fea_test, labels_test, cl_mean_oned, labels_type_test, 1)
    print("err_proj_train", err_proj)
    print("err_proj_test", err_proj_test)




