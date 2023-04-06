import pandas as pd
import numpy as np

#All the process functions for the dataset

def load_dataset_from_csv(data_path):
    data = pd.read_csv(data_path, header=None)
    data_np = data.to_numpy()
    return data_np


def data_info(data):
    #Return: orig_data, fea_map, labels, labels_type, fea_num, label_num, data_num, num_data_per_class, classes_data_list
    if isinstance(data, str):
        orig_data = load_dataset_from_csv(data)
    elif isinstance(data, np.ndarray):
        orig_data = data
    fea_map = orig_data[:, :-1]
    labels = orig_data[:, -1]
    labels_type = np.unique(labels)
    fea_num = fea_map.shape[1]
    label_num = labels_type.shape[0]
    data_num = orig_data.shape[0]
    num_data_per_class = np.zeros(label_num)
    classes_data_list = []
    for n in range(label_num):
        classes_data_list.append([])
    for i in range(data_num):
        data = fea_map[i]
        label = labels[i]
        for lab in range(label_num):
            if label == labels_type[lab]:
                classes_data_list[lab].append(data.tolist())
                num_data_per_class[lab] += 1


    return orig_data, fea_map, labels, labels_type, fea_num, label_num, data_num, num_data_per_class, classes_data_list

def norm_fea(fea_map, fea_num, mean, std):
    norm_fea_map = np.zeros(fea_map.shape)
    for i in range(fea_num):
        fea = fea_map[:, i]
        #mean = np.mean(fea)
        #std = np.std(fea)
        norm_fea_map[:, i] = (fea - mean[i]) / std[i]
    return norm_fea_map

def rm_cal(m):
    #fea_num = 2 defaultly
    if isinstance(m, int) or isinstance(m, float):
        rm = np.zeros(2)
        if m>=0 and m<=9:
            rm = np.array([10, m])
        elif m>=10 and m<=29:
            rm = np.array([20-m, 10])
        elif m>=30 and m<=39:
            rm = np.array([-10, 40-m])
        return rm
    else:
        print("m error")
        return

def proj(a, b):
    #a project on b
    t = np.dot(a,b)/np.dot(b,b)
    proj_b = t * b
    cos_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return proj_b, cos_ab

def proj_fea(fea_map, rm):
    fea_size = fea_map.shape
    proj_fea = np.zeros(fea_size)
    oned_fea = np.zeros(fea_size[0])
    zero = np.zeros(fea_size[1])
    for n in range(fea_size[0]):
        fea = fea_map[n]
        #print(fea.shape)
        #print(rm.shape)
        proj_fea[n], cos_fea = proj(fea, rm)
        if cos_fea > 0:
            oned_fea[n] = np.linalg.norm(proj_fea[n] - zero)
        else:
            oned_fea[n] = -np.linalg.norm(proj_fea[n] - zero)

    return proj_fea, oned_fea


#test
data_path = '../HW1_datasets/dataset1_test.csv'
orig_data, fea_map, labels, labels_type, fea_num, label_num, data_num, num_data_per_class, classes_data_list = data_info(data_path)
print('orig_data', orig_data)
print('fea_map', fea_map)
print('labels', labels)
print('labels_type', labels_type)
print('fea_num', fea_num)
print('label_num', label_num)
print('data_num', data_num)
print('num_data_per_class', num_data_per_class)
print('classes_data_list', classes_data_list)




