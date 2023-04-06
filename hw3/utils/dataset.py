import numpy as np
import pandas as pd
from random import shuffle

def load_dataset_from_csv(data_path):
    data = pd.read_csv(data_path, header=None)
    data_np = data.to_numpy()
    return data_np

def load_dataset_from_npy(data_path):
    loadData = np.load(data_path)
    return loadData

def data_gen(data, name):
    #name = [hw1, hw3]
    #Return: orig_data, fea_map, labels, labels_type, fea_num, label_num, data_num, num_data_per_class, classes_data_list
    if isinstance(data, str):
        if name == 'hw1':
            orig_data = load_dataset_from_csv(data)
        elif name == 'hw3':
            orig_data = load_dataset_from_npy(data)
            orig_data = np.concatenate((orig_data[:, 1:], orig_data[:, 0].reshape(orig_data.shape[0], 1)), axis = 1)

    elif isinstance(data, np.ndarray):
        orig_data = data
    fea_map = orig_data[:, :-1]
    labels = orig_data[:, -1]
    labels_type = np.unique(labels)
    return orig_data, fea_map, labels, labels_type
    #return orig_data, fea_map, labels, labels_type, fea_num, label_num, data_num, num_data_per_class, classes_data_list

def data_shuffle(data):
    idx = [i for i in range(data.shape[0])]
    shuffle(idx)
    shuf_data = data[idx, :]
    return shuf_data

def reflect_data(data, labels_type):
    label_num = labels_type.shape[0]
    if label_num != 2:
        print("reflecting data is only for two classes classification!")
        return data
    labels = data[:, -1]
    fea_ma = data[:, :-1]
    fea_ref = np.zeros(fea_ma.shape)
    for i in range(data.shape[0]):
        if labels[i] == labels_type[1]:
            fea_ref[i] = fea_ma[i] * -1
        else:
            fea_ref[i] = fea_ma[i]
    ref_data = np.concatenate((fea_ref, labels.reshape(labels.shape[0], 1)), axis=1)
    return ref_data

'''
#test
#data_path = '../HW1_datasets/dataset1_test.csv'
data_path = '../HW3_datasets/breast_cancer_test.npy'
orig_data, fea_map, labels, labels_type = data_gen(data_path, 'hw3')
#orig_data = data_shuffle(orig_data)
rig_data = reflect_data(orig_data, labels_type)
print('orig_data', rig_data)
print('fea_map', fea_map)
print('labels', labels)
print('labels_type', labels_type)
'''
