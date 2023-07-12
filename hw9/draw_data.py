import numpy as np

def draw_data(data_num):
    k = np.random.randint(1, 3, data_num)
    mean1 = np.array([-4, 0])
    var1 = np.array([[4, 0], [0, 1]])
    mean2 = np.array([-0.5, 0])
    var2 = np.array([[0.16, 0], [0, 9]])

    cla1_help = np.random.rand(data_num) >= 0.7
    cla1_data = np.random.multivariate_normal(mean1, var1, size=data_num) * np.tile(((cla1_help -1) * (-1)).reshape((len(cla1_help), 1)), 2)
    cla1_data += np.concatenate(((np.random.rand(data_num)*2).reshape((data_num, 1)), (np.random.rand(data_num)*2-1).reshape((data_num, 1))), axis=1) * np.tile(cla1_help.reshape((len(cla1_help), 1)), 2)
    cla2_data = np.random.multivariate_normal(mean2, var2, size=data_num)
    data = cla1_data * np.tile(((k-2)*(-1)).reshape((len(k), 1)), 2) + cla2_data * np.tile((k-1).reshape((len(k), 1)), 2)
    data = np.concatenate((data, k.reshape((len(k), 1))), axis=1)
    return data



