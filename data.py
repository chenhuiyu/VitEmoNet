'''
Date: 2021-07-13 23:37:15
LastEditors: Chenhuiyu
LastEditTime: 2021-08-05 20:30:39
FilePath: \\2021-07-AttenEmotionNet\\data_processing\\generate_npy.py
'''
import os
import scipy.io
import numpy as np


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def map_to_2d(data_1d):
    data_2d = np.zeros((data_1d.shape[0], 9, 9, 5))
    data_2d[:, 0, 3:6, :] = data_1d[:, 0:3, :]
    data_2d[:, 1, 3, :] = data_1d[:, 3, :]
    data_2d[:, 1, 5, :] = data_1d[:, 4, :]
    data_2d[:, 2, :, :] = data_1d[:, 5:14, :]
    data_2d[:, 3, :, :] = data_1d[:, 14:23, :]
    data_2d[:, 4, :, :] = data_1d[:, 23:32, :]
    data_2d[:, 5, :, :] = data_1d[:, 32:41, :]
    data_2d[:, 6, :, :] = data_1d[:, 41:50, :]
    data_2d[:, 7, 1:8, :] = data_1d[:, 50:57, :]
    data_2d[:, 8, 2:7, :] = data_1d[:, 57:62, :]
    return data_2d


dataset_path = 'E:/dataset/EEG-Emotion/SEED/ExtractedFeatures'
label_file = os.path.join(dataset_path, 'label.mat')
label = scipy.io.loadmat(label_file)
label_dict = label['label'].squeeze() + 1
# print(label)

X = []
Y = []

for j, file in enumerate(os.listdir(dataset_path)):
    if len(file.split('_')) == 2:
        # 第i个被试
        subject_i = file.split('_')[0]
        # 每个被试有3个section
        section_j = j % 3

        print('subject:', subject_i, ' section:', section_j, file)
        # 读取被试subject_i第section_j次实验的数据，其中包含15个trails
        data = scipy.io.loadmat(os.path.join(dataset_path, file))
        # 对每个trail进行处理
        # trail_id = 1
        for trail_id in range(1, 16):
            for key in data.keys():
                if key == 'de_LDS' + str(trail_id):
                    # print(key)
                    # 读取de特征值，shape(62,2XX,5)
                    # 表示62个导联数，2xx个时间点，5个频带
                    de_data = data[key]
                    data_1d = de_data.transpose((1, 0, 2))
                    data_2d = map_to_2d(data_1d)
                    label = label_dict[trail_id - 1]

                    for i in range(0, len(data_2d), 10):
                        if i + 10 > len(data_2d):
                            break
                        print(i)
                        X.append(data_2d[i:i + 10])
                        Y.append(label)
X = np.array(X)
Y = np.array(Y)

data_path = ".\\input_data"
np.save(os.path.join(data_path, "X.npy"), X)
np.save(os.path.join(data_path, "Y.npy"), Y)
