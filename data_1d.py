'''
Date: 2021-07-13 23:37:15
LastEditors: Chenhuiyu
LastEditTime: 2021-08-05 20:30:39
FilePath: \\2021-07-AttenEmotionNet\\data_processing\\generate_npy.py
'''
import os
import scipy.io
import numpy as np


def awgn(x, snr=30):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    seed = np.random.randint(0, 955)
    np.random.seed(seed)  # 设置随机种子
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(x.shape[0], x.shape[1], x.shape[2]) * np.sqrt(npower)
    return x + noise


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
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

train_data = []
train_label = []
test_data = []
test_label = []
val_data = []
val_label = []
for j, file in enumerate(os.listdir(dataset_path)):
    if len(file.split('_')) == 2:
        # 第i个被试
        subject_i = int(file.split('_')[0])
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

                    CropLength = 25

                    # if section_j == 0:
                    if subject_i == 10:
                        # if subject_i % 2 == 0:
                        data_1d = de_data.transpose((1, 0, 2))
                        data_1d = standardization(data_1d)
                        label = label_dict[trail_id - 1]

                        for i in range(0, len(data_1d), CropLength):
                            if i + CropLength > len(data_1d):
                                break
                            val_data.append(data_1d[i:i + CropLength].transpose(2, 0, 1))
                            val_label.append(label)

                    elif subject_i == 9:
                        data_1d = de_data.transpose((1, 0, 2))
                        data_1d = standardization(data_1d)
                        label = label_dict[trail_id - 1]

                        for i in range(0, len(data_1d), CropLength):
                            if i + CropLength > len(data_1d):
                                break
                            test_data.append(data_1d[i:i + CropLength].transpose(2, 0, 1))
                            test_label.append(label)

                    else:
                        data_1d = de_data.transpose((1, 0, 2))
                        data_1d = standardization(data_1d)
                        label = label_dict[trail_id - 1]

                        for i in range(0, len(data_1d), CropLength):
                            if i + CropLength > len(data_1d):
                                break
                            train_data.append(data_1d[i:i + CropLength].transpose(2, 0, 1))
                            train_label.append(label)

                        # for i in range(5):
                        #     data_1d = de_data.transpose((1, 0, 2))
                        #     data_1d = standardization(data_1d)
                        #     data_1d = awgn(data_1d, snr=30)
                        #     data_2d = map_to_2d(data_1d)
                        #     label = label_dict[trail_id - 1]

                        #     for i in range(0, len(data_2d), CropLength):
                        #         if i + CropLength > len(data_2d):
                        #             break
                        #         train_data.append(data_2d[i:i + CropLength])
                        #         train_label.append(label)

train_data = np.array(train_data)
train_label = np.array(train_label)
val_data = np.array(val_data)
val_label = np.array(val_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

data_path = ".\\input_data_1d"
np.save(os.path.join(data_path, "train_data.npy"), train_data)
np.save(os.path.join(data_path, "train_label.npy"), train_label)

np.save(os.path.join(data_path, "val_data.npy"), val_data)
np.save(os.path.join(data_path, "val_label.npy"), val_label)

np.save(os.path.join(data_path, "test_data.npy"), test_data)
np.save(os.path.join(data_path, "test_label.npy"), test_label)