# -*- coding:utf-8 -*-
u'''
Created on 2021年4月19日
@author: xianyu
@description：主程序中需要调用的若干工具函数
'''
__author__ = 'xianyu'
__version__ = '1.0.0'
__company__ = u'STDU'
__updated__ = '2021-04-19'

import os
import shutil
import random
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split  # 用sklearn的PCA
from sklearn.decomposition import PCA
from Pretreatment import *


class Utils():
    def __init__(self):

        # 以下为全新定义的变量
        self.img_marked_info_1 = []
        self.img_marked_info_2 = []
        self.img_unmarked_info_1 = []
        self.img_unmarked_info_2 = []

        pass

    def get_Label_List(self, dataSet_name):
        data_target = sio.loadmat(dataSet_name)['target'][0]
        return np.unique(data_target).tolist()

    def init_DataSet(self, dataSet_name):
        target = sio.loadmat(dataSet_name)['target'][0]
        data = sio.loadmat(dataSet_name)['data']
        return data, target

    def PCA(self, data, PCA_n=300):
        pca = PCA(n_components=PCA_n)
        pca.fit(np.array(data))
        data = pca.transform(np.array(data))
        return data

    def RSM(self, data, RSM_len):
        pretreatment = Pretreatment()
        data_RSM_1 = pretreatment.reduceFeatureWithRSM(data, RSM_len)
        data_RSM_2 = pretreatment.reduceFeatureWithRSM(data, RSM_len)
        return data_RSM_1, data_RSM_2

    def split_Data(self, ratio_train, ratio_validation, data, target, isRSM=True, RSM_len=900):
        if isRSM:
            pretreatment = Pretreatment()
            data_RSM = pretreatment.reduceFeatureWithRSM(data, RSM_len)
            
            img_model_x, img_validation_x, img_model_y, img_validation_y = \
                train_test_split(data_RSM, target, test_size=ratio_validation, random_state=0)
            img_marked_x_1, img_unmarked_x_1, img_marked_y_1, img_unmarked_y_1 = \
                train_test_split(img_model_x, img_model_y, test_size=1 - ratio_train, random_state=0)

            img_marked_x_2, img_unmarked_x_2, img_marked_y_2, img_unmarked_y_2 = \
                train_test_split(img_model_x, img_model_y, test_size=1 - ratio_train, random_state=0)
        else:
            img_model_x, img_validation_x, img_model_y, img_validation_y = \
                train_test_split(data, target, test_size=ratio_validation, random_state=0)
            img_marked_x_1, img_unmarked_x_1, img_marked_y_1, img_unmarked_y_1 = \
                train_test_split(img_model_x, img_model_y, test_size=1 - ratio_train, random_state=0)

            img_marked_x_2, img_unmarked_x_2, img_marked_y_2, img_unmarked_y_2 = \
                train_test_split(img_model_x, img_model_y, test_size=1 - ratio_train, random_state=0)

        for i in range(len(img_marked_x_1)):
            self.img_marked_info_1.append([img_marked_x_1[i], img_marked_y_1[i], 0])
        for i in range(len(img_unmarked_x_1)):
            self.img_unmarked_info_1.append([img_unmarked_x_1[i], img_unmarked_y_1[i], 0])
        for i in range(len(img_marked_x_2)):
            self.img_marked_info_2.append([img_marked_x_2[i], img_marked_y_2[i], 0])
        for i in range(len(img_unmarked_x_2)):
            self.img_unmarked_info_2.append([img_unmarked_x_2[i], img_unmarked_y_2[i], 0])

        # 返回两套样本信息的长度
        return len(img_marked_x_1), len(img_unmarked_x_1), img_validation_x, img_validation_y


    def get_train_test_data(self, batch_size):
        # global sub_index
        if batch_size > len(self.img_marked_info_1):
            print('batch size大于样本数，训练结束！')
            return [], [], [], [], [], [], [], []
        else:
            index = np.arange(0, len(self.img_marked_info_1))
            random.shuffle(index)  # 随机打乱数组顺序

            sub_index = index[0:batch_size]
            sub_index = np.sort(sub_index)

            img_marked_info_sampled_1 = [self.img_marked_info_1[i] for i in sub_index]
            img_marked_info_sampled_2 = [self.img_marked_info_2[i] for i in sub_index]

            # 被抽中的样本权重值+1
            for i in sub_index:
                self.img_marked_info_1[i][2] = self.img_marked_info_1[i][2] + 1
                self.img_marked_info_2[i][2] = self.img_marked_info_2[i][2] + 1

            train_x_1 = [i[0] for i in img_marked_info_sampled_1]
            test_x_1 = [i[0] for i in self.img_unmarked_info_1]
            train_y_1 = [i[1] for i in img_marked_info_sampled_1]
            test_y_1 = [i[1] for i in self.img_unmarked_info_1]
            train_x_2 = [i[0] for i in img_marked_info_sampled_2]
            test_x_2 = [i[0] for i in self.img_unmarked_info_2]
            train_y_2 = [i[1] for i in img_marked_info_sampled_2]
            test_y_2 = [i[1] for i in self.img_unmarked_info_2]

            return np.array(train_x_1), np.array(test_x_1), np.array(train_y_1), np.array(test_y_1),\
                   np.array(train_x_2), np.array(test_x_2), np.array(train_y_2), np.array(test_y_2)

    def update_DataSet(self, same_index, threshold_Weight):
        # 判断权重值，移除参与训练次数大于阈值的样本
        print('before len self.img_marked_info_1 = ', len(self.img_marked_info_1))
        print('before len self.img_unmarked_info_1 = ', len(self.img_unmarked_info_1))
        self.img_marked_info_1 = [i for i in self.img_marked_info_1 if i[2] < threshold_Weight]

        # 转入BvSB算法选出的样本
        for i in same_index:
            # print('i = ', i)
            self.img_marked_info_1.append(self.img_unmarked_info_1[i])

        self.img_unmarked_info_1 = [self.img_unmarked_info_1[i] for i in range(len(self.img_unmarked_info_1)) if
                                   i not in same_index]
        print('after len self.img_marked_info_1 = ', len(self.img_marked_info_1))
        print('after len self.img_unmarked_info_1 = ', len(self.img_unmarked_info_1))


        # 判断权重值，移除参与训练次数大于阈值的样本
        print('before len self.img_marked_info_2 = ', len(self.img_marked_info_2))
        print('before len self.img_unmarked_info_2 = ', len(self.img_unmarked_info_2))
        self.img_marked_info_2 = [i for i in self.img_marked_info_2 if i[2] < threshold_Weight]

        # 转入BvSB算法选出的样本
        for i in same_index:
            # print('i = ', i)
            self.img_marked_info_2.append(self.img_unmarked_info_2[i])

        self.img_unmarked_info_2 = [self.img_unmarked_info_2[i] for i in range(len(self.img_unmarked_info_2)) if
                                   i not in same_index]
        print('after len self.img_marked_info_2 = ', len(self.img_marked_info_2))
        print('after len self.img_unmarked_info_2 = ', len(self.img_unmarked_info_2))


if __name__ == '__main__':
    utils = Utils()
    result = utils.get_Label_List()
    print(result)

    utils.init_DataSet(0.5)
    utils.get_train_test_data(30)