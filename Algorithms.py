# -*- coding:utf-8 -*-
u'''
Created on 2021年4月19日
@author: xianyu
@description：主程序中需要调用的若干算法函数
'''
__author__ = 'xianyu'
__version__ = '1.0.0'
__company__ = u'STDU'
__updated__ = '2021-04-19'

# import sys
# import os
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前程序文件的目录
# sys.path.append(BASE_DIR) #添加环境变量

import random
import numpy as np
import cv2
from sklearn import svm
import operator
from sklearn.metrics import accuracy_score
from Utils import *

#import torch
#import torch.nn
#import torchvision.models as models
#from torch.autograd import Variable
#import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from PIL import Image

TARGET_IMG_SIZE = 224
#img_to_tensor = transforms.ToTensor()


class Algorithms():
    def __init__(self):
        pass






    # 特征提取
    def get_VGG_Features(self, model, imgpath):
        model.eval()  # 必须要有，不然会影响特征提取结果
        img = Image.open(imgpath)  # 读取图片
        # print('图像通道数：', len(img.getbands()))
        if len(img.getbands()) == 3:
            img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            tensor = img_to_tensor(img)  # 将图片转化成tensor
            tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
            # print(tensor.shape)  # [3, 224, 224]
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)

            # print(tensor.shape)  # [1,3, 224, 224]
            tensor = tensor.cuda()

            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
            # print(result_npy)

            return result_npy[0].tolist()  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]
        else:
            return []

    def RSM_VGG(self, features_in, num):  # VGG特征的RSM算法
        index = np.arange(0, len(features_in))
        # print('len(features_in)=', len(features_in), 'num=', num)
        # print('index = ', index)
        random.shuffle(index)  # 随机打乱数组顺序

        sub_index = index[0:num]
        sub_index = np.sort(sub_index)

        features_out = [features_in[i] for i in sub_index]  # 按照抽样的子空间索引并输出
        return features_out

    def BvSB(self, Y_pred_proba_1, Y_pred_proba_2, ratio, threshold_BvSB, isBvSB=True):  # BvSB算法实现
        # 分类器1给出的预测值处理
        diff_B_SB_1 = []  # 二维数组，每行如下：[样本序号, best和second best的差值, best的下标, best的值]
        i = 0
        for proba_pic in Y_pred_proba_1:
            best_index = proba_pic.index(max(proba_pic))
            best_val = max(proba_pic)
            proba_pic[best_index] = 0  # 将最大值的部分清零，剩下的部分继续寻找最大值即为second best
            second_best_index = proba_pic.index(max(proba_pic))
            second_best_val = max(proba_pic)
            # print(best_index, best_val, second_best_index, second_best_val)
            # 获取best和second best的差值，按顺序存入diff_B_SB中，并存入best的下标
            diff_B_SB_1.append([i, (best_val - second_best_val), best_index, best_val])
            i = i + 1

        # print(diff_B_SB_1)
        if isBvSB:
            diff_B_SB_1.sort(key=operator.itemgetter(1), reverse=True)  # 将差值由高到低进行排序
        else:
            diff_B_SB_1.sort(key=operator.itemgetter(3), reverse=True)  # 将差值由高到低进行排序
        diff_B_SB_1 = diff_B_SB_1[0:max(int(ratio * len(diff_B_SB_1)), 1)]  # 取前5%，要是少于20张，20x5%小于1，则取1
        diff_B_SB_1 = [x for x in diff_B_SB_1 if x[3] > threshold_BvSB]  # best-secondbest的差值取满足阈值的样本

        # 分类器2给出的预测值处理
        diff_B_SB_2 = []  # 二维数组，每行如下：[样本序号, best和second best的差值, best的下标]
        i = 0
        for proba_pic in Y_pred_proba_2:
            best_index = proba_pic.index(max(proba_pic))
            best_val = max(proba_pic)
            proba_pic[best_index] = 0  # 将最大值的部分清零，剩下的部分继续寻找最大值即为second best
            second_best_index = proba_pic.index(max(proba_pic))
            second_best_val = max(proba_pic)
            # print(best_index, best_val, second_best_index, second_best_val)
            # 获取best和second best的差值，按顺序存入diff_B_SB中，并存入best的下标
            diff_B_SB_2.append([i, (best_val - second_best_val), best_index, best_val])
            i = i + 1

        # print(diff_B_SB_2)
        if isBvSB:
            diff_B_SB_2.sort(key=operator.itemgetter(1), reverse=True)  # 将差值由高到低进行排序
        else:
            diff_B_SB_2.sort(key=operator.itemgetter(3), reverse=True)  # 将差值由高到低进行排序
        diff_B_SB_2 = diff_B_SB_2[0:int(ratio * len(diff_B_SB_2))]  # 取前5%
        diff_B_SB_2 = [x for x in diff_B_SB_2 if x[3] > threshold_BvSB]  # best-secondbest的差值取满足阈值的样本

        return diff_B_SB_1, diff_B_SB_2

    def SVM_Training_And_Testing(self, X_train_1, X_train_2, Y_train_1, Y_train_2, X_test_1, X_test_2, Y_test_1,
                                 Y_test_2, validation_x, validation_y, label_list):
        # 分别利用X_train_1和X_train_2以及Y_train训练两个支持向量机分类器svm_classifier_1和svm_classifier_2
        # svm_classifier_1 = svm.SVC(C=300, kernel='rbf', decision_function_shape='ovr', gamma='auto', probability=True)
        svm_classifier_1 = svm.SVC(C=1000, kernel='rbf', decision_function_shape='ovr', gamma='auto', probability=True)
        svm_classifier_1.fit(X_train_1, Y_train_1)
        # print('svm_classifier_1 = ', svm_classifier_1)
        svm_classifier_2 = svm.SVC(C=1000, kernel='rbf', decision_function_shape='ovr', gamma='auto', probability=True)
        svm_classifier_2.fit(X_train_2, Y_train_2)
        # print('svm_classifier_2 = ', svm_classifier_2)

        # 分别使用svm_classifier_1和svm_classifier_2对测试集进行测试，输出每个样本对应各类的概率值
        Y_pred_proba_1 = svm_classifier_1.predict_proba(validation_x)
        Y_pred_proba_2 = svm_classifier_2.predict_proba(validation_x)
        Y_pred_proba_1 = Y_pred_proba_1.tolist()
        Y_pred_proba_2 = Y_pred_proba_2.tolist()

        # 此处加入修改后的精度度量方法
        Y_pred = []
        for i in range(len(validation_y)):
            if Y_pred_proba_1[i].index(max(Y_pred_proba_1[i])) == Y_pred_proba_2[i].index(max(Y_pred_proba_2[i])):
                Y_pred.append(label_list[Y_pred_proba_1[i].index(max(Y_pred_proba_1[i]))])
            else:
                if max(Y_pred_proba_1[i]) >= max(Y_pred_proba_2[i]):
                    Y_pred.append(label_list[Y_pred_proba_1[i].index(max(Y_pred_proba_1[i]))])
                else:
                    Y_pred.append(label_list[Y_pred_proba_2[i].index(max(Y_pred_proba_2[i]))])

        Y_pred_1 = [label_list[i.index(max(i))] for i in Y_pred_proba_1]
        Y_pred_2 = [label_list[i.index(max(i))] for i in Y_pred_proba_2]
        # print('Y_pred_1 = ', Y_pred_1, 'Y_pred_2 = ', Y_pred_2)
        score_1 = accuracy_score(validation_y, Y_pred_1)
        score_2 = accuracy_score(validation_y, Y_pred_2)
        score = accuracy_score(validation_y, Y_pred)
        # print('score_1 = ', score_1, 'score_2 = ', score_2)

        Y_pred_proba_1_o = svm_classifier_1.predict_proba(X_test_1)
        Y_pred_proba_2_o = svm_classifier_2.predict_proba(X_test_2)
        Y_pred_proba_1_o = Y_pred_proba_1_o.tolist()
        Y_pred_proba_2_o = Y_pred_proba_2_o.tolist()

        return Y_pred_proba_1_o, Y_pred_proba_2_o, score_1, score_2, score


if __name__ == '__main__':
    algorithms = Algorithms()
    model = algorithms.make_model()
    for i in range(1, 9):
        feature = algorithms.get_VGG_Features(model, '../image/source/001.ak47/001_000' + str(1) + '.jpg')
        print(i, len(feature), feature)