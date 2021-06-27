# -*- coding:utf-8 -*-
u'''
Created on 2021年4月19日
@author: xianyu
@description：主程序，训练流程实现
'''
__author__ = 'xianyu'
__version__ = '1.0.0'
__company__ = u'STDU'
__updated__ = '2021-05-28'

import sys
import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前程序文件的目录
# sys.path.append(BASE_DIR) #添加环境变量

from Algorithms import *
from Utils import *
from Pretreatment import *

import matplotlib.pyplot as plt
import time

# 创建数据集部分
dataSet_name = './data_set/caltech_vgg_data_l2.mat'  # 数据集输入
ratio_marked_num = 0.05  # 数据集划分中，有标签这样本数量占除验证集之外的样本数量的比例
ratio_validation = 0.5   # 数据集划分中，验证集样本数量占总样本数量的比例
same_index = []
stop_ratio = 0.5  # 测试集中训练停止比例。目前迭代训练的终止条件是，添加超过一半的无标记样本参与训练
ratio_BvSB = 0.05
threshold_BvSB = 0.7
threshold_Weight = 100  # 参与训练次数的权重。大于该值的样本不参与训练
batch_size = 198

# 存储设置
dir = './result_storage/caltech/'
set_val = '_11vgg_' + str(ratio_marked_num) + '_' + str(round(1-ratio_marked_num, 2)) + '_' + \
          str(ratio_validation) + '_C1000'

str_title = str(ratio_marked_num) + '_' + str(round(1-ratio_marked_num, 2)) + '_' + \
          str(ratio_validation)

# 配置部分
isRSM = True
isBvSB = True
RSM_len = 900
isAll4Train = True


if __name__ == '__main__':

    # 记录程序开始运行的时间
    start_time = time.time()

    # 导入所需的工具和算法包
    utils = Utils()
    algorithms = Algorithms()
    pretreatment = Pretreatment()

    # 初始化数据集及权重，返回有编辑样本和无标记样本的数量
    data, target = utils.init_DataSet(dataSet_name)
    num_img_marked, num_img_unmarked, validation_x, validation_y = \
        utils.split_Data(ratio_marked_num, ratio_validation, data, target, isRSM=isRSM, RSM_len=RSM_len)

    print('len(validation_x[0]) = ', len(validation_x[0]))

    # 程序运行前的准备工作
    num_unmarked_img_orig = num_img_unmarked  # 记录unmarked文件夹下原始的图片数量
    num_loop = 1  # 记录训练的论数
    unchanged_times = 0  # 记录训练结果不变的次数，由此表征训练结果是否达到稳定
    threshold_satble = 10  # 训练结果不变的次数超过threshold_satble次，则证明训练已达稳定，训练终止
    accuracy_score_buf_1 = []  # 记录分类器1的精度
    accuracy_score_buf_2 = []  # 记录分类器2的精度
    accuracy_score_buf = []  # 记录两个分类器的度量精度
    moving_ratio = []

    label_list = utils.get_Label_List(dataSet_name)
    print('label_list = ', label_list)

    if isAll4Train:
        batch_size = num_img_marked

    while True:

        print('------------------------------第' + str(num_loop) + '轮训练--------------------------------')

        batch_size = batch_size + len(same_index)
        print('batch_size+len(same_index) = ', batch_size)

        # 获取数据集中的样本
        X_Train_1, X_Test_1, Y_Train_1, Y_Test_1, X_Train_2, X_Test_2, Y_Train_2, Y_Test_2\
            = utils.get_train_test_data(batch_size)

        if len(X_Train_1) == len(X_Test_1) == len(Y_Train_1) == len(Y_Test_1) == len(X_Train_2) == len(X_Test_2) == \
                len(Y_Train_2) == len(Y_Test_2) == 0:
            break

        # SVM训练
        Y_pred_proba_1, Y_pred_proba_2, score_1, score_2, score = algorithms.SVM_Training_And_Testing(
            X_Train_1, X_Train_2, Y_Train_1, Y_Train_2, X_Test_1, X_Test_2, Y_Test_1, Y_Test_2, validation_x,
            validation_y, label_list)
        print('score_1 = ', score_1, 'score_2 = ', score_2, 'score = ', score)

        # 记录两个分类器的精度
        accuracy_score_buf_1.append(score_1)
        accuracy_score_buf_2.append(score_2)
        accuracy_score_buf.append(score)

        # BvSB算法输出前ratio的候选值
        diff_B_SB_1, diff_B_SB_2 = algorithms.BvSB(Y_pred_proba_1, Y_pred_proba_2, ratio_BvSB, threshold_BvSB,
                                                   isBvSB=isBvSB)

        print('diff_B_SB_1', diff_B_SB_1)
        print('diff_B_SB_2', diff_B_SB_2)

        # 判断两个预测结果是否相同且是否正确
        diff_B_SB_1_index = [i[0] for i in diff_B_SB_1]  # 获取BvSB得到的样本下标
        diff_B_SB_2_index = [i[0] for i in diff_B_SB_2]  # 获取BvSB得到的样本下标
        same_index = [x for x in diff_B_SB_1_index if x in diff_B_SB_2_index]  # 寻找相同的元素
        print('same_index', same_index)

        # 移动数据集
        utils.update_DataSet(same_index, threshold_Weight)

        # 判断是否还有待移动的图片，若超过10次没有图片可以移动，则证明训练已达稳定，训练终止
        if len(same_index) == 0:
            unchanged_times = unchanged_times + 1
        else:
            unchanged_times = 0
        # 训练终止的条件:训练达到稳定
        if unchanged_times >= threshold_satble:
            print('训练已达到稳定，结束训练！')
            break

        # 训练终止的条件：超过50%的无标签样本被移动
        if (num_unmarked_img_orig - len(Y_Test_1)) / num_unmarked_img_orig > stop_ratio:
            print('超过'+str(stop_ratio*100)+'%的无标签样本被移动，结束训练！')
            break

        num_loop = num_loop + 1
        moving_ratio.append((num_unmarked_img_orig - len(Y_Test_1)) / num_unmarked_img_orig)
        print('{:.2%}的无标签样本被移动'.format((num_unmarked_img_orig - len(Y_Test_1)) / num_unmarked_img_orig))

        # 打印程序运行时间
        end_time = time.time()
        dtime = end_time - start_time
        print("程序运行时间：%.8s s" % dtime)  # 显示到微秒
        print('SVM1训练精度的最大值为：', round(max(accuracy_score_buf_1), 4))
        print('SVM2训练精度的最大值为：', round(max(accuracy_score_buf_2), 4))
        print('协同训练精度的最大值为：', round(max(accuracy_score_buf), 4))
        print()

    #     # 绘制精度变化曲线
    # plt.figure(0, figsize=(14, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(accuracy_score_buf_1)
    # plt.title('classifier 1')
    # plt.xlabel('caltech')
    # plt.ylabel('accuracy')
    # plt.title(str_title)
    #
    # # plt.figure(1)
    # plt.subplot(1, 2, 2)
    # plt.plot(accuracy_score_buf_2)
    # plt.title('classifier 2')
    # plt.xlabel('caltech')
    # plt.ylabel('accuracy')
    # plt.title(str_title)
    #
    # plt.savefig(dir + 'classifier' + set_val + '.png')

    # 绘制无标签样本被移动比例图
    plt.figure(1, figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_score_buf)
    plt.title('classifier')
    plt.xlabel('caltech')
    plt.ylabel('accuracy')
    plt.title(str_title)

    plt.subplot(1, 2, 2)
    plt.plot(moving_ratio)
    plt.title('moving ratio')
    plt.xlabel('caltech')
    plt.ylabel('ratio')
    plt.title(str_title)

    plt.savefig(dir + 'moving_ratio' + set_val + '.png')

    # 打印程序运行时间
    end_time = time.time()
    dtime = end_time - start_time
    print("程序运行时间：%.8s s" % dtime)  # 显示到微秒
    print('SVM1训练精度的最大值为：', round(max(accuracy_score_buf_1), 4))
    print('SVM2训练精度的最大值为：', round(max(accuracy_score_buf_2), 4))
    print('协同训练精度的最大值为：', round(max(accuracy_score_buf), 4))

    # 以文档的方式存储数据
    filename = dir + 'record' + set_val + '.txt'
    with open(filename, 'w') as file_object:
        file_object.write(str(len(accuracy_score_buf)) + '\n')
        file_object.write(str(accuracy_score_buf_1) + '\n')
        file_object.write(str(accuracy_score_buf_2) + '\n')
        file_object.write(str(accuracy_score_buf) + '\n')
        file_object.write(str(moving_ratio) + '\n')
        file_object.write(str(dtime) + '\n')
        file_object.write(str(max(accuracy_score_buf)) + '\n')



    plt.show()









