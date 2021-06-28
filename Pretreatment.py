import random

import numpy as np
import os
import cv2
from typing import Tuple
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn import preprocessing
from scipy.cluster.vq import *
import scipy.io as sio

'''
https://bbs.csdn.net/topics/392551610
需要安装cuda ，安装带cuda 的pytorch
import torch
print(torch.cuda.is_available())
这个代码可以看cuda 是否可用
conda 安装带CUDA的pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

没有CUDA：
conda install pytorch torchvision torchaudio cpuonly -c pytorch
'''
IMAGE_SIZE = 244


class Pretreatment():
    @staticmethod
    def readGrayImageToDATA(rootPath: str, image_size=IMAGE_SIZE) -> Tuple[list, list]:
        imageList = Pretreatment._getfileList(rootPath)
        dataList = []
        targetList = []
        for img in imageList:
            _data = cv2.imread(img)
            if image_size is not None:
                _data = cv2.resize(_data, (image_size, image_size))
            dataList.append(_data)
            filename = os.path.basename(img)
            targetList.append(int(filename.split("_")[0]))
        return dataList, targetList

    @staticmethod
    def _getfileList(rootPath) -> list:
        if os.path.isabs(rootPath) is False:
            rootPath = os.path.abspath(rootPath)
        assert os.path.exists(rootPath)
        if os.path.isdir(rootPath) is False:
            return []
        result = []
        for file in os.listdir(rootPath):
            file_path = os.path.join(rootPath, file)
            if os.path.isfile(file_path):
                result.append(file_path)
            else:
                result.extend(Pretreatment._getfileList(file_path))
        return result

    @staticmethod
    def make_model():
        model = models.vgg16(pretrained=False)  # 其实就是定位到第28层，对照着上面的key看就可以理解
        if torch.cuda.is_available():
            pre = torch.load(r'./source/vgg16-397923af.pth')
        else:
            pre = torch.load(r'./source/vgg16-397923af.pth', map_location=torch.device('cpu'))
        model.load_state_dict(pre)
        model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        if torch.cuda.is_available():
            model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
        return model

    @staticmethod
    def featureExtractionVGG(imageList: list) -> list:
        model = Pretreatment.make_model()
        img_to_tensor = transforms.ToTensor()
        resultList = []
        for img in imageList:
            tensor = img_to_tensor(img)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy()
            resultList.append(result_npy[0].tolist())
        return resultList

    @staticmethod
    def featureExtractionSIFT(imageList: list) -> list:
        resultList = []
        des_list = []
        kps_list = []
        numWords = 1000
        # SIFT特征计算
        sift = cv2.xfeatures2d.SIFT_create()
        for img in imageList:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kpts, des = sift.detectAndCompute(gray, None)

            kps_list.append(kpts)
            des_list.append((img, des))

        # Stack all the descriptors vertically in a numpy array
        # image_path为图片路径，descriptor为对应图片的特征
        # 将所有特征纵向堆叠起来,每行当做一个特征词
        descriptors = des_list[0][1]
        for image_path, descriptor in des_list[1:]:
            # vstack对矩阵进行拼接，将所有的特征word拼接到一起
            # print descriptor.shape, descriptors.shape
            # if descriptor != None:
            descriptors = np.vstack((descriptors, descriptor))

        # 对特征词使用k-menas算法进行聚类
        print("Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
        # "Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0])
        # 最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion
        voc, variance = kmeans(descriptors, numWords, iter=1)

        # 初始化一个bag of word矩阵，每行表示一副图像，每列表示一个视觉词，下面统计每副图像中视觉词的个数
        im_features = np.zeros((len(imageList), numWords), "float32")
        for i in range(len(imageList)):
            # 计算每副图片的所有特征向量和voc中每个特征word的距离，返回为匹配上的word
            descriptor = des_list[i][1]
            # if descriptor != None:
            # 根据聚类中心将所有数据进行分类des_list[i][1]为数据, voc则是kmeans产生的聚类中心.
            # vq输出有两个:一是各个数据属于哪一类的label,二是distortion
            words, distance = vq(des_list[i][1], voc)
            for w in words:
                im_features[i][w] += 1

        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0 * len(imageList) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

        # L2归一化
        im_features = im_features * idf
        im_features = preprocessing.normalize(im_features, norm='l2')
        # im_features = preprocessing.transform(im_features)
        print('cal_bow 结束了')
        print(im_features)
        return im_features

    @staticmethod
    def featureExtractionHOG(imageList: list) -> list:
        resultList = []
        des_list = []
        kps_list = []
        numWords = 1000

        for img in imageList:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cell_size = (6, 6)
            num_cells_per_block = (2, 2)
            block_size = (num_cells_per_block[0] * cell_size[0],
                          num_cells_per_block[1] * cell_size[1])

            # Calculate the number of cells that fit in our image in the x and y directions
            x_cells = gray_image.shape[1] // cell_size[0]
            y_cells = gray_image.shape[0] // cell_size[1]

            h_stride = 1
            v_stride = 1

            # Block Stride in pixels (horizantal, vertical). Must be an integer multiple of Cell Size
            block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)

            # Number of gradient orientation bins
            num_bins = 9

            win_size = (x_cells * cell_size[0], y_cells * cell_size[1])

            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

            # Compute the HOG Descriptor for the gray scale image
            hog_descriptor = hog.compute(gray_image)
            resultList.append(hog_descriptor.reshape((-1,)))

        return resultList

    # 整体RSM，防止对每一个数据RSM之后，每个数据RMS获取的数据维度不同。
    @staticmethod
    def reduceFeatureWithRSM(data: np.ndarray, featureNumber: int) -> np.ndarray:
        assert data.ndim == 2
        assert data.shape[1] > featureNumber
        index = np.arange(data.shape[1])
        np.random.shuffle(index)
        delete_index = index[0:data.shape[1] - featureNumber]
        return np.delete(data, delete_index, axis=1)

    @staticmethod
    def createRSMIndex(Dimension, PresetD):
        assert Dimension > PresetD
        return np.sort(np.random.permutation(np.array(range(Dimension)))[0:PresetD])



if __name__ == '__main__':
    [a, b] = Pretreatment.readGrayImageToDATA(r"F:\dataset4\flowers17\flowers17\test", image_size=500)
    print(len(a))
    print(len(b))

# print('start extract features by VGG16')
# [a, b] = Pretreatment.readGrayImageToDATA("../image/source")
# print('得到图像数据,开始特征提取')
# data = Pretreatment.featureExtractionVGG(a)
# print('特征提取完成，开始保存')
# data = np.array(data)
# target = np.array(b)
# sio.savemat('vggdata.mat', {'data': data, 'target': target})
# print("特征数据保存为vggdata.mat")
