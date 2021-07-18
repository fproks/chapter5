import random

import numpy as np
import os
import cv2
from typing import Tuple
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from torch.autograd import Variable
from sklearn import preprocessing
from scipy.cluster.vq import *
import scipy.io as sio
import gc

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
                try:
                    _data = cv2.resize(_data, (image_size, image_size))
                except Exception as e:
                    print(str(e))
                    print(f"error image is {img}")
                    raise e
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
                if file_path.endswith(r".jpg"):
                    result.append(file_path)
                else:
                    print(f"not image delete it: {file_path}")
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
        i = 0
        for img in imageList:
            print(f"VGG提取第{i}张图像特征")
            tensor = img_to_tensor(img)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
            result = model(Variable(tensor))
            result_npy = result.data.cpu().numpy()
            resultList.append(result_npy[0].tolist())
            print(f"VGG提取第{i}张图像完成")
            i += 1
        return resultList

    @staticmethod
    def featureExtractionSIFT(imageList: list, targetList: np.ndarray) -> Tuple[list, list]:
        des_list = []
        target_modified = []
        numWords = 1000
        i = 0
        # SIFT特征计算
        sift = cv2.xfeatures2d.SIFT_create()
        print('正在提取特征SIFT！')
        for img in imageList:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kpts, des = sift.detectAndCompute(gray, None)
                # kps_list.append(kpts)
                if des is not None:
                    des_list.append(des)
                    target_modified.append(targetList[i])
            except Exception as e:
                print(str(e))
            i = i + 1
            if i % 100 == 0:
                print('已提取' + str(i) + '张')
        print(f'去除掉没有sift特征的数据后剩余数据个数为{len(target_modified)}')

        del imageList, targetList, sift
        gc.collect()
        # Stack all the descriptors vertically in a numpy array
        # image_path为图片路径，descriptor为对应图片的特征
        # 将所有特征纵向堆叠起来,每行当做一个特征词
        print('将所有特征纵向堆叠起来,每行当做一个特征词')
        # print(descriptors)
        descriptors = np.vstack(des_list)
        # vstack对矩阵进行拼接，将所有的特征word拼接到一起
        # 对特征词使用k-menas算法进行聚类
        print(f"K-means聚类类中心个数{numWords},数据量{descriptors.shape[0]}")
        # 最后输出的结果其实是两维的,第一维是聚类中心,第二维是损失distortion
        voc, variance = kmeans(descriptors, numWords, iter=1)
        del descriptors, variance
        gc.collect()
        # 初始化一个bag of word矩阵，每行表示一副图像，每列表示一个视觉词，下面统计每副图像中视觉词的个数
        print('进行词袋计算')
        im_features = np.zeros((len(des_list), numWords), "float32")
        for i in range(len(des_list)):
            # 计算每副图片的所有特征向量和voc中每个特征word的距离，返回为匹配上的word
            # descriptor = des_list[i][1]
            # if descriptor != None:
            # 根据聚类中心将所有数据进行分类des_list[i][1]为数据, voc则是kmeans产生的聚类中心.
            # vq输出有两个:一是各个数据属于哪一类的label,二是distortion
            words, distance = vq(des_list[i], voc)
            for w in words:
                im_features[i][w] += 1
            if i % 100 == 0:
                print('已计算' + str(i) + '张')
        del des_list, voc
        gc.collect()
        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0 * len(target_modified) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

        # L2归一化
        im_features = im_features * idf
        im_features = preprocessing.normalize(im_features, norm='l2')
        print('cal_bow 结束了')
        print(im_features)
        return im_features, target_modified

    @staticmethod
    def featureExtractionHOG(imageList: list) -> list:
        resultList = []
        i = 0
        imageSize = (128, 128)  # 重置所有图像数据的大小为128
        cell_size = (16, 16)
        num_cells_per_block = (1, 1)
        block_size = (num_cells_per_block[0] * cell_size[0],
                      num_cells_per_block[1] * cell_size[1])
        # Calculate the number of cells that fitWithRSMAndBVSB in our image in the x and y directions
        x_cells = imageSize[0] // cell_size[0]
        y_cells = imageSize[1] // cell_size[1]
        h_stride = 1
        v_stride = 1

        # Block Stride in pixels (horizantal, vertical). Must be an integer multiple of Cell Size
        block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)

        # Number of gradient orientation bins
        num_bins = 9

        win_size = (x_cells * cell_size[0], y_cells * cell_size[1])
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        for img in imageList:
            print(f"HOG提取第{i}张图像特征")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, imageSize)
            hog_descriptor = hog.compute(img)
            resultList.append(hog_descriptor.reshape((-1,)))
            print(f"HOG提取第{i}张图像完成")
            i += 1
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
        assert Dimension >= PresetD
        if Dimension == PresetD:
            return np.arange(Dimension)
        return np.sort(np.random.permutation(np.array(range(Dimension)))[0:PresetD])

    @staticmethod
    def dimensionWithPCA(data, n_components=300):
        pca = PCA(n_components=300)
        return pca.fit_transform(data)

    @staticmethod
    def removeDataAsToLess(data: np.ndarray, target: np.ndarray, lessFilter=10):
        deleteIndexArray = []
        nuq = np.unique(target)
        for i in nuq:
            if np.sum(target == i) < lessFilter:
                tmp = np.argwhere(target == i)
                deleteIndexArray.extend(tmp.tolist())
        data = np.delete(data, deleteIndexArray, axis=0)
        target = np.delete(target, deleteIndexArray).flatten().astype(int)
        return data, target


if __name__ == '__main__':
    # [a, b] = Pretreatment.readGrayImageToDATA(r"E:\data_chapter5\256_ObjectCategories")
    # print(len(a))
    # print(len(b))
    # np.savez("caltech256Data.npz",data=a,target=b)
    d = np.load("caltech256Data.npz")
    a = d['data']
    target = d['target']
    sizes = len(target)
    # sio.savemat("caltech256Data.mat",{'data':a,'target':b})
    # data = Pretreatment.featureExtractionVGG(a)
    # print('VGG特征提取完成，进行保存')
    # data = np.array(data)
    # target = np.array(b)
    # np.savez('caltech256-vgg.npz', data=data, target=target)
    # sio.savemat('caltech256-vgg.mat',{'data':data,'target':target})
    print("开始提取HOG特征")
    data = Pretreatment.featureExtractionHOG(a)
    data = np.array(data)
    np.savez('caltech256-HOG.npz', data=data, target=target)
    # sio.savemat('caltech256-HOG.mat',{'data':data,'target':target})
    # print("HOG特征提取结束，开始提取SIFT特征")
    # for i in range(12):
    #     start = i * 3000
    #     end = min((i + 1) * 3000, sizes)
    #     print(f"开始提取第{start}到第{end}个数据的SIFT特征")
    #     if start < sizes:
    #         data = a[start:end]
    #         tmp_target = target[start:end]
    #         [data, tmp_target] = Pretreatment.featureExtractionSIFT(data, tmp_target)
    #         gc.collect()
    #         np.savez(f'caltech256-SIFT-{i}.npz', data=data, target=tmp_target)
    #         print(f'第{start}个数据到第{end}个数据 SIFT 特征提取结束,共获取数据{len(tmp_target)}个')
    #     else:
    #         print(f"所有数据特征都已提取结束，SIFT算法结束,start={start}")
    #         break
    # sio.savemat('caltech256-SIFT.mat',{'data':data,'target':target})
    print("---------------------FINISH----------------------------------")
