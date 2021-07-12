import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from coTraining import COTraining
from sklearn.decomposition import PCA

#_tmp = sio.loadmat(r"E:\data_chapter5\mnist_vgg_data_l2.mat")
_tmp=sio.loadmat(r"E:\data_chapter5\cifar_hog_data_l2.mat")
data = _tmp["data"]
stand = _tmp["target"].flatten()
stand = LabelEncoder().fit_transform(stand)

pca = PCA(n_components=300)  #pca降维后数据维度
data = pca.fit_transform(data)

label_size=0.05  #已标记样本数量
train_x, test_x, train_y, test_y = train_test_split(data, stand, test_size=1-label_size, random_state=42)

final_test_x = np.copy(test_x)
final_test_y = np.copy(test_y)
from collections import Counter

ttt = Counter(stand)

'''
    当不需要RSM时，needRSM=False
    当不需要BVSB时，needBVSB=False
    bvsbFilter 设定大于多少的bvsb 才能被筛选
    RSMSize 设定 每个训练器随机子空间的维度大小
    iterPrecent 设定每次最多可以加入训练的数据有多少
    maxRemainPercent 设定最多可以剩余多少数据
'''

co = COTraining(train_x, train_y, test_x, test_y, RSMSize=0.8, bvsbFilter=0.7, iterPrecent=0.1, needRSM=True,
                needBVSB=True,maxRemainPercent=0.5)
co.fit(final_test_x, final_test_y)
print(co.score(final_test_x, final_test_y))
print("-----------------------------OVER--------------------------------------------")
