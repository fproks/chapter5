import numpy as np
import scipy.io as sio
from  sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from coTraining import COTraining
_tmp=sio.loadmat(r"E:\data_chapter5\cifar_vgg_data_l2.mat")
data = _tmp["data"]
stand = _tmp["target"].flatten()
stand=LabelEncoder().fit_transform(stand)
train_x,test_x,train_y,test_y=train_test_split(data,stand,test_size=0.95,random_state=42)

svm=SVC()
svm.fit(train_x,train_y)
print(svm.score(test_x,test_y))
print("------------------------------------------------------------------------------------------")
print("---------------------------------SVM------------------------------------------------------")






