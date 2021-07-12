import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from coTraining import COTraining
from Pretreatment import Pretreatment

_tmp=sio.loadmat(r"E:\data_chapter5\cifar_vgg_data_l2.mat")
data = _tmp["data"]
stand = _tmp["target"].flatten()
stand = LabelEncoder().fit_transform(stand)

data=Pretreatment.dimensionWithPCA(data,n_components=300)

train_x, test_x, train_y, test_y = train_test_split(data, stand, test_size=0.95, random_state=42)
final_test_x = np.copy(test_x)
final_test_y = np.copy(test_y)

co = COTraining(train_x, train_y, test_x, test_y, needRSM=False, bvsbFilter=0.7, iterPrecent=0.1, needBVSB=True)
co.fit(final_test_x, final_test_y)
tmp = co.score(final_test_x, final_test_y)
print(tmp)
print("------------------------------------------------------------------------------------------")
print("---------------------------------CO-VGG-BVSB------------------------------------------------------")
