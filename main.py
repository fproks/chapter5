import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from coTraining import COTraining
_tmp = sio.loadmat("./data_set/vggdata.mat")
data = _tmp["data"]
stand = _tmp["target"].flatten()

train_x,test_x,train_y,test_y=train_test_split(data,stand,test_size=0.8)

final_test_x=np.copy(test_x)
final_test_y=np.copy(test_y)
from  collections import Counter

ttt=Counter(stand)


co=COTraining(train_x,train_y,test_x,test_y)
co.fit(final_test_x,final_test_y)
print(co.score(final_test_x,final_test_y))







