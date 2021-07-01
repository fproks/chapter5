import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from coTraining import COTraining
from sklearn.decomposition import PCA

_tmp = sio.loadmat("./data_set/vggdata.mat")
data = _tmp["data"]
stand = _tmp["target"].flatten()
stand = LabelEncoder().fit_transform(stand)

pca=PCA(n_components=300)
data=pca.fit_transform(data)


train_x, test_x, train_y, test_y = train_test_split(data, stand, test_size=0.95,random_state=42)

final_test_x = np.copy(test_x)
final_test_y = np.copy(test_y)
from collections import Counter

ttt = Counter(stand)

co = COTraining(train_x, train_y, test_x, test_y,RSMSize=0.8,bvsbFilter=0.7,iterPrecent=0.2)
co.fit(final_test_x, final_test_y)
print(co.score(final_test_x, final_test_y))
