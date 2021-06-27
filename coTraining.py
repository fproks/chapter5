import numpy
from sklearn.svm import SVC
import numpy as np
import operator


class COTraining(object):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, iter_x: np.ndarray, iter_y: np.ndarray):
        assert train_x.ndim == 2
        assert train_y.ndim == 1
        assert iter_x.ndim == 2
        assert iter_y.ndim == 1
        assert train_x.shape[0] == train_y.shape[0]
        assert iter_x.shape[0] == iter_y.shape[0]
        assert train_x.shape[1] == iter_x.shape[1]
        self.firstClassifier = SVC(probability=True)
        self.secondClassifier = SVC(probability=True)
        self.train_x = train_x
        self.train_y = train_y
        self.iter_x = iter_x
        self.iter_y = iter_y
        self.isIter = True
        self.singleIterMaxLength = len(self.iter_y) * 0.05
        self.allIterMaxLength=len(self.iter_y)*0.5

    def fit(self,test_x,test_y):
        while self.isIter:
            self.firstClassifier.fit(self.train_x, self.train_y)
            self.secondClassifier.fit(self.train_x, self.train_y)
            proba1 = self.firstClassifier.predict_proba(self.iter_x)
            proba2 = self.secondClassifier.predict_proba(self.iter_x)
            iterIndex = self.getIterDataIndex(proba1, proba2)
            if len(iterIndex) == 0:
                self.isIter = False
            if len(self.iter_y)<self.allIterMaxLength:
                self.isIter=False
            self.resetTrainAndIterData(iterIndex)
            print(self.score(test_x,test_y))

    '''
        获取类别相同的数据的索引
    '''

    def _getSameCategoryIndex(self, proba1: np.ndarray, proba2: np.ndarray) -> np.ndarray:
        category1 = np.argmax(proba1, axis=1)
        category2 = np.argmax(proba2, axis=1)
        sameIndex = np.argwhere(category1 == category2).flatten().astype(int)
        return sameIndex

    '''
    获取BVSB中大于过滤值且属于前10%的数据
    '''

    def _bvsbIndex(self, proba1, filter=0.7):
        assert proba1.shape[1] >= 2
        tmp = np.sort(proba1, axis=1)
        _bvsb = tmp[:, -1] = tmp[:, -2]
        resIndex = []
        for i in range(len(proba1)):
            if _bvsb[i] > filter:
                resIndex.append(i)
        maxSize = min(self.singleIterMaxLength * 2, len(resIndex)) * -1
        resultIndex = np.argsort(_bvsb[resIndex])[maxSize:]
        return resultIndex

    # 获取下一次迭代需要添加的数据的索引
    def getIterDataIndex(self, proba1, proba2):
        sameIndex = self._getSameCategoryIndex(proba1, proba2)
        _bvsb1_index = self._bvsbIndex(proba1[sameIndex])
        _bvsb2_index = self._bvsbIndex(proba2[sameIndex])
        tmp = np.argwhere(_bvsb2_index == _bvsb1_index).flatten().astype(int)
        data_index = sameIndex[_bvsb1_index[tmp]]
        return data_index

    # 将下一次需要迭代训练的数据添加到训练集中
    def resetTrainAndIterData(self, iterDataIndex):
        prepareData = self.iter_x[iterDataIndex]
        prepareY = self.iter_y[iterDataIndex]
        self.train_x = numpy.append(self.train_x, prepareData, axis=0)
        self.train_y = numpy.append(self.train_y, prepareY, axis=0)
        self.iter_x = np.delete(self.iter_x, iterDataIndex, axis=0)
        self.iter_y = np.delete(self.iter_y, iterDataIndex, axis=0)

    def predict(self, test_x):
        proba1 = self.firstClassifier.predict_proba(test_x)
        proba2 = self.firstClassifier.predict_proba(test_x)
        res1 = np.argmax(proba1, axis=1)
        res2 = np.argmax(proba2, axis=1)
        sameIndex = np.argwhere(res1 == res2).flatten().astype(int)
        result = np.zeros((test_x.shape[0],))
        result[sameIndex] = res1[sameIndex] + 1
        if len(sameIndex)<test_x.shape[0]:
            notSameIndex = np.delete(np.arange(len(res1)), sameIndex, axis=0)
            maxProba1=proba1[:, res1[notSameIndex]]
            maxProba2=proba2[:, res2[notSameIndex]]
            noSamePredict = self._getMaxProbaIndex(maxProba1, maxProba2)
            result[notSameIndex] = noSamePredict + 1
        return result

    def _getMaxProbaIndex(self, maxProba1, maxProba2, index1, index2):
        res = np.zeros((len(maxProba1),))
        for i in range(len(maxProba1)):
            if maxProba1[i] > maxProba2[i]:
                res[i] = index1[i]
            else:
                res[i] = index2[i]
        return res

    def score(self, test_x, test_y):
        predict = self.predict(test_x)
        from sklearn.metrics import accuracy_score
        return accuracy_score(test_y, predict)
