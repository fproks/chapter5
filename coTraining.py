import numpy
from sklearn.svm import SVC
import numpy as np
import operator
from Pretreatment import Pretreatment



class COTraining(object):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, iter_x: np.ndarray, iter_y: np.ndarray,RSMSize=0.9):

        assert train_x.ndim == 2
        assert train_y.ndim == 1
        assert iter_x.ndim == 2
        assert iter_y.ndim == 1
        assert train_x.shape[0] == train_y.shape[0]
        assert iter_x.shape[0] == iter_y.shape[0]
        assert train_x.shape[1] == iter_x.shape[1]
        self.originDataDim=train_x.shape[1]
        self.firstClassifier = SVC(probability=True)
        self.secondClassifier = SVC(probability=True)
        self.first_train_x = np.copy(train_x)
        self.second_train_x=np.copy(train_x)
        self.train_y = train_y
        self.first_iter_x = np.copy(iter_x)
        self.second_iter_x=np.copy(iter_x)
        self.iter_y = iter_y
        self.isIter = True
        self.singleIterMaxLength = len(self.iter_y) * 0.05
        self.allIterMaxLength = len(self.iter_y) * 0.5
        assert  RSMSize>0
        assert RSMSize<self.originDataDim
        if RSMSize<1:
            self.RSMSize=int(train_x.shape[1]*RSMSize)
        else:
            self.RSMSize=int(RSMSize)
        self.firstRSMIndex=np.array(range(self.originDataDim))
        self.secondRSMIndex=np.array(range(self.originDataDim))

    def createRSM(self):
        self.firstRSMIndex=Pretreatment.createRSMIndex(self.originDataDim,self.RSMSize)
        self.secondRSMIndex=Pretreatment.createRSMIndex(self.originDataDim,self.RSMSize)
        self.first_train_x=self.first_train_x[:,self.firstRSMIndex]
        self.second_train_x=self.second_train_x[:,self.secondRSMIndex]
        self.first_iter_x=self.first_iter_x[:,self.firstRSMIndex]
        self.second_iter_x=self.second_iter_x[:,self.secondRSMIndex]

    def fit(self, test_x, test_y):
        while self.isIter:
            self.firstClassifier.fit(self.first_train_x, self.train_y)
            self.secondClassifier.fit(self.second_train_x, self.train_y)
            proba1 = self.firstClassifier.predict_proba(self.first_iter_x)
            proba2 = self.secondClassifier.predict_proba(self.second_iter_x)
            iterIndex = self.getIterDataIndex(proba1, proba2)
            if len(iterIndex) == 0:
                self.isIter = False
            if len(self.iter_y) < self.allIterMaxLength:
                self.isIter = False
            self.resetTrainAndIterData(iterIndex)
            print(self.score(test_x, test_y))

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
        _bvsb = tmp[:, -1] - tmp[:, -2]
        resIndex = []
        for i in range(len(proba1)):
            if _bvsb[i] > filter:
                resIndex.append(i)
        maxSize = int(min(self.singleIterMaxLength * 2, len(resIndex)) * -1)
        #满足的值的前maxSize个索引
        index = np.argsort(_bvsb[resIndex])[maxSize:]
        #最终的索引
        return np.array(resIndex)[index]

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
        prepareY = self.iter_y[iterDataIndex]
        self.first_train_x = numpy.append(self.first_train_x, self.first_iter_x[iterDataIndex], axis=0)
        self.second_train_x=numpy.append(self.second_train_x,self.second_iter_x[iterDataIndex],axis=0)
        self.train_y = numpy.append(self.train_y, prepareY, axis=0)
        self.first_iter_x = np.delete(self.first_iter_x, iterDataIndex, axis=0)
        self.second_iter_x=np.delete(self.second_iter_x,iterDataIndex,axis=0)
        self.iter_y = np.delete(self.iter_y, iterDataIndex, axis=0)

    def predict(self, test_x):
        first_test_x=test_x[:,self.firstRSMIndex]
        second_test_x=test_x[:,self.secondRSMIndex]
        proba1 = self.firstClassifier.predict_proba(first_test_x)
        proba2 = self.secondClassifier.predict_proba(second_test_x)
        res1 = np.argmax(proba1, axis=1)
        res2 = np.argmax(proba2, axis=1)
        sameIndex = np.argwhere(res1 == res2).flatten().astype(int)
        result = np.zeros((test_x.shape[0],))
        result[sameIndex] = res1[sameIndex] + 1
        if len(sameIndex) < test_x.shape[0]:
            notSameIndex = np.delete(np.arange(len(res1)), sameIndex)
            first_max_p=np.sort(proba1,axis=1)[notSameIndex,-1]
            first_max_i=np.argsort(proba1,axis=1)[notSameIndex,-1]
            second_max_p=np.sort(proba2,axis=1)[notSameIndex,-1]
            second_max_i=np.sort(proba2,axis=1)[notSameIndex,-1]
            notSameC=[]
            for i in range(len(notSameIndex)):
                if first_max_p[i]>second_max_p[i]:
                    notSameC.append(first_max_i[i])
                else:
                    notSameC.append(second_max_i[i])
            result[notSameIndex] = np.array(notSameC) + 1
        return result

    def score(self, test_x, test_y):
        predict = self.predict(test_x)
        from sklearn.metrics import accuracy_score
        return accuracy_score(test_y, predict)
