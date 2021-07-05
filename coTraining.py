import numpy
from sklearn.svm import SVC
import numpy as np
import operator
from Pretreatment import Pretreatment
import time
from config import LOGGER


class COTraining(object):
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, iter_x: np.ndarray, iter_y: np.ndarray,
                 needRSM=True, RSMSize=0.9,
                 needBVSB=True,bvsbFilter=0.7,
                 iterPrecent=0.1,maxRemainPercent=0.5):
        assert train_x.ndim == 2
        assert train_y.ndim == 1
        assert iter_x.ndim == 2
        assert iter_y.ndim == 1
        assert train_x.shape[0] == train_y.shape[0]
        assert iter_x.shape[0] == iter_y.shape[0]
        assert train_x.shape[1] == iter_x.shape[1]
        self.originDataDim = train_x.shape[1]
        self.firstClassifier = SVC(probability=True)
        self.secondClassifier = SVC(probability=True)
        self.first_train_x = np.copy(train_x)
        self.second_train_x = np.copy(train_x)
        self.train_y = train_y
        self.first_iter_x = np.copy(iter_x)
        self.second_iter_x = np.copy(iter_x)
        self.iter_y = iter_y
        self.isIter = True
        self.firstRSMIndex = []
        self.secondRSMIndex = []
        self.singleIterMaxLength = int(len(self.iter_y) * iterPrecent)
        LOGGER.debug(f"单次最多加入训练集数据个数为{self.singleIterMaxLength}")
        self.allIterMaxLength = int(len(self.iter_y) * maxRemainPercent)
        LOGGER.debug(f'最多可以加入训练集的数据个数为{self.allIterMaxLength}')
        self.bvsbFilter = bvsbFilter
        LOGGER.debug(f"bvsbFilter is {bvsbFilter},bvsb 小于该数值的都会被过滤掉")
        self.needRSM = needRSM
        if self.needRSM:
            self.createRSM(RSMSize)
        else:
            LOGGER.debug(f"needRSMis {needRSM},将不会进行RSM")
        self.needBVSB =needBVSB

    def createRSM(self, RSMSize):
        assert RSMSize > 0
        assert RSMSize < self.originDataDim
        if RSMSize < 1:
            RSMSize = int(self.originDataDim * RSMSize)
        LOGGER.info(f"RSM 后数据的维度个数为:{RSMSize}")
        self.firstRSMIndex = Pretreatment.createRSMIndex(self.originDataDim, RSMSize)
        self.secondRSMIndex = Pretreatment.createRSMIndex(self.originDataDim, RSMSize)
        self.first_train_x = self.first_train_x[:, self.firstRSMIndex]
        self.second_train_x = self.second_train_x[:, self.secondRSMIndex]
        self.first_iter_x = self.first_iter_x[:, self.firstRSMIndex]
        self.second_iter_x = self.second_iter_x[:, self.secondRSMIndex]

    def fit(self, test_x, test_y):
        print("------------------------TRAIN----------------------------------")
        i = 1
        while self.isIter:
            start = time.time_ns()
            print(f'当前训练集合大小为{len(self.train_y)}')
            self.firstClassifier.fit(self.first_train_x, self.train_y)
            self.secondClassifier.fit(self.second_train_x, self.train_y)
            proba1 = self.firstClassifier.predict_proba(self.first_iter_x)
            proba2 = self.secondClassifier.predict_proba(self.second_iter_x)
            if len(self.iter_y) < self.allIterMaxLength:
                self.isIter = False
                LOGGER.warn(f"加入训练集的数据超过了所有无标记样本的一半，训练结束,不更新训练集")
                break
            iterIndex = self.getIterDataIndex(proba1, proba2)
            if len(iterIndex) == 0:
                self.isIter = False
                LOGGER.warn(f"获取的迭代训练数据为空,训练结束")
                break
            self.resetTrainAndIterData(iterIndex)
            print(f'第{i}次训练结束，共获取可迭代数据{len(iterIndex)}个')
            print(f'第{i}次训练结束,测试结果为{self.score(test_x, test_y):0.6f}')
            end = time.time_ns()
            print(f'本次训练共用时{(end - start) / 1000 / 1000 / 1000:0.2f}秒')
            i = i + 1

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
        LOGGER.debug(f"进行BVSB筛选，filter is {filter}")
        assert proba1.shape[1] >= 2
        tmp = np.sort(proba1, axis=1)
        _bvsb = tmp[:, -1] - tmp[:, -2]
        resIndex = []
        for i in range(len(proba1)):
            if _bvsb[i] > filter:
                resIndex.append(i)
        LOGGER.debug(f'满足bvsb间隔大于{filter}的个数为{len(resIndex)}个')
        maxSize = int(min(self.singleIterMaxLength , len(resIndex)) * -1)
        # 满足的值的前maxSize个索引
        index = np.argsort(_bvsb[resIndex])[maxSize:]
        LOGGER.debug(f'根据条件共需要选出前{maxSize * -1}个')
        # 最终的索引
        return np.array(resIndex)[index]

    '''
        根据概率对数据进行排序，取排序的单次最大个数
    '''
    def _probaIndex(self,proba1):
        LOGGER.debug(f"根据预测概率进行筛选")
        assert  proba1.shape[1]>=2
        tmp=np.max(proba1,axis=1)
        maxSize=int(min(self.singleIterMaxLength,len(tmp))*-1)
        index=np.argsort(tmp)[maxSize:]
        return index


    # 获取下一次迭代需要添加的数据的索引
    def getIterDataIndex(self, proba1, proba2):
        sameIndex = self._getSameCategoryIndex(proba1, proba2)
        LOGGER.debug(f'共获取标签相同元素{len(sameIndex)}个')
        if self.needBVSB:
            LOGGER.info(f"needBvSB is {self.needBVSB}，进行BVSB筛选")
            index1 = self._bvsbIndex(proba1[sameIndex], self.bvsbFilter)
            index2 = self._bvsbIndex(proba2[sameIndex], self.bvsbFilter)
        # tmp = np.argwhere(_bvsb2_index == _bvsb1_index).flatten().astype(int)
        # LOGGER.debug(f'均满足条件的数据的个数有{len(tmp)}个')
        # data_index = sameIndex[_bvsb1_index[tmp]]
        else:
            LOGGER.info(f"needBVSB is {self.needBVSB},进行概率筛选")
            index1=self._probaIndex(proba1[sameIndex])
            index2=self._probaIndex(proba2[sameIndex])
        LOGGER.debug(f"训练器1 筛选出来{len(index1)}个数据")
        LOGGER.debug(f"训练器2筛选出来{len(index2)}个数据")
        id = np.intersect1d(index1,index2 )
        data_index = sameIndex[id]
        LOGGER.debug(f'均满足条件的数据个数有{len(id)}个')
        return data_index

    # 将下一次需要迭代训练的数据添加到训练集中
    def resetTrainAndIterData(self, iterDataIndex):
        prepareY = self.iter_y[iterDataIndex]
        self.first_train_x = numpy.append(self.first_train_x, self.first_iter_x[iterDataIndex], axis=0)
        self.second_train_x = numpy.append(self.second_train_x, self.second_iter_x[iterDataIndex], axis=0)
        self.train_y = numpy.append(self.train_y, prepareY, axis=0)
        self.first_iter_x = np.delete(self.first_iter_x, iterDataIndex, axis=0)
        self.second_iter_x = np.delete(self.second_iter_x, iterDataIndex, axis=0)
        self.iter_y = np.delete(self.iter_y, iterDataIndex, axis=0)
        print(f'剩余训练数据个数为{len(self.iter_y)}')

    def predict(self, test_X):
        if self.needRSM:
            first_test_x = test_X[:, self.firstRSMIndex]
            second_test_x = test_X[:, self.secondRSMIndex]
        else:
            first_test_x = np.copy(test_X)
            second_test_x = np.copy(test_X)
        proba1 = self.firstClassifier.predict_proba(first_test_x)
        proba2 = self.secondClassifier.predict_proba(second_test_x)
        result = np.zeros(test_X.shape[0], dtype=np.int)
        first_max_p = np.max(proba1, axis=1)
        first_max_i = np.argmax(proba1, axis=1)
        second_max_p = np.max(proba2, axis=1)
        second_max_i = np.argmax(proba2, axis=1)
        for i in range(test_X.shape[0]):
            if first_max_p[i] > second_max_p[i]:
                result[i] = first_max_i[i]
            else:
                result[i] = second_max_i[i]
        return result


    def score(self, test_x, test_y):
        predict = self.predict(test_x).astype(int)
        from sklearn.metrics import accuracy_score
        return accuracy_score(test_y, predict)
