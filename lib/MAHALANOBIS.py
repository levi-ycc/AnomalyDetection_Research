from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from lib.preprocess import *
from lib.model import Model

class MahalanobisDistance(Model):
    def __init__(self, time='2019/8/30'):
        self.df = pd.read_csv('consolidate.csv', index_col=0,
        encoding='big5')
        self.new_df = eliminate_time(self.df)
        self.array = self.new_df.drop(columns=['日期', '時間'])
        self.array.drop(columns=[x for x in list(self.array.columns) if 'vwap' in x], inplace=True)
        self.__endDate = split_index(self.new_df, time)
        self.__split()
        self.__get_df()
        self.__standarization()
        self.__threshold()
        self.__mahalanobis(self.stdTrainArray)
        self.__out()

    def __split(self):
        self.trainArray = self.array[0:self.__endDate]
        self.testArray = self.array[self.__endDate:]
        self.dfTrain = self.new_df.iloc[0:self.__endDate]
        self.dfTest = self.new_df.iloc[self.__endDate:]

    def __get_df(self, threshold=0.8):
        corr = self.trainArray[[i for i in self.array.columns if 'close' in i]].corr()
        count = 0
        for i in corr.values:
            for j in i:
                if j>threshold:
                    count +=1
        self.__freedom = self.trainArray.shape[1]-(count-corr.shape[0])/2

    def __standarization(self):
        stdScaler = StandardScaler()
        stdScaler.fit(self.trainArray)
        self.stdTrainArray = stdScaler.transform(self.trainArray)
        self.stdTestArray = stdScaler.transform(self.array)

    def __mahalanobis(self, array):
        self.__muTrain = array.mean(axis=0)
        cov = np.cov(array.T)
        self.__covInverse = np.linalg.inv(cov)

    def __threshold(self, alpha=0.01):
        kSquare = chi2.ppf(1-alpha, self.__freedom)
        self.__k = np.sqrt(kSquare)

    def __predict(self,array):
        mDistList = []
        for i in range(array.shape[0]):
            mDistance = distance.mahalanobis(self.__muTrain, array[i], self.__covInverse)
            mDistList.append(mDistance)
        return mDistList

    def __out(self):
        self.dfTrain['MDistance']=self.__predict(self.stdTrainArray)
        self.new_df['MDistance']=self.__predict(self.stdTestArray)
        self.new_df['outlier']=self.new_df['MDistance']>self.__k

    def save(self, save_dir='result/md.csv'):
        self.new_df[['日期','時間','outlier']].to_csv(save_dir)



