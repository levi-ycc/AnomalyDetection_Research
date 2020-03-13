from lib.preprocess import *
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as lof
from lib.model import Model

class LocalOutlierFactor(Model):
    def __init__(self, n_neighbors=2, novelty=True, time='2019/8/30'):
        self.df=pd.read_csv('consolidate.csv', encoding='big5', index_col=0)
        self.new_df=eliminate_time(self.df)
        self.__endDate=split_index(self.new_df, time)
        self.__n_neighbors=n_neighbors
        self.__novelty=novelty
        self.__split()
        self.__train()
        self.__predict()
        self.__out()

    def __split(self):
        self.array=self.new_df.drop(columns=['日期', '時間']).values
        self.trainArray = self.array[0:self.__endDate]
        self.testArray = self.array[self.__endDate:]
        self.dfTrain = self.new_df.iloc[0:self.__endDate]
        self.dfTest = self.new_df.iloc[self.__endDate:]

    def __train(self):
        self.__clf = lof(n_neighbors=self.__n_neighbors, novelty=self.__novelty).fit(self.trainArray)

    def __predict(self):
        self.y_train = self.__clf.predict(self.trainArray)

    def __out(self):
        self.dfTrain['outlier']=self.y_train
        self.new_df['outlier']=self.__clf.predict(self.array)
        self.new_df['outlier']=self.new_df['outlier']==-1

    def save(self, save_dir='result/lof.csv'):
        self.new_df[['日期','時間','outlier']].to_csv(save_dir)