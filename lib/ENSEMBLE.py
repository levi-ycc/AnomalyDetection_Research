from lib.LOCALOUTLIERFACTOR import LocalOutlierFactor as lof
from lib.ISOLATIONFOREST import IsolationForest as isf
from lib.MAHALANOBIS import MahalanobisDistance as md
import pandas as pd
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')


class Ensemble:
    def __init__(self, isfParams={'outliers_fraction':0.015, 'n_estimators':100,
                'max_samples':256},
                 lofParams={'n_neighbors':2, 'novelty':True}, time='2019/8/30'):
        self.isfParams = isfParams
        self.lofParams = lofParams
        self.__switch_dict = {'0':'green', '1':'yellow', '2':'orange', '3':'red'}
        self.__inv_switch_dict = dict(zip(self.__switch_dict.values(), self.__switch_dict.keys()))
        self.__consolidate(time)
        self.__rank()

    def __consolidate(self, time):
        mdOut = md(time)
        isfOut = isf(self.isfParams.get('outliers_fraction'),
                     self.isfParams.get('n_estimators'),
                     self.isfParams.get('max_samples'),time)
        lofOut = lof(self.lofParams.get('n_neighbors'),
                     self.lofParams.get('novelty'),time)
        self._md=mdOut.new_df
        self._isf=isfOut.new_df
        self._lof=lofOut.new_df

        self.df=self._md[['日期','時間','outlier']]
        self.df.columns=['日期','時間','MD']
        self.df['ISF']=self._isf['outlier']
        self.df['LOF']=self._lof['outlier']

    def __rank(self):
        anomaly = self.df.apply(lambda x: sum(x[['LOF','MD','ISF']]), axis=1)
        self.df['anomaly']=anomaly
        self.df['rank']=self.df.apply(lambda x: self.__switch_dict.get(str(x['anomaly'])), axis=1)

    def show(self, emergency='red', columns=['日期','時間','rank']):
        return self.df[self.df['rank']==emergency][columns]

    def metric(self, benchmark='大立光', stdNum=2):
        if 'close' not in benchmark:
            benchmark+='_close'
        self.metric_df=self.df
        self.metric_df['close']=self._md.get(benchmark)
        self.metric_df['diff']=self.metric_df['close'].diff().shift(-1)
        self.metric_df.fillna(method='ffill', inplace=True)
        self.anomaly_threshold=self.metric_df['diff'].abs().mean()+stdNum*(self.metric_df['diff'].abs().std())
        total_anomaly_num = self.metric_df[self.metric_df['diff']>=self.anomaly_threshold].count()[-1]
        discovered_anomaly_num = self.metric_df[self.metric_df['diff']>=self.anomaly_threshold][self.metric_df['anomaly']==1].count()[-1]
        self.measure = discovered_anomaly_num/total_anomaly_num

        return self.measure

    def getCompany(self):
        return [i.replace('_close','') for i in self._md.columns if 'close' in i]
