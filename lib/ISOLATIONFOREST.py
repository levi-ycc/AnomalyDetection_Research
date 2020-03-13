from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest as iforest
from lib.preprocess import *
import pandas as pd
import numpy as np
from lib.model import Model

class IsolationForest(Model):
    def __init__(self, outliers_fraction=0.015, n_estimators=100, max_samples=256,
                 time='2019/8/30'):
        self.df = pd.read_csv('consolidate.csv', index_col=0, encoding='big5')
        self.new_df = eliminate_time(self.df)
        self.endDate = split_index(self.new_df, time)
        self.outliers_fraction = outliers_fraction
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.rng = np.random.RandomState(123)
        self.__useColumns()
        self.__split()
        self.__train()
        self.__predict()
        self.__out()

    def __useColumns(self):
        self.usedColumns = list(self.new_df.columns)
        self.usedColumns.remove('日期')
        self.usedColumns.remove('時間')

    def __split(self):
        self.train_X = self.new_df[self.usedColumns][0:self.endDate]
        self.test_X = self.new_df[self.usedColumns][self.endDate:]

    def __train(self):
        self.clf = iforest(max_samples=self.max_samples, contamination=self.outliers_fraction,
                           random_state=self.rng, n_estimators=self.n_estimators)
        self.clf.fit(self.train_X)

    def __predict(self):
        self.y_pred_train = self.clf.predict(self.train_X)
        self.y_pred_test = self.clf.predict(self.test_X)
        self.pred = self.clf.predict(self.new_df[self.usedColumns])

        self.train_X_df = self.new_df[:self.endDate]
        self.train_X_df['pred'] = self.y_pred_train

        self.test_X_df = self.new_df[self.endDate:]
        self.test_X_df['pred'] = self.y_pred_test

        self.new_df['pred']=self.pred

    def __out(self):
        self.new_df['outlier']=self.new_df['pred']==-1

    def save(self, save_dir='result/if.csv'):
        self.consolidated_df[['日期', '時間', 'outlier']].to_csv(save_dir)