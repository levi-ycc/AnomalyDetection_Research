import os
from lib.util import *
import pandas as pd

def eliminate_time(df):
    new_df = df[~((df['時間']=='09:00:00') | (df['時間']=='13:15:00'))]
    new_df.index=range(len(new_df))

    return new_df

def split_index(df, time='2019/8/30'):
    endDate=df[df['日期']==time].index[-1]+1

    return endDate