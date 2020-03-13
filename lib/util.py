import pandas as pd
import os
import re

mapping_csv = 'mapping.csv'
mapping_pd = pd.read_csv(mapping_csv, encoding='big5')
mapping_dict = dict(zip(mapping_pd['c.ID'], mapping_pd['c.Name']))

def readCSV(file):
    return pd.read_csv(file, encoding='big5')

def changColName(file):
    df = readCSV(file)
    cID = re.findall("[0-9]+", file)[0]
    cID = mapping_dict.get(int(cID))
    df.columns = ['日期', '時間', cID+'_open', cID+'_high', cID+'_low', cID+'_close', cID+'_vol']
    df['vwap'] = vwap(df[cID+'_high'], df[cID+'_low'], df[cID+'_close'], df[cID+'_vol'])
    df.drop(axis=1, columns=[cID+'_open', cID+'_high', cID+'_low'], inplace=True)

    return df

def mergeTables(df1, df2=None):
    df1 = df1.merge(df2, on=['日期', '時間'])
    return df1

def vwap(high, low, close, vol):
    return vol*(high+low+close)/3

def exclude_finance(df, droplist = ['富邦金', '國泰金', '玉山金', '兆豐金', '中信金']):
    mapping_csv = 'mapping.csv'
    mapping_pd = pd.read_csv(mapping_csv, encoding='big5')
    mapping_dict = dict(zip(mapping_pd['c.ID'], mapping_pd['c.Name']))
    droplistCode = []
    for l in droplist:
        droplistCode.append(str((dict(zip(mapping_dict.values(), mapping_dict.keys())).get(l)))+'.csv')

    path = 'data/'
    files = [path+i for i in os.listdir(path) if re.findall('\d+',i) and i not in droplistCode]

    consolidateDF = changColName(files[0])
    for i in range(len(files)-1):
        tmp_df = changColName(files[i+1])
        consolidateDF = mergeTables(consolidateDF, tmp_df)

    consolidateDF.to_csv('consolidate.csv', encoding='big5')