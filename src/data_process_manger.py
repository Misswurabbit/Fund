from os import path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


def data_process():
    data = pd.read_csv("../data/original/ODS_MDS.FUNDMG.csv", encoding="gb18030",
                       usecols=['SYMBOL', 'FUNDMG1', 'FUNDMG2', 'FUNDMG5', 'FUNDMG8', 'FUNDMG10'])
    data['SYMBOL'] = pd.to_numeric(data['SYMBOL'], errors="coerce")
    data.dropna(inplace=True)
    data = data[data["FUNDMG5"] == '基金经理']
    data.reset_index(drop=True, inplace=True)
    manger = data['FUNDMG2']
    manger.drop_duplicates(inplace=True)
    manger.reset_index(drop=True, inplace=True)
    manger = list(manger)
    data.rename(columns={'FUNDMG1': 'start', 'FUNDMG8': 'experience', 'FUNDMG10': 'end'}, inplace=True)
    data['start'] = pd.to_datetime(data['start'])
    data['end'] = pd.to_datetime(data['end'])
    data['manger'] = 0
    for index in range(len(data)):
        data.loc[index, 'manger'] = manger.index(data.loc[index, "FUNDMG2"])
        if pd.isna(data.loc[index, "end"]) or data.loc[index, 'start'] > data.loc[index, 'end']:
            data.loc[index, 'end'] = pd.datetime.now()
    data.drop(['FUNDMG2', 'FUNDMG5'], inplace=True, axis=1)
    data = data[data['end'] > data['start']]
    temp = pd.read_csv("../data/processed_data/processed_data.csv")
    temp.merge(data, on='SYMBOL', how='inner')
    print("Over")


data_process()
