from os import path
import pandas as pd
import config
from fund_type_classfier import data_process


def add_feature(filename, feature_name, feature_value_list,null_value_set):
    fundtype = pd.read_csv(path.abspath(path.join(__file__, "../../data/original/" + filename)),
                           encoding="gb18030", usecols=["SYMBOL", feature_name])
    # 对dataframe的dtypes数据格式进行处理
    fundtype["SYMBOL"] = pd.to_numeric(fundtype["SYMBOL"], errors="coerce")
    # 建立一个‘SYMBOL’（基金序号）和‘TYPE’（基金的类型）相对应的dataframe以方便之后的操作
    fundtype = fundtype[fundtype["FUNDTYPES2"].isin(feature_value_list)]
    temp = pd.read_csv("../data/processed_data/processed_data.csv")
    temp = temp.merge(fundtype, how="left")
    temp.fillna(null_value_set)
    temp.to_csv("../data/processed_data/processed_data_with_extra_feature.csv")
    return temp

def test_feature():
    data_process("processed_data.csv", "fund_data.csv")
    data_process("processed_data_with_extra_feature.csv", "fund_data_with_extra_feature.csv")
