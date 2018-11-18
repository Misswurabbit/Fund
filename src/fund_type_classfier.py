from os import path
import pandas as pd
import numpy as np
import config


def data_process():
    # 读入基金类别的原始表格
    fundtype = pd.read_csv(path.abspath(path.join(__file__, "../../data/original/ODS_MDS.FUNDTYPES.csv")),
                           encoding="gb18030", usecols=["SYMBOL", "FUNDTYPES2", "CHANGEDATE", "ENDDATE"])
    # 对dataframe的dtypes数据格式进行处理
    fundtype.rename(columns={'CHANGEDATE': "start", "ENDDATE": "end"}, inplace=True)
    fundtype['start'] = pd.to_datetime(fundtype['start'])
    fundtype['end'] = pd.to_datetime(fundtype['end'])
    # 建立一个‘SYMBOL’（基金序号）和‘TYPE’（基金的类型）相对应的dataframe以方便之后的操作
    fundtype = fundtype[fundtype["FUNDTYPES2"].isin([100101, 100201, 100301, 100401, 100501])]
    fundtype.reset_index(inplace=True,drop=True)
    for index in range(len(fundtype)):
        if pd.isna(fundtype.loc[index, "end"]) or fundtype.loc[index, 'start'] > fundtype.loc[index, 'end']:
            fundtype.loc[index, 'end'] = pd.datetime.now()
    fundtype["SYMBOL"] = pd.to_numeric(fundtype["SYMBOL"], errors="coerce")
    temp = pd.read_csv("../data/processed_data/processed_data.csv")
    temp = temp.merge(fundtype, how="inner")
    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp[temp['start']<temp['date']]
    temp = temp[temp['date']<temp['end']]
    temp.drop(['start','end'],inplace=True,axis =1)
    share_data = temp[temp.FUNDTYPES2.isin([100101, 100201])]
    d = share_data.describe()
    share_data['p5'] = share_data.max_ratio > d.loc["25%", "max_ratio"]
    share_data['p10'] = share_data.max_ratio > d.loc["50%", "max_ratio"]
    share_data['p15'] = share_data.max_ratio > d.loc["75%", "max_ratio"]
    share_data['p-5'] = share_data.min_ratio < d.loc["75%", "min_ratio"]
    share_data['p-10'] = share_data.min_ratio < d.loc["50%", "min_ratio"]
    share_data['p-15'] = share_data.min_ratio < d.loc["25%", "min_ratio"]

    bond_data = temp[temp.FUNDTYPES2.isin([100301, 100201])]
    bond_data['p5'] = bond_data.max_ratio > d.loc["25%", "max_ratio"]
    bond_data['p10'] = bond_data.max_ratio > d.loc["50%", "max_ratio"]
    bond_data['p15'] = bond_data.max_ratio > d.loc["75%", "max_ratio"]
    bond_data['p-5'] = bond_data.min_ratio < d.loc["75%", "min_ratio"]
    bond_data['p-10'] = bond_data.min_ratio < d.loc["50%", "min_ratio"]
    bond_data['p-15'] = bond_data.min_ratio < d.loc["25%", "min_ratio"]
    share_data.to_csv("../data/processed_data/shares_fund_data.csv", index=False)
    bond_data.to_csv("../data/processed_data/bond_fund_data.csv", index=False)
    return

    # classfier = pd.DataFrame(columns=["SYMBOL", "TYPE"])
    # # 对所有选中的基金进行遍历
    # for name in config.Config.name_list:
    #     temp = fundtype[fundtype["SYMBOL"] == name]
    #     # 以下一个遍历用来确定每一个基金对应的类型，由主观分析确定100101（股票型基金），100201（混合型基金），100301（债券型基金）
    #     # 100401（货币式基金），100501（QDII基金）为主要分类标准，打标签
    #     for type in temp['FUNDTYPES2']:
    #         if type in [100101, 100201, 100301, 100401, 100501]:
    #             classfier = classfier.append(pd.DataFrame({'SYMBOL': [name], 'TYPE': [type]}))
    #             break
    # classfier.reset_index(inplace=True, drop=True)
    # # 将几种不同类型基金的‘SYMBOL’(基金代码)分别输出
    # shares_fund = classfier[classfier['TYPE'] == 100101].SYMBOL.tolist()
    # mixed_fund = classfier[classfier['TYPE'] == 100201].SYMBOL.tolist()
    # bond_fund = classfier[classfier['TYPE'] == 100301].SYMBOL.tolist()
    # curr_fund = classfier[classfier['TYPE'] == 100401].SYMBOL.tolist()
    # QDII_fund = classfier[classfier['TYPE'] == 100501].SYMBOL.tolist()
    # type_list = [shares_fund, mixed_fund, bond_fund, curr_fund, QDII_fund]
    # type_list_name = ['shares_fund', 'mixed_fund', 'bond_fund', 'curr_fond', 'QDII_fund']
    # for num in range(len(type_list)):
    #     # 如果某一种基金不存在，则break
    #     if len(type_list[num]) == 0:
    #         break
    #     # 该种类型的基金代码
    #     fund_name = type_list[num]
    #     # 合并所有该种类型的基金表
    #     temp = pd.read_csv("../data/processed_data/" + str(fund_name[0]) + "__processed_data.csv")
    #     for index in range(1, len(fund_name)):
    #         temp = temp.append(pd.read_csv("../data/processed_data/" + str(fund_name[index]) + "__processed_data.csv"))
    #     # 对分类的结果进行标签的注释
    #     temp.drop(["p5", "p10", "p15", "p-5", "p-10", "p-15"],inplace=True,axis=1)
    #     temp['p5'] = temp.max_ratio > 1.02
    #     temp['p10'] = temp.max_ratio > 1.06
    #     temp['p15'] = temp.max_ratio > 1.1
    #     temp['p-5'] = temp.min_ratio < 0.98
    #     temp['p-10'] = temp.min_ratio < 0.94
    #     temp['p-15'] = temp.min_ratio < 0.9
    #     temp.reset_index(drop=True, inplace=True)
    #     # 输出该种类型的基金表
    #     temp.to_csv("../data/processed_data/" + type_list_name[num] + "_data.csv", index=False)
