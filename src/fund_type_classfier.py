from os import path
import pandas as pd
import config


def fund_type_class():
    fundtype = pd.read_csv(path.abspath(path.join(__file__, "../../data/original/ODS_MDS.FUNDTYPES.csv")),
                           encoding="gb18030")
    fundtype["SYMBOL"] = pd.to_numeric(fundtype["SYMBOL"], errors="coerce")
    classfier = pd.DataFrame(columns=["SYMBOL", "TYPE"])
    for name in config.Config.name_list:
        temp = fundtype[fundtype["SYMBOL"] == name]
        for type in temp['FUNDTYPES2']:
            if type in [100101, 100201, 100301, 100401, 100501]:
                classfier = classfier.append(pd.DataFrame({'SYMBOL': [name], 'TYPE': [type]}))
                break
    classfier.reset_index(inplace=True, drop=True)
    shares_fund = classfier[classfier['TYPE'] == 100101].SYMBOL.tolist()
    mixed_fund = classfier[classfier['TYPE'] == 100201].SYMBOL.tolist()
    bond_fund = classfier[classfier['TYPE'] == 100301].SYMBOL.tolist()
    QDII_fund = classfier[classfier['TYPE'] == 100501].SYMBOL.tolist()
    type_list = [shares_fund, mixed_fund, bond_fund, QDII_fund]
    type_list_name = ['shares_fund', 'mixed_fund', 'bond_fund', 'QDII_fund']
    for num in range(len(type_list)):
        if len(type_list[num]) == 0:
            break
        fund_name = type_list[num]
        temp = pd.read_csv("../data/processed_data/" + str(fund_name[0]) + "__processed_data.csv")
        for index in range(1, len(fund_name)):
            temp = temp.append(pd.read_csv("../data/processed_data/" + str(fund_name[index]) + "__processed_data.csv"))
        temp.reset_index(drop=True, inplace=True)
        temp.to_csv(type_list_name[num] + "_data.csv", index=False)


fund_type_class()
