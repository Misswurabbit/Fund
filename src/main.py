import pandas as pd
from os import path
from config import Config

original_nav_csv = path.abspath(path.join(__file__, "../../../data/original/ODS_MDS.BSHEET.csv"))
file = pd.read_csv(original_nav_csv, encoding="gb18030")
temp = file.loc[:, ["SYMBOL", "BSHEET10", "BSHEET11", "BSHEET12", "BSHEET13"]]
temp.dropna(inplace=True)
temp.reset_index(drop=True,inplace=True)
nameusing = pd.DataFrame(data={"SYMBOL":Config.name_list})
namelist = Config.name_list
absense = namelist
for name in temp["SYMBOL"]:
    if int(name) in namelist:
        absense.remove(name)
print("Over")
