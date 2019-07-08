import numpy as np
import os
import pandas as pd


def load_data(path):
    data_path=os.path.join(path)
    if path.rfind(".csv")!=-1:
        data=pd.read_csv(data_path)
    elif path.rfind(".excel")!=-1:
        data=pd.read_excel(data_path)
    elif path.rfind(".json") != -1:
        data=pd.read_json(data_path)
    elif path.rfind(".txt") != -1:
        data=np.loadtxt(data_path)


