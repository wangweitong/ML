from etl.load_data import load_data
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# test = load_data('E:\data\input\\test.txt')
# print(test)

data = pd.read_csv("E:\data\input\champs-scalar-coupling\\train.csv")[2:]
test = data.sample(frac=0.02)[['atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]
X = test[['atom_index_0', 'atom_index_1']]
Y = test['scalar_coupling_constant'].astype(float)
# C 是正则化系数，越小，正则化越强
lr = linear_model.LogisticRegression(C=1e5)
print(test.describe)
lr.fit(X, Y)
Z = lr.predict(X * 2)
print(test)
