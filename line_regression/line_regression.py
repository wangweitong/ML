import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("E:\data\input\champs-scalar-coupling\\train.csv")
data['scalar_coupling_constant'] = abs(data['scalar_coupling_constant'])
test = data.sample(frac=0.02)[['atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]
# print(test)
features = test[['atom_index_0', 'atom_index_1']]
label = abs(test['scalar_coupling_constant'])

# C 是正则化系数，越小，正则化越强
lr = LinearRegression().fit(features, label)

print(lr.score(features, label))
Z = lr.predict(features * 2)
print(Z)
