import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# test = load_data('E:\data\input\\test.txt')
# print(test)

data = pd.read_csv("E:\data\input\champs-scalar-coupling\\train.csv")
data['scalar_coupling_constant'] = abs(data['scalar_coupling_constant'])
test = data.sample(frac=0.02)[['atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]
# print(test)
features = test[['atom_index_0', 'atom_index_1']]
label = abs(test['scalar_coupling_constant'])
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(features, label)

print(knn.predict(features*2))
