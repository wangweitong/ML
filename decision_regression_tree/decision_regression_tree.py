import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# ensemble 整体 集成
# test = load_data('E:\data\input\\test.txt')
# print(test)

data = pd.read_csv("E:\data\input\santander-customer-transaction-prediction\\train.csv")
data['target'] = abs(data['target'])
test = data.sample(frac=0.02)[['var_0', 'var_2', 'var_4']]
# print(test)
features = test[['var_0', 'var_2']]
label = abs(test['var_4'])
lr = DecisionTreeRegressor(random_state=0).fit(features, label)

print(lr.predict(features * 2))
