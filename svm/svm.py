import pandas as pd
from sklearn.svm import SVC

# test = load_data('E:\data\input\\test.txt')
# print(test)

data = pd.read_csv("E:\data\input\santander-customer-transaction-prediction\\train.csv")
data['target'] = abs(data['target'])
test = data.sample(frac=0.02)[['var_0', 'var_2', 'target']]
# print(test)
features = test[['var_0', 'var_2']]
label = abs(test['target'])
svm = SVC(gamma='auto').fit(features, label)

print(svm.predict(features * 2))
