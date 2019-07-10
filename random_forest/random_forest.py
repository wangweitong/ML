import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ensemble 整体 集成


data = pd.read_csv("E:\data\input\santander-customer-transaction-prediction\\train.csv")
data['target'] = abs(data['target'])
test = data.sample(frac=0.02)[['var_0', 'var_2', 'target']]
# print(test)
features = test[['var_0', 'var_2']]
label = abs(test['target'])
lr = RandomForestClassifier(bootstrap='true', class_weight=None, criterion='gini',max_depth=2,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                            oob_score=False, random_state=0, verbose=0, warm_start=False).fit(features, label)

print(lr.predict(features * 2))
