#!/usr/bin/env python
# coding: utf-8

# # LightGBM模型调优
# 

# In[1]:


# 首先 import 必要的模块
import pandas as pd 
import numpy as np

import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 读取数据

# In[2]:


# 读取数据
train = pd.read_csv("FE_train.csv")
train = train.drop(["ID"], axis=1)
train=train.fillna(0)
y_train = train['Disbursed'] 
X_train = train.drop(["Disbursed"], axis=1)
X_train.head(5)


# In[3]:



#保存特征名字以备后用（可视化）
feat_names = X_train.columns 


# ## LightGBM超参数调优

# LightGBM的主要的超参包括：
# 1. 树的数目n_estimators 和 学习率 learning_rate
# 2. 树的最大深度max_depth 和 树的最大叶子节点数目num_leaves（LightGBM采用叶子优先的方式生成树，num_leaves很重要，设置成比 2^max_depth 小）
# 3. 叶子结点的最小样本数:min_data_in_leaf(min_data, min_child_samples)
# 4. 每棵树的列采样比例：feature_fraction/colsample_bytree
# 5. 每棵树的行采样比例：bagging_fraction （需同时设置bagging_freq=1）/subsample
# 6. 正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)
# 
# 7. 两个非模型复杂度参数，但会影响模型速度和精度。可根据特征取值范围和样本数目修改这两个参数
# 1）特征的最大bin数目max_bin：默认255；
# 2）用来建立直方图的样本数目subsample_for_bin：默认200000。
# 
# 对n_estimators，用LightGBM内嵌的cv函数调优，因为同XGBoost一样，LightGBM学习的过程内嵌了cv，速度极快。
# 其他参数用GridSearchCV

# In[4]:


MAX_ROUNDS = 10000


# ### 相同的交叉验证分组
# 样本数太多（87020），CV折数越多，cv性能越好。
# 可能是由于GBDT是很复杂的模型，其实数据越多越好（cv折数多，每次留出的样本少，参数模型训练的样本更多）

# In[5]:


# prepare cross validation
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)


# ### 1. n_estimators

# In[6]:


#直接调用lightgbm内嵌的交叉验证(cv)，可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证，且速度相对较慢
def get_n_estimators(params , X_train , y_train , early_stopping_rounds=10):
    lgbm_params = params.copy()
     
    lgbmtrain = lgbm.Dataset(X_train , y_train )
     
    #num_boost_round为弱分类器数目，下面的代码参数里因为已经设置了early_stopping_rounds
    #即性能未提升的次数超过过早停止设置的数值，则停止训练
    cv_result = lgbm.cv(lgbm_params , lgbmtrain , num_boost_round=MAX_ROUNDS , nfold=5,  metrics='auc' , early_stopping_rounds=early_stopping_rounds,seed=3 )
     
    print('best n_estimators:' , len(cv_result['auc-mean']))
    print('best cv score:' , cv_result['auc-mean'][-1])
     
    return len(cv_result['auc-mean'])


# In[7]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          #'categorical_feature': names:'City', 'Employer_Name', 'Salary_Account','Device_Type','Filled_Form','Gender','Mobile_Verified','Source','Var1','Var2','Var4',
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.1,
          #'n_estimators':n_estimators_1,
          'num_leaves': 60,
          'max_depth': 6,
          'colsample_bytree': 0.7,
          'verbosity':5
         }

#categorical_feature = ['City', 'Employer_Name', 'Salary_Account','Device_Type','Filled_Form','Gender','Mobile_Verified','Source','Var1','Var2','Var4']
n_estimators_1 = get_n_estimators(params, X_train , y_train)


# ### 2. num_leaves & max_depth=7
# num_leaves建议70-80，搜索区间50-80,值越大模型越复杂，越容易过拟合
# 相应的扩大max_depth=7

# In[8]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          #'num_leaves': 60,
          'max_depth': 6,
          'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

num_leaves_s = range(50,90,10) #50,60,70,80
tuned_parameters = dict( num_leaves = num_leaves_s)

grid_search = GridSearchCV(lg, n_jobs=4, param_grid=tuned_parameters, cv = kfold, scoring="roc_auc", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
#grid_search.best_estimator_


# In[9]:


# examine the best model
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[10]:


# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

n_leafs = len(num_leaves_s)

x_axis = num_leaves_s
plt.plot(x_axis, test_means)
#plt.errorbar(x_axis, -test_means, yerr=test_stds,label = ' Test')
#plt.errorbar(x_axis, -train_means, yerr=train_stds,label = ' Train')
plt.xlabel( 'num_leaves' )
plt.ylabel( 'AUC' )
plt.show()


# In[11]:


test_means


# #### 性能抖动，取系统推荐值：70

# ### 3. min_child_samples
# 叶子节点的最小样本数目
# 
# 叶子节点数目：70，共2类，平均每类35个叶子节点
# 每棵树的样本数目数目最少的类（稀有事件）的样本数目：8w * 4/5 * 1.4% = 840
# 所以每个叶子节点约840/35 = 25个样本点
# 
# 搜索范围：10-50

# In[12]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'num_leaves': 70,
          'max_depth': 6,
          'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

min_child_samples_s = range(10,50,10) 
tuned_parameters = dict( min_child_samples = min_child_samples_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="roc_auc", verbose=5, refit = False)
grid_search.fit(X_train , y_train)


# In[13]:


# examine the best model
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[14]:


# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = min_child_samples_s

plt.plot(x_axis, test_means)
#plt.errorbar(x_axis, -test_scores, yerr=test_stds ,label = ' Test')
#plt.errorbar(x_axis, -train_scores, yerr=train_stds,label =  +' Train')

plt.show()


# In[15]:


test_means


# #### 再次细调

# In[16]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'num_leaves': 70,
          'max_depth': 6,
          'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

min_child_samples_s = range(40,60,10) 
tuned_parameters = dict( min_child_samples = min_child_samples_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="roc_auc", verbose=5, refit = False)
grid_search.fit(X_train , y_train)


# In[17]:


# examine the best model
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[18]:


# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = min_child_samples_s

plt.plot(x_axis, test_means)
#plt.errorbar(x_axis, -test_scores, yerr=test_stds ,label = ' Test')
#plt.errorbar(x_axis, -train_scores, yerr=train_stds,label =  +' Train')

plt.show()


# #### min_child_samples=40

# ### 列采样参数 sub_feature/feature_fraction/colsample_bytree

# In[19]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'num_leaves': 70,
          'max_depth': 6,
          'min_child_samples':40
          #'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

colsample_bytree_s = [i/10.0 for i in range(5,10)]
tuned_parameters = dict( colsample_bytree = colsample_bytree_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="roc_auc", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
#grid_search.best_estimator_


# In[20]:


# examine the best model
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[21]:


# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = colsample_bytree_s

plt.plot(x_axis, test_means)
#plt.errorbar(x_axis, -test_scores[:,i], yerr=test_stds[:,i] ,label = str(max_depths[i]) +' Test')
#plt.errorbar(x_axis, -train_scores[:,i], yerr=train_stds[:,i] ,label = str(max_depths[i]) +' Train')

plt.show()


# 再调小一点

# In[22]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'num_leaves': 70,
          'max_depth': 6,
          'min_child_samples':40
          #'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

colsample_bytree_s = [i/10.0 for i in range(3,6)]
tuned_parameters = dict( colsample_bytree = colsample_bytree_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="roc_auc", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
#grid_search.best_estimator_


# In[23]:


# examine the best model
print(grid_search.best_score_)
print(grid_search.best_params_)


# #### colsample_bytree=0.4

# ### 正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)感觉不用调了

# ### 减小学习率，调整n_estimators

# In[24]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.01,
          #'n_estimators':n_estimators_1,
          'num_leaves': 70,
          'max_depth': 6,
          'min_child_samples':40,
          'colsample_bytree': 0.4
         }
n_estimators_2 = get_n_estimators(params , X_train , y_train, early_stopping_rounds=50)


# ### 用所有训练数据，采用最佳参数重新训练模型
# 由于样本数目增多，模型复杂度稍微扩大一点？
# num_leaves增多5
# #min_child_samples按样本比例增加到15

# In[25]:


params = {'boosting_type': 'goss',
          'objective': 'binary',
          'is_unbalance':True,
          'categorical_feature': [0,1,3,5,6,12,15,16,17,18,19,20],
          'n_jobs': 4,
          'learning_rate': 0.01,
          #'n_estimators':n_estimators_1,
          'num_leaves': 75,
          'max_depth': 6,
          'min_child_samples':40,
          'colsample_bytree': 0.4
         }

lg = LGBMClassifier(silent=False,  **params)
lg.fit(X_train, y_train)


# ## 保存模型，用于后续测试

# In[26]:


import pickle

pickle.dump(lg, open("HappyBank_LightGBM_.pkl", 'wb'))


# ### 特征重要性

# In[27]:


df = pd.DataFrame({"columns":list(feat_names), "importance":list(lg.feature_importances_.T)})
df = df.sort_values(by=['importance'],ascending=False)


# In[28]:


df


# In[29]:


plt.bar(range(len(lg.feature_importances_)), lg.feature_importances_)
plt.show()


# In[ ]:




