#!/usr/bin/env python
# coding: utf-8

# # Happy Customer Bank目标客户（贷款成功的客户）识别

# 利用LightGBM/XGboost实现Happy Customer Bank目标客户（贷款成功的客户）识别
# 
# 
# 一、	任务说明：Happy Customer Bank目标客户识别
# https://discuss.analyticsvidhya.com/t/hackathon-3-x-predict-customer-worth-for-happy-customer-bank/3802
# 

# In[1]:


# 首先 import 必要的模块
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 读取数据

# In[2]:


# 读取数据
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
train.head(5)
test.head(5)


# In[3]:


train.info()


# 

# In[4]:


## 各属性的统计特性
train.describe()


# # 标签的分布

# In[5]:


# Target 分布，看看各类样本分布是否均衡
# Target 分布，两类样本分布严重不均衡，只有1.4%的样本Disbursed为1
sns.countplot(train['Disbursed']);
plt.xlabel('Disbursed');
plt.ylabel('Number of occurrences');


# 发现数据严重不均衡

# In[6]:


#合成一个总的data，方便一起做特征工程
train['source']= 'train'
test['source'] = 'test'
data = pd.concat([train, test],ignore_index=True)
data.shape


# ### 检查数据质量：异常点、缺省值

# In[7]:


data.apply(lambda x: sum(x.isnull()))


# In[8]:


cat_features = ['Gender','City','Employer_Name','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source','Var4']
#图太多，还是显示文字
#for col in cat_features:
# col= cat_features[0]
# sns.countplot(train[col]);
# plt.xlabel(col);
# plt.ylabel('Number of occurrences');
# col= cat_features[1]
# sns.countplot(train[col]);
# plt.xlabel(col);
# plt.ylabel('Number of occurrences');
for col in cat_features:
    num_vlaules = len(data[col].unique())
    print('\n%s属性有%d的不同取值，各取值及其出现的次数\n'% (col,num_vlaules))
    print(data[col].value_counts())


# In[9]:


cat_features = ['City','Employer_Name','Salary_Account', 'Source']
rare_thresholds = [100, 30, 40, 40]
j=0
for col in cat_features:
    #每个取值的样本数目
    value_counts_col =  data[col].value_counts(dropna=False)

    #样本数目小于阈值的取值为稀有取值
    rare_threshold = rare_thresholds[j]
    value_counts_rare = list(value_counts_col[value_counts_col < rare_threshold ].index)

    #稀有值合并为：others
    rare_index = data[col].isin(value_counts_rare)
    data.loc[ data[col].isin(value_counts_rare), col] = "Others"
    
    j = j+1


# ### City、Employer_Name、Salary_Account、Source
# 这些特征都是取值很多,
# 取前几个重要的，其余合并成一个：others
# 
# LightGBM对类别特征建立直方图时，当特征取值数目超过max_bin(默认255)，会去掉样本数目少的类别：
# 统计该特征下每一种离散值出现的次数，并从高到低排序，并过滤掉出现次数较少的特征值, 
# 然后为每一个特征值，建立一个bin容器, 对于在bin容器内出现次数较少的特征值直接过滤掉，不建立bin容器。

# ### DOB
# DOB是出生的具体日期，具体日期可能没作用，转换成年龄(申请贷款的年龄)

# In[10]:


#创建一个年龄的字段Age
data['Age'] = pd.to_datetime(data['Lead_Creation_Date']).dt.year - pd.to_datetime(data['DOB']).dt.year
#data['Age'].head()
#把原始的DOB字段去掉:
data.drop(['DOB', 'Lead_Creation_Date'],axis=1,inplace=True)


# ### Loan Tenure 

# In[11]:


#不合理的贷款年限，设为缺失值
data['Loan_Tenure_Applied'].replace([10,6,7,8,9],value = np.nan, inplace = True)
data['Loan_Tenure_Submitted'].replace(6, np.nan, inplace = True)


# ### 类别特征先编码成数值，LightGBM无需One-hot编码

# In[12]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
feats_to_encode = ['City', 'Employer_Name', 'Salary_Account','Device_Type','Filled_Form','Gender','Mobile_Verified','Source','Var1','Var2','Var4']
for col in feats_to_encode:
    data[col] = le.fit_transform(data[col].astype(str))


# ### 发现EMI_Loan_Submitted和Var5特征数据有个别异常值，对其进行处理

# In[13]:


data =data.replace("N","NaN")
data =data.replace("HBXX","NaN")


# ## 最终的数据样式

# In[14]:


data.head()


# # 特征之间的相关系数

# In[15]:


#get the names of all the columns
cols = data.columns 

# Calculates pearson co-efficient for all combinations，通常认为相关系数大于0.5的为强相关
feat_corr = data.corr().abs()

plt.subplots(figsize=(13, 9))
sns.heatmap(feat_corr,annot=True)

# Mask unimportant features
sns.heatmap(feat_corr, mask=feat_corr < 1, cbar=False)
plt.show()


# In[16]:


#Set the threshold to select only highly correlated attributes
threshold = 0.5
# List of pairs along with correlation above threshold
corr_list = []
#size = data.shape[1]
size = feat_corr.shape[0]

#Search for the highly correlated pairs
for i in range(0, size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (feat_corr.iloc[i,j] >= threshold and feat_corr.iloc[i,j] < 1) or (feat_corr.iloc[i,j] < 0 and feat_corr.iloc[i,j] <= -threshold):
            corr_list.append([feat_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# 特征之间相关性很高，一定要加正则，也可以考虑对特征进行降维（PCA/t-SNE）

# In[17]:


data.fillna(0)


# ## 特征工程

# In[18]:


#data=data.astype(float)
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop(['source','LoggedIn'],axis=1,inplace=True)
test.drop(['source','Disbursed','LoggedIn'],axis=1,inplace=True)
train.to_csv('FE_train.csv',index=False)
test.to_csv('FE_test.csv',index=False)


# In[19]:


train.head(5)


# In[20]:


train.shape


# In[ ]:





# In[ ]:




