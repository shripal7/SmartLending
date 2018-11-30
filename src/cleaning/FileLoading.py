#!/usr/bin/env python
# coding: utf-8

# In[499]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# In[500]:


df = pd.read_csv('/data/fa18-cmpe255-1/smartlending/combine.csv')


# In[501]:

print(df.shape)

print("header data")

print(df.head())


# In[463]:


filterList1 = list(df)


# In[464]:


df.shape


# In[465]:


df.index.values


# In[466]:


# df.loan_amnt == 8400


# In[467]:


# new_header = df.iloc[0] #grab the first row for the header
# df = df[1:] #take the data less the header row
# df.columns = new_header #set the header row as the df header


# In[468]:


# df1 = df[['loan_amnt', 'funded_amnt']]
df1 = df.iloc[:,0:2]


# In[469]:


df1.head()


# In[470]:


df.head()


# In[471]:


int(df.memory_usage(deep=False).sum()/1000000)


# In[472]:


# check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
# df.drop(check_null[check_null>0.6].index, axis=1, inplace=True)


# In[473]:


df = df.loc[df.count(1) > df.shape[1]/2, df.count(0) > df.shape[0]/2]


# In[474]:


df.shape


# In[475]:


filterList2 = list(df)


# In[476]:


unmatched_items_10 = [d for d in filterList1 if d not in filterList2]


# In[477]:


unmatched_items_10


# In[478]:


print(df.isnull().sum())


# In[479]:


df.dtypes


# In[480]:


df.dropna(axis=0, thresh=30, inplace=True)
delete_features = ['zip_code', 'policy_code', 'pymnt_plan', 'application_type', 'acc_now_delinq']
df.drop(delete_features , axis=1, inplace=True)


# In[481]:


df.shape


# In[482]:


df['term'].head()


# In[483]:


def string_strip(df = df, header = 'term'):
    df[header] = df[header].str.split(' ').str[1]
    return df


# In[484]:


df = string_strip(df, 'term')
df['term'].head()


# In[485]:


df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].fillna(df.emp_length.median())


# In[486]:


col_dates = df.dtypes[df.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    df[d] = df[d].dt.to_period('M')
print(df['last_pymnt_d'].head())


# In[487]:


df['amount_diff_inv'] = 'same'
df.loc[(df['funded_amnt'] - df['funded_amnt_inv']) > 0,'amount_diff_inv'] = 'low'
df['acc_ratio'] = df.open_acc / df.total_acc


# In[488]:


def give_categorical(df = df):
    df['pub_rec_cat'] = 'no'
    df.loc[df['pub_rec']> 0,'pub_rec_cat'] = 'yes'
    df['delinq_2yrs_cat'] = 'no'
    df.loc[df['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'
    df['inq_last_6mths_cat'] = 'no'
    df.loc[df['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'
    return df


# In[489]:


def get_features():
    features = ['loan_amnt', 'amount_diff_inv', 'term',
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat',
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',
            'loan_status'
           ]
    return features
# inq_last_6mths_cat


# In[490]:


df = give_categorical(df)
f = get_features()


# In[496]:


# df['inq_last_6mths_cat']


# In[492]:


cat_features = ['term','amount_diff_inv', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']


# In[493]:


# df['acc_ratio'] = df.open_acc / loan_data.total_acc
X_clean = df.loc[df.loan_status != 'Current', f]
mask = (X_clean.loan_status == 'Charged Off')
X_clean['target'] = 0
X_clean.loc[mask,'target'] = 1

X_clean.dropna(axis=0, how = 'any', inplace = True)


# In[494]:


# X_All.to_csv("Cleaned_8lakh_All_Null_Removed.csv")


# In[495]:


X_clean


# In[ ]:




