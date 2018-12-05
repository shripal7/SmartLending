
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# In[36]:


df = pd.read_csv('../../data/combine.csv')


# In[37]:


df.head()


# In[38]:


filterList1 = list(df)


# In[39]:


df.shape


# In[40]:


df.index.values


# In[41]:


df1 = df.iloc[:,0:2]


# In[42]:


df1.head()


# In[43]:


df.head()


# In[13]:


int(df.memory_usage(deep=False).sum()/1000000)


# In[14]:


# check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
# df.drop(check_null[check_null>0.6].index, axis=1, inplace=True)


# In[15]:


df = df.loc[df.count(1) > df.shape[1]/2, df.count(0) > df.shape[0]/2]


# In[16]:


df.shape


# In[17]:


filterList2 = list(df)


# In[44]:


unmatched_items_10 = [d for d in filterList1 if d not in filterList2]


# In[45]:


unmatched_items_10


# In[46]:


print(df.isnull().sum())


# In[47]:


df.dtypes


# In[48]:


def get_features():
    features = ['loan_amnt', 'amount_diff_inv', 'term',
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat',
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',
            'loan_status','int_rate'
           ]
    return features


# In[49]:


def string_strip(df = df, header = 'term'):
    df[header] = df[header].str.split(' ').str[1]
    return df


# In[50]:


def give_categorical(df = df):
    df['pub_rec_cat'] = 'n'
    df.loc[df['pub_rec']> 0,'pub_rec_cat'] = 'y'
    df['delinq_2yrs_cat'] = 'n'
    df.loc[df['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'y'
    df['inq_last_6mths_cat'] = 'n'
    df.loc[df['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'y'
    return df


# In[51]:


df.dropna(axis=0, thresh=30, inplace=True)
remove_features = ['zip_code', 'policy_code', 'pymnt_plan', 'application_type', 'acc_now_delinq']
df.drop(remove_features , axis=1, inplace=True)


# In[52]:


df.shape


# In[53]:


df['term'].head()


# In[54]:


df = string_strip(df, 'term')
df['term'].head()


# In[55]:


c_dates = df.dtypes[df.dtypes == 'datetime64[ns]'].index
for d in c_dates:
    df[d] = df[d].dt.to_period('M')
print(df['last_pymnt_d'].head())


# In[56]:


df['amount_diff_inv'] = 'same'
df.loc[(df['funded_amnt'] - df['funded_amnt_inv']) > 0,'amount_diff_inv'] = 'low'
df['acc_ratio'] = df.open_acc / df.total_acc


# In[57]:


df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].fillna(df.emp_length.median())


# In[58]:


df = give_categorical(df)
f = get_features()


# In[59]:


X_modified = df.loc[df.loan_status != 'Current', f]
mask = (X_modified.loan_status == 'Charged Off')
X_modified['target'] = 0
X_modified.loc[mask,'target'] = 1



# In[60]:


X_modified.dropna(axis=0, how = 'any', inplace = True)


# In[61]:


X_modified.shape


# In[62]:


X_modified.to_csv('../../data/example.csv')


# In[65]:


X_modified

