
# coding: utf-8

# <a href="https://colab.research.google.com/github/sunny-udhani/SmartLending/blob/master/src/tensorflow/neural_network_updated.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def read_dataset(filename=None):
  # read in the dataset
  frame = pd.read_csv(
      filepath_or_buffer=filename, 
      index_col=0)

  int_rate = frame.iloc[:,19:20]
  int_rate = int_rate['int_rate'].str.split('%').str.get(0)
  int_rate = pd.to_numeric(int_rate)
  
  frame = frame.drop(['int_rate'], axis=1)
  frame = frame.drop(['target'], axis=1)

  return frame,int_rate


# In[ ]:


def scale_my_dataset(data=None):
  scaler = preprocessing.StandardScaler()
  return scaler.fit_transform(data)


# In[ ]:


def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, 
                       activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],
                       )),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model


# In[ ]:


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [%]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()


# In[ ]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


# In[ ]:


df,int_rate=read_dataset("../../data/example.csv")


# In[121]:


le = preprocessing.LabelEncoder()
df['amount_diff_inv'] = le.fit_transform(df['amount_diff_inv'])
df['grade'] = le.fit_transform(df['grade'])
df['home_ownership'] = le.fit_transform(df['home_ownership'])
df['verification_status'] = le.fit_transform(df['verification_status'])
df['purpose'] = le.fit_transform(df['purpose'])
df['loan_status'] = le.fit_transform(df['loan_status'])
df['initial_list_status'] = le.fit_transform(df['initial_list_status'])
df['delinq_2yrs_cat'] = le.fit_transform(df['delinq_2yrs_cat'])
df['inq_last_6mths_cat'] = le.fit_transform(df['inq_last_6mths_cat'])
df['pub_rec_cat'] = le.fit_transform(df['pub_rec_cat'])
df


# In[ ]:


scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(df)


# In[ ]:


X_scaled = scale_my_dataset(df)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, int_rate, test_size=0.33, random_state=21)


# In[126]:


model = build_model()
model.summary()


# In[127]:



EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])


# In[128]:


plot_history(history)


# In[129]:


[loss, mae] = model.evaluate(X_test, y_test, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}%".format(mae))


# In[130]:


test_predictions = model.predict(X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


# In[131]:


error = test_predictions - y_test
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [%]")
plt.ylabel("Count")
_ = plt.xlim([-1,1])

