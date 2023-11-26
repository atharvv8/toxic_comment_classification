#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


import warnings


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[5]:


pwd


# In[6]:


# %cd drive/MyDrive/dataset


# In[7]:


# get_ipython().run_line_magic('cd', os.getcwd())


# In[8]:


df=pd.read_csv('train.csv/train.csv')


# In[9]:


# df.head()


# In[10]:


# df.iloc[0]['comment_text']


# In[11]:


from tensorflow.keras.layers import TextVectorization


# In[12]:


# TextVectorization??


# In[13]:


# df.columns


# In[14]:


# df.columns[2:]


# In[15]:


X = df['comment_text']
y = df[df.columns[2:]].values


# In[16]:


# X


# In[17]:


# y


# In[18]:


MAX_WORDS = 200000 #NO OF WORDS IN VOCAB


# In[19]:


vectorizer = TextVectorization(max_tokens=MAX_WORDS,output_sequence_length=1800,output_mode='int')


# In[20]:


# type(X)


# In[21]:


# X.values #Comments represented as numpy array


# In[22]:


vectorizer.adapt(X.values) #teach our vector our vocabulary.
#Adapter is going to learn all words that are in our vocabulary


# In[23]:


vectorizer.get_vocabulary()


# In[24]:


vectorized_text = vectorizer(X.values)


# In[25]:


# vectorized_text


# In[26]:


# len(X)


# In[27]:


#MCSHBAP : MAP,CACHE, SHUFFLE ,BATCH, PREFETCH
#MAKING A DATA PIPELINE IN CASE LARGE DATA WONT FIT IN MEMORY
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)#helps prevent bottlenecks


# In[28]:


batch_X, batch_y = dataset.as_numpy_iterator().next()


# In[29]:


train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


# In[30]:


train_generator = train.as_numpy_iterator()


# In[31]:


train_generator.next()


# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding


# In[33]:


# model= Sequential()
#Create the embedding layer
# model.add(Embedding(MAX_WORDS+1, 32))
#Creating BidirectionalLSTM LAYER tanh because of gpu relu for normal
# model.add(Bidirectional(LSTM(32, activation='tanh')))
#Feature extractors fully connected layers
# model.add(Dense(128,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(128,activation='relu'))
#Final Layers
#6 Final units as of 6 labels
# model.add(Dense(6,activation='sigmoid'))


# In[34]:


# model.compile(loss='BinaryCrossentropy', optimizer='Adam')


# In[35]:

model = tf.keras.models.load_model('toxicity.h5')
model.summary()


# In[41]:


# history= model.fit(train,epochs=5,validation_data=val)


# In[42]:


# history.history


# In[43]:


# plt.figure(figsize=(8,5))
# pd.DataFrame(history.history).plot()
# plt.show()


# **Making Predictions**

# In[44]:


batch= test.as_numpy_iterator().next()


# In[45]:


batch_X, batch_y = dataset.as_numpy_iterator().next()


# In[46]:


# input_text=vectorizer('You Freaking Suck! I am going to hit you.')


# In[47]:


res = model.predict(batch_X)


# In[48]:


res.flatten()


# In[49]:


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


# In[50]:


pre = Precision()
re= Recall()
acc= CategoricalAccuracy()


# In[51]:


for batch in test.as_numpy_iterator():
    # Unpack the batch
    X_true, y_true = batch
    # Make a prediction
    yhat = model.predict(X_true)

    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


# In[52]:


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# In[5]:


# !pip install gradio jinja2


# In[55]:


# !pip install typing_extensions==3.10.0.0


# In[36]:


# import gradio as gr


# In[57]:


# model.save('toxicity.h5')


# In[37]:




# In[38]:


# input_str = vectorizer('hey i freaken hate you!')


# In[39]:


# res = model.predict(np.expand_dims(input_str,0))


# In[40]:


# print(res)


# In[41]:


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)

    return text


# In[49]:


# interface = gr.Interface(fn=score_comment,inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'), outputs='text')


# In[46]:


# interface.launch(share=True)


# In[56]:


choice="1"


# In[57]:


while choice!="0":
    print("Enter your comment")
    str=input()
    print(score_comment(str))
    print("Enter 0 to exit")
    choice=input()


# In[ ]:




