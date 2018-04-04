
# coding: utf-8

# In[18]:


#Load the required Libraries
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy
import pickle


# In[28]:


#load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255


# In[20]:


#declare the model
model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu',padding='same',input_shape=(x_train[0].shape[0], x_train[0].shape[1],3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='sigmoid'))


# In[29]:


#the hyperparameters
batch_size = 2500
epochs = 20


# In[30]:


#compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[31]:


#start the training
model.fit(x_train,to_categorical(y_train,num_classes=10),epochs=epochs,batch_size=batch_size)

