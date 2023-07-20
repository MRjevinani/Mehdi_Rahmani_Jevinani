

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:26:25 2023

@author: 10
"""

#-----------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data =np.load('matrix_3d_1000000_re.npz')
lst = data.files
for item in lst:
    total_data_in=data[item]

#------------------------------------------------------------------------------

total_data_re = total_data_in[:,:,1:2]

total_data_ph = total_data_in[:,:,2:3]

pi=22/7
total_data_ph = (180/pi)*total_data_ph


total_data_re_ph = np.concatenate([total_data_re, total_data_ph] , axis=1)

total_data = total_data_re_ph

#-----------------------------------------------------------------------------
total_data_log = np.log10(total_data)

c=np.log10(460)

total_data_log_scaler = total_data_log/np.log10(460)

total_data_log_rescaler = c*total_data_log_scaler

total_data_antilog = pow(10, total_data_log_rescaler)

#-----------------------------------------------------------------------------

data1 =np.load('resistivity_1000000.npz')
lst1 = data1.files
for item in lst1:
    resistivity = data1[item]

data2 =np.load('thickness_1000000.npz')
lst2 = data2.files
for item in lst2:
    thickness1 = data2[item]


thickness2 = thickness1.sum(axis=1) 
thickness3 = 10000 - thickness2
thickness4 = thickness3.reshape(len(thickness3), 1)
thickness = np.hstack((thickness1, thickness4))

Y_total_re_th = np.concatenate([resistivity, thickness] , axis=1)

#-----------------------------------------------------------------------------

Y_total_re_th_3d = Y_total_re_th.reshape((Y_total_re_th.shape[0], Y_total_re_th.shape[1], 1))

#------------------------------------------------------------------------------
Y_total = Y_total_re_th_3d

print("Y_total", Y_total.shape)

#-----------------------------------------------------------------------------

Y_total_log = np.log10(Y_total)

a=np.log10(6334)

Y_total_log_scaler = Y_total_log/np.log10(6334)

Y_total_log_rescaler = a*Y_total_log_scaler

Y_total_antilog = pow(10, Y_total_log_rescaler)

#-----------------------------------------------------------------------------

X_total_train, X_total_test, Y_total_train , Y_total_test  = train_test_split(total_data_log_scaler , Y_total_log_scaler , test_size = 0.01 ,random_state = 42)

#------------------------------------------------------------------------------

X_total_train = np.reshape(X_total_train, (len(X_total_train), 72))
print("X_total_train", X_total_train.shape)

X_total_test = np.reshape(X_total_test, (len(X_total_test), 72))
print("X_total_test", X_total_test.shape)

Y_total_train = np.reshape(Y_total_train, (len(Y_total_train), 10))
print("Y_total_train", Y_total_train.shape)

Y_total_test = np.reshape(Y_total_test, (len(Y_total_test), 10))
print("Y_total_test", Y_total_test.shape)


#-----------------------------------------------------------------------------

import keras
from keras import layers
from keras.models import  Model
from keras.layers import Dense , Flatten, add, Concatenate , Activation, ZeroPadding2D, BatchNormalization , Conv2D , MaxPool2D , Input , Cropping2D ,Conv2DTranspose, Dropout, UpSampling2D, concatenate
from keras.optimizers import SGD, Adam
import tensorflow as tf
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from numpy import savez_compressed


my_input=Input(shape=(72,), name = 'my_input')
#-----------------------
# Dense Bloke 1

merge_input1 = my_input
# check if the number of filters needs to be increase, assumes channels last format
if my_input.shape[-1] != 64:
		merge_input1 = Dense(64)(my_input)
merge_input1 = BatchNormalization()(merge_input1)


dense1 = Dense(64, name = 'dense1-1',kernel_regularizer=l2(0.0001),  bias_regularizer=l2(0.001))(my_input)
dense1 = BatchNormalization()(dense1)
dense1 = Activation('relu')(dense1)

dense1 = Dense(64, name = 'dense1-2',)(dense1)
dense1 = BatchNormalization()(dense1)
dense1 = Activation('relu')(dense1)

dense1 = Dense(64, name = 'dense1-3',)(dense1)
dense1 = BatchNormalization()(dense1)

layer_out1 = add([dense1, merge_input1])
layer_out1 = Activation('relu')(layer_out1)
#-------------------
# Identity Bloke 1-1
merge_input2 = layer_out1
# check if the number of filters needs to be increase, assumes channels last format
if layer_out1.shape[-1] != 64:
		merge_input2 = Dense(64)(layer_out1)


dense2 = Dense(64, name = 'dense2-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out1)
dense2 = BatchNormalization()(dense2)
dense2 = Activation('relu')(dense2)

dense2 = Dense(64, name = 'dense2-2',)(dense2)
dense2 = BatchNormalization()(dense2)
dense2 = Activation('relu')(dense2)

dense2 = Dense(64, name = 'dense2-3',)(dense2)
dense2 = BatchNormalization()(dense2)

layer_out2 = add([dense2, merge_input2])
layer_out2 = Activation('relu')(layer_out2)
#---------------------
# Identity Bloke 1-2
merge_input3 = layer_out2
# check if the number of filters needs to be increase, assumes channels last format
if layer_out2.shape[-1] != 64:
		merge_input3 = Dense(64)(layer_out2)


dense3 = Dense(64, name = 'dense3-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out2)
dense3 = BatchNormalization()(dense3)
dense3 = Activation('relu')(dense3)

dense3 = Dense(64, name = 'dense3-2',)(dense3)
dense3 = BatchNormalization()(dense3)
dense3 = Activation('relu')(dense3)

dense3 = Dense(64, name = 'dense3-3',)(dense3)
dense3 = BatchNormalization()(dense3)

layer_out3 = add([dense3, merge_input3])
layer_out3 = Activation('relu')(layer_out3)
#-------------------------
# Dense Bloke 2

merge_input4 = layer_out3
# check if the number of filters needs to be increase, assumes channels last format
if layer_out3.shape[-1] != 128:
		merge_input4 = Dense(128)(layer_out3)
merge_input4 = BatchNormalization()(merge_input4)

dense4 = Dense(128, name = 'dense4-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out3)
dense4 = BatchNormalization()(dense4)
dense4 = Activation('relu')(dense4)

dense4 = Dense(128, name = 'dense4-2',)(dense4)
dense4 = BatchNormalization()(dense4)
dense4 = Activation('relu')(dense4)

dense4 = Dense(128, name = 'dense4-3',)(dense4)
dense4 = BatchNormalization()(dense4)

layer_out4 = add([dense4, merge_input4])
layer_out4 = Activation('relu')(layer_out4)
#-------------------------------------
# Identity Bloke 2-1

merge_input5 = layer_out4
# check if the number of filters needs to be increase, assumes channels last format
if layer_out4.shape[-1] != 128:
		merge_input5 = Dense(128)(layer_out4)


dense5 = Dense(128, name = 'dense5-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out4)
dense5 = BatchNormalization()(dense5)
dense5 = Activation('relu')(dense5)

dense5 = Dense(128, name = 'dense5-2',)(dense5)
dense5 = BatchNormalization()(dense5)
dense5 = Activation('relu')(dense5)

dense5 = Dense(128, name = 'dense5-3',)(dense5)
dense5 = BatchNormalization()(dense5)

layer_out5 = add([dense5, merge_input5])
layer_out5 = Activation('relu')(layer_out5)
#-------------------------------------
# Identity Bloke 2-2

merge_input6 = layer_out5
# check if the number of filters needs to be increase, assumes channels last format
if layer_out5.shape[-1] != 128:
		merge_input6 = Dense(128)(layer_out5)


dense6 = Dense(128, name = 'dense6-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out5)
dense6 = BatchNormalization()(dense6)
dense6 = Activation('relu')(dense6)

dense6 = Dense(128, name = 'dense6-2',)(dense6)
dense6 = BatchNormalization()(dense6)
dense6 = Activation('relu')(dense6)

dense6 = Dense(128, name = 'dense6-3',)(dense6)
dense6 = BatchNormalization()(dense6)

layer_out6 = add([dense6, merge_input6])
layer_out6 = Activation('relu')(layer_out6)
#----------------------------------
# Dense Bloke 3

merge_input7 = layer_out6
# check if the number of filters needs to be increase, assumes channels last format
if layer_out6.shape[-1] != 256:
		merge_input7 = Dense(256)(layer_out6)
merge_input7 = BatchNormalization()(merge_input7)

dense7 = Dense(256, name = 'dense7-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out6)
dense7 = BatchNormalization()(dense7)
dense7 = Activation('relu')(dense7)

dense7 = Dense(256, name = 'dense7-2',)(dense7)
dense7 = BatchNormalization()(dense7)
dense7 = Activation('relu')(dense7)

dense7 = Dense(256, name = 'dense7-3',)(dense7)
dense7 = BatchNormalization()(dense7)

layer_out7 = add([dense7, merge_input7])
layer_out7 = Activation('relu')(layer_out7)
#------------------------------------
# Identity Bloke 3-1

merge_input8 = layer_out7
# check if the number of filters needs to be increase, assumes channels last format
if layer_out7.shape[-1] != 256:
		merge_input8 = Dense(256)(layer_out7)


dense8 = Dense(256, name = 'dense8-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out7)
dense8 = BatchNormalization()(dense8)
dense8 = Activation('relu')(dense8)

dense8 = Dense(256, name = 'dense8-2',)(dense8)
dense8 = BatchNormalization()(dense8)
dense8 = Activation('relu')(dense8)

dense8 = Dense(256, name = 'dense8-3',)(dense8)
dense8 = BatchNormalization()(dense8)

layer_out8 = add([dense8, merge_input8])
layer_out8 = Activation('relu')(layer_out8)
#---------------------------------
# Identity Bloke 3-2

merge_input9 = layer_out8
# check if the number of filters needs to be increase, assumes channels last format
if layer_out8.shape[-1] != 256:
		merge_input9 = Dense(256)(layer_out8)


dense9 = Dense(256, name = 'dense9-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out8)
dense9 = BatchNormalization()(dense9)
dense9 = Activation('relu')(dense9)

dense9 = Dense(256, name = 'dense9-2',)(dense9)
dense9 = BatchNormalization()(dense9)
dense9 = Activation('relu')(dense9)

dense9 = Dense(256, name = 'dense9-3',)(dense9)
dense9 = BatchNormalization()(dense9)

layer_out9 = add([dense9, merge_input9])
layer_out9 = Activation('relu')(layer_out9)
#-----------------------------------
# Dense Bloke 4

merge_input10 = layer_out9
# check if the number of filters needs to be increase, assumes channels last format
if layer_out9.shape[-1] != 512:
		merge_input10 = Dense(512)(layer_out9)
merge_input10 = BatchNormalization()(merge_input10)

dense10 = Dense(512, name = 'dense10-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out9)
dense10 = BatchNormalization()(dense10)
dense10 = Activation('relu')(dense10)

dense10 = Dense(512, name = 'dense10-2',)(dense10)
dense10 = BatchNormalization()(dense10)
dense10 = Activation('relu')(dense10)

dense10 = Dense(512, name = 'dense10-3',)(dense10)
dense10 = BatchNormalization()(dense10)

layer_out10 = add([dense10, merge_input10])
layer_out10 = Activation('relu')(layer_out10)
#---------------------------------
# Identity Bloke 4-1

merge_input11 = layer_out10
# check if the number of filters needs to be increase, assumes channels last format
if layer_out10.shape[-1] != 512:
		merge_input11 = Dense(512)(layer_out10)


dense11 = Dense(512, name = 'dense11-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out10)
dense11 = BatchNormalization()(dense11)
dense11 = Activation('relu')(dense11)

dense11 = Dense(512, name = 'dense11-2',)(dense11)
dense11 = BatchNormalization()(dense11)
dense11 = Activation('relu')(dense11)

dense11 = Dense(512, name = 'dense11-3',)(dense11)
dense11 = BatchNormalization()(dense11)

layer_out11 = add([dense11, merge_input11])
layer_out11 = Activation('relu')(layer_out11)
#---------------------------------
# Identity Bloke 4-2


merge_input12 = layer_out11
# check if the number of filters needs to be increase, assumes channels last format
if layer_out11.shape[-1] != 512:
		merge_input12 = Dense(512)(layer_out11)


dense12 = Dense(512, name = 'dense12-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(layer_out11)
dense12 = BatchNormalization()(dense12)
dense12 = Activation('relu')(dense12)

dense12 = Dense(512, name = 'dense12-2',)(dense12)
dense12 = BatchNormalization()(dense12)
dense12 = Activation('relu')(dense12)

dense12 = Dense(512, name = 'dense12-3',)(dense12)
dense12 = BatchNormalization()(dense12)

layer_out12 = add([dense12, merge_input12])
layer_out12 = Activation('relu')(layer_out12)
#-----------------------------------
# output layer

dense13 = Dense(256, name = 'dense13')(layer_out12)

dense14 = Dense(128, name = 'dense14')(dense13)

dense15 = Dense(10, name = 'dense15')(dense14)


model = Model(inputs=[my_input], outputs=[dense15])
model.summary()


from keras.utils import plot_model
plot_model(model , to_file = 'model_resnet.pdf',show_shapes=True)

#------------------------------------------------
n_epochs = 500
learning_rate = 0.001
decay_rate = learning_rate / n_epochs
momentum = 0.8
sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer=sgd , loss='mse' , metrics=['accuracy'])
            
#------------------------------------------------------------------------------

# learning schedule callback
eary_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#------------------------------------------------------------------------------
import datetime
start = datetime.datetime.now()

network_history = model.fit(X_total_train, Y_total_train,
                epochs=n_epochs,
                batch_size=512,
                shuffle=True,
                validation_split=0.01,
                callbacks=[eary_stopping])

history=network_history.history

import matplotlib.pyplot as plt

losses = history['loss']
val_losses = history['val_loss']
accuraceis = history['accuracy']
val_accuraceis = history['val_accuracy']

plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(losses)
plt.plot(val_losses)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('epochs')
plt.ylabel('accuraceis')
plt.plot(accuraceis)
plt.plot(val_accuraceis)
plt.legend(['accuraceis','val_accuraceis'])
plt.figure()

end = datetime.datetime.now()
elapsed = end - start
print('total traning time : ', str(elapsed))

model.save('model_resnet.h5')

model.save_weights('model_weights_resnet.h5')

#-----------------------------------------------------------------------------

Y_hat_predict=model.predict(X_total_test)

print("Y_hat_predict", Y_hat_predict.shape)

#-----------------------------------------------------------------------------
Y_hat_predict_log_rescaler = a*Y_hat_predict
Y_hat_predict_antilog = pow(10, Y_hat_predict_log_rescaler)


savez_compressed('Y_hat_predict_antilog.npz', Y_hat_predict_antilog)

#-----------------------------------------------------------------------------
Y_total_test_log_rescaler = a*Y_total_test
Y_total_test_antilog = pow(10, Y_total_test_log_rescaler)


savez_compressed('Y_total_test_antilog.npz', Y_total_test_antilog)

