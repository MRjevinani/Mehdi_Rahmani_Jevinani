
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:47:01 2022

@author: 10
"""

#-----------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split

data =np.load('matrix_3d_1000000_re.npz')
lst = data.files
for item in lst:
    total_data=data[item]
    #print('total_data:\n',total_data)

#------------------------------------------------------------------------------

total_data = total_data[:,:,1:3]

total_data1 = total_data[:,:,1:2]

pi=22/7
total_data2 = (180/pi)*total_data1

total_data3 = total_data[:,:,0:1]

total_data4 = np.dstack((total_data3, total_data2))

#-----------------------------------------------------------------------------
total_data_log = np.log10(total_data4)

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

#-----------------------------------------------------------------------------

resistivity_3d = resistivity.reshape((resistivity.shape[0], resistivity.shape[1], 1))

thickness_3d = thickness.reshape((thickness.shape[0], thickness.shape[1], 1))

#------------------------------------------------------------------------------
Y_total = np.dstack((resistivity_3d, thickness_3d))

#-----------------------------------------------------------------------------

Y_total_log = np.log10(Y_total)

a=np.log10(6334)

Y_total_log_scaler = Y_total_log/np.log10(6334)

Y_total_log_rescaler = a*Y_total_log_scaler

Y_total_antilog = pow(10, Y_total_log_rescaler)

#-----------------------------------------------------------------------------

X_total_train, X_total_test, Y_total_train , Y_total_test  = train_test_split(total_data_log_scaler , Y_total_log_scaler , test_size = 0.01 ,random_state = 42)

#------------------------------------------------------------------------------

X_total_train = np.reshape(X_total_train, (len(X_total_train), 36, 2, 1))
print("X_total_train", X_total_train.shape)

X_total_test = np.reshape(X_total_test, (len(X_total_test), 36, 2, 1))
print("X_total_test", X_total_test.shape)

# اضافه کردن یک بعد به ماتریس سه بعدی داده های خروجی برای معماری کانولوشن
Y_total_train = np.reshape(Y_total_train, (len(Y_total_train), 5, 2, 1))
print("Y_total_train", Y_total_train.shape)


#-----------------------------------------------------------------------------

import keras
from keras import layers
from keras.models import  Model
from keras.layers import Dense , Flatten , ZeroPadding2D, BatchNormalization , Conv2D , MaxPool2D , Input , Cropping2D ,Conv2DTranspose, Dropout, UpSampling2D, concatenate
from keras.optimizers import SGD, Adam 
import tensorflow as tf
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import savez_compressed



my_input=Input(shape=(36,2,1), name = 'my_input')
conv1 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv1-1')(my_input)
conv1 = BatchNormalization()(conv1)
#conv1 = Dropout(0.5)(conv1)
conv1 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv1-2')(conv1)
conv1 = BatchNormalization()(conv1)
#conv1 = Dropout(0.3)(conv1)

conv1 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv1-3')(conv1)
conv1 = BatchNormalization()(conv1)
#conv1 = Dropout(0.1)(conv1)
conv1 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv1-4')(conv1)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPool2D(pool_size=(2, 1), name = 'pool1')(conv1)


conv2 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv2-1')(pool1)
conv2 = BatchNormalization()(conv2)
#conv2 = Dropout(0.5)(conv2)
conv2 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv2-2' )(conv2)
conv2 = BatchNormalization()(conv2)
#conv2 = Dropout(0.3)(conv2)

conv2 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv2-3')(conv2)
conv2 = BatchNormalization()(conv2)
#conv2 = Dropout(0.1)(conv2)
conv2 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv2-4' )(conv2)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPool2D(pool_size=(2, 1), name = 'pool2')(conv2)


conv3 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv3-1')(pool2)
conv3 = BatchNormalization()(conv3)
#conv3 = Dropout(0.5)(conv3)
conv3 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv3-2')(conv3)
conv3 = BatchNormalization()(conv3)
#conv3 = Dropout(0.3)(conv3)

conv3 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv3-3')(conv3)
conv3 = BatchNormalization()(conv3)
#conv3 = Dropout(0.1)(conv3)
conv3 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv3-4')(conv3)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPool2D(pool_size=(2, 1), name = 'pool3')(conv3)


conv4 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv4-1' )(pool3)
conv4 = BatchNormalization()(conv4)
#conv4 = Dropout(0.5)(conv4)
conv4 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv4-2')(conv4)
conv4 = BatchNormalization()(conv4)
#conv4 = Dropout(0.3)(conv4)

conv4 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv4-3' )(conv4)
conv4 = BatchNormalization()(conv4)
#conv4 = Dropout(0.1)(conv4)
conv4 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv4-4')(conv4)
conv4 = BatchNormalization()(conv4)
pool4 = MaxPool2D(pool_size=(2, 1), name = 'pool4')(conv4)


conv5 = Conv2D(512, (3, 2), activation='relu', padding='same', name = 'conv5-1')(pool4)
conv5 = BatchNormalization()(conv5)
#conv5 = Dropout(0.5)(conv5)
conv5 = Conv2D(512, (3, 2), activation='relu', padding='same', name = 'conv5-2')(conv5)
conv5 = BatchNormalization()(conv5)
#conv5 = Dropout(0.3)(conv5)

conv5 = Conv2D(512, (3, 2), activation='relu', padding='same', name = 'conv5-3')(conv5)
conv5 = BatchNormalization()(conv5)
#conv5 = Dropout(0.1)(conv5)
conv5 = Conv2D(512, (3, 2), activation='relu', padding='same', name = 'conv5-4')(conv5)
conv5 = BatchNormalization()(conv5)


up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 1), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-1' )(up6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-2')(conv6)
conv6 = BatchNormalization()(conv6)

conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-3' )(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-4')(conv6)
conv6 = BatchNormalization()(conv6)


up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 1), padding='same')(conv6), Cropping2D(cropping=((1,0),(0,0)))(conv3)], axis=3)
conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-1' )(up7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-2')(conv7)
conv7 = BatchNormalization()(conv7)

conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-3' )(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-4')(conv7)
conv7 = BatchNormalization()(conv7)


up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 1), padding='same')(conv7), Cropping2D(cropping=((1,1),(0,0)))(conv2)], axis=3)
conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(up8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-2',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv8)
conv8 = BatchNormalization()(conv8)

conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-3',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-4',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv8)
conv8 = BatchNormalization()(conv8)


up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 1), padding='same')(conv8), Cropping2D(cropping=((2,2),(0,0)))(conv1)], axis=3)
conv9 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv9-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(up9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv9-2',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv9)
conv9 = BatchNormalization()(conv9)

conv9 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv9-3',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(32, (3, 2), activation='relu', padding='same', name = 'conv9-4',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv9)
conv9 = BatchNormalization()(conv9)

conv10 = Conv2D(1, (1, 1), activation='relu',name = 'conv10' )(conv9)

out_layer = Cropping2D(cropping = ((13,14),(0,0)))(conv10)


model = Model(inputs=[my_input], outputs=[out_layer])
model.summary()

from keras.utils import plot_model
plot_model(model , to_file = 'model_U-net.pdf',show_shapes=True)

#------------------------------------------------
n_epochs = 500
learning_rate = 0.01
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

model.save('model_u_net.h5')

model.save_weights('model_weights_u_net.h5')

#-----------------------------------------------------------------------------

Y_hat_predict=model.predict(X_total_test)

print("Y_hat_predict", Y_hat_predict.shape)

#-----------------------------------------------------------------------------


Y_hat_predict_w_o = np.reshape(Y_hat_predict, (len(Y_hat_predict), 5, 2))
print("Y_hat_predict_w_o", Y_hat_predict_w_o.shape)

#-----------------------------------------------------------------------------
# برگرداندن یا rescale کردن داده های Y_total_test
Y_total_test_log_rescaler = a*Y_total_test
Y_total_test_antilog = pow(10, Y_total_test_log_rescaler)


# ذخیره ماتریس Y_total_test_antilog در فرمت NPZ
savez_compressed('Y_total_test_antilog.npz', Y_total_test_antilog)

#-----------------------------------------------------------------------------
Y_hat_predict_log_rescaler = a*Y_hat_predict_w_o
Y_hat_predict_antilog = pow(10, Y_hat_predict_log_rescaler)


# ذخیره ماتریس Y_hat_predict_antilog در فرمت NPZ
savez_compressed('Y_hat_predict_antilog.npz', Y_hat_predict_antilog)


