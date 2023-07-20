
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:47:01 2022

@author: 10
"""

#-----------------------------------------------------------------------------
# فراخوانی و آماده سازی داده ها ورودی

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
from keras.models import  Model
from keras.layers import Dense , Flatten, LeakyReLU, Lambda , BatchNormalization , Conv2D , MaxPool2D , Input , Cropping2D ,Conv2DTranspose, Dropout, UpSampling2D, concatenate, Reshape
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from numpy import savez_compressed
from keras.backend import int_shape, random_normal, shape, exp, mean, square, sum

import numpy as np
import matplotlib.pyplot as plt


my_input = Input(shape=(36,2,1), name = 'my_input')
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

shape_before_flatten = int_shape(conv5)[1:]
encoder_flatten = Flatten()(conv5)


latent_space_dim = 2


encoder_mu = Dense(units=latent_space_dim, name="encoder_mu")(encoder_flatten)
encoder_log_variance = Dense(units=latent_space_dim, name="encoder_log_variance")(encoder_flatten)

encoder_mu_log_variance_model = Model(my_input, (encoder_mu, encoder_log_variance), name="encoder_mu_log_variance_model")

def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = random_normal(shape=shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + exp(log_variance/2) * epsilon
    return random_sample

encoder_output = Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])

encoder = Model(my_input, encoder_output, name="encoder_model")

encoder.summary()

#Building the Decoder

decoder_input = Input(shape=(latent_space_dim,), name="decoder_input")

decoder_dense_layer1 = Dense(units=2048, name="decoder_dense_1")(decoder_input)

decoder_reshape = Reshape((2,2,512))(decoder_dense_layer1)


up6 = Conv2DTranspose(256, (2, 2), strides=(2, 1), padding='same')(decoder_reshape)
conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-1' )(up6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-2')(conv6)
conv6 = BatchNormalization()(conv6)

conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-3' )(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(256, (3, 2), activation='relu', padding='same', name = 'conv6-4')(conv6)
conv6 = BatchNormalization()(conv6)

up7 = Conv2DTranspose(128, (2, 2), strides=(2, 1), padding='same')(conv6)
conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-1' )(up7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-2')(conv7)
conv7 = BatchNormalization()(conv7)

conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-3' )(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(128, (3, 2), activation='relu', padding='same', name = 'conv7-4')(conv7)
conv7 = BatchNormalization()(conv7)


up8 = Conv2DTranspose(64, (2, 2), strides=(2, 1), padding='same')(conv7)
conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-1',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(up8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-2',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv8)
conv8 = BatchNormalization()(conv8)

conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-3',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(64, (3, 2), activation='relu', padding='same', name = 'conv8-4',kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.001))(conv8)
conv8 = BatchNormalization()(conv8)


up9 = Conv2DTranspose(32, (2, 2), strides=(2, 1), padding='same')(conv8)
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

decoder_output = out_layer

decoder = Model(inputs = decoder_input, outputs = decoder_output , name="decoder_model")

decoder.summary()


#Building the VAE
vae_input = Input(shape=(36, 2, 1), name="VAE_input")

vae_encoder_output = encoder(vae_input)

vae_decoder_output = decoder(vae_encoder_output)

vae = Model(vae_input, vae_decoder_output, name="VAE")

vae.summary()

#------------------------------------------------------------------------

# compile of model 
# The implementation of the loss function is given below

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = mean(square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * sum(1.0 + encoder_log_variance - square(encoder_mu) - exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * sum(1.0 + encoder_log_variance - square(encoder_mu) - exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss
#------------------------------------------------
n_epochs = 500
learning_rate = 0.0005
decay_rate = learning_rate / n_epochs
momentum = 0.8
sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
vae.compile(optimizer=sgd , loss='mse' , metrics=['accuracy'])


#------------------------------------------------------------------------------

# learning schedule callback
eary_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#------------------------------------------------------------------------------
import datetime
start = datetime.datetime.now()


network_history = vae.fit(X_total_train, Y_total_train,
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

#-----------------------------------------------------------------------------

# ذحیره ساختار مدل و وزن های آموزش دیده شده
vae.save('model_vae.h5')

# ذخیره وزن های مدل
vae.save_weights('model_weights_vae.h5')

#-----------------------------------------------------------------------------
# پیش بینی مدل

Y_hat_predict=vae.predict(X_total_test)

print("Y_hat_predict", Y_hat_predict.shape)

#-----------------------------------------------------------------------------

# برگرداندن Y_hat_predict به مفادیر اصلی با دستور INVERSE
# بعد چهارم برای مقایسه با Y_test از بین می بریم

Y_hat_predict_w_o = np.reshape(Y_hat_predict, (len(Y_hat_predict), 5, 2))
print("Y_hat_predict_w_o", Y_hat_predict_w_o.shape)

#-----------------------------------------------------------------------------
# برگرداندن یا rescale کردن داده های Y_total_test
Y_total_test_log_rescaler = a*Y_total_test
Y_total_test_antilog = pow(10, Y_total_test_log_rescaler)


# ذخیره ماتریس Y_total_test_antilog در فرمت NPZ
savez_compressed('Y_total_test_antilog.npz', Y_total_test_antilog)

#-----------------------------------------------------------------------------
# برگرداندن یا rescal کردن داده های Y_hat_predict
Y_hat_predict_log_rescaler = a*Y_hat_predict_w_o
Y_hat_predict_antilog = pow(10, Y_hat_predict_log_rescaler)


# ذخیره ماتریس Y_hat_predict_antilog در فرمت NPZ
savez_compressed('Y_hat_predict_antilog.npz', Y_hat_predict_antilog)

