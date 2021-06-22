import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *
from pyvibdmc.simulation_utilities.tensorflow_descriptors.cupy_distance import DistIt

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow.keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cupy as cp

# Set tensorflow GPU settings
tf.keras.backend.set_floatx('float64')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Load in training and validation set - Load in test data later
train = np.load("train_xy.npz")
train_x = train['train_x']
num_atoms = len(train_x[0])
train_y = train['train_y']
val = np.load("val_xy.npz")
val_x = val['val_x']
val_y = val['val_y']

# For NN number of layers
num_atoms = len(train_x[0])

#Transform your data to the appropriate descriptor

train_spf = tf.convert_to_tensor(...)
val_spf = tf.convert_to_tensor(...)

# Transforming energy !
train_y = np.log10(train_y/1000+1)
val_y = np.log10(val_y/1000+1)

print('~~~~~~~~SHAPES~~~~~~~~~')
print(train_spf.shape)
print(train_y.shape)
print(val_spf.shape)
print(val_y.shape)
print('~~~~~~~~SHAPES~~~~~~~~~')

#Define mae metric for monitoring training+validation while training
def mae(y_true, y_pred):
    return K.mean(K.abs( (10**(y_true)-1)*1000 - (10**(y_pred)-1)*1000) )
   
print('input layer shape')
print(len(train_spf[0]))
print('input layer shape')
# NN structure
model = tf.keras.Sequential(
[
    InputLayer(input_shape=(len(train_spf[0]),)),
    Dense(10*(3*num_atoms-6), activation=tf.nn.swish, use_bias=False),
    Dense(10*(3*num_atoms-6), activation=tf.nn.swish), use_bias=False),
    Dense(10*(3*num_atoms-6), activation=tf.nn.swish, use_bias=False),
    Dense(1, activation='relu', use_bias=False) # Should we bias the output node as well?
]
)

# Callbacks to automate training
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# Reduce the learning rate once the change in the mae is < 0.5 cm-1 for 5 consecutive epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='mae', 
                                                 factor=0.5,
                                                 patience=5,
                                                 min_delta=0.5,
                                                 min_lr=0.001/1024)

# Checkpoints and saves the model's weights after each epoch
model_chkpt = tf.keras.callbacks.ModelCheckpoint('train.{epoch:02d}-{mae:.2f}.hdf5', 
                                                 monitor='mae',
                                                 verbose=1, 
                                                 save_best_only=False,
                                                 save_weights_only=True,
                                                 save_freq='epoch')

# Initial learning rate that will decrease throughout training
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001/4)
lr_metric = get_lr_metric(optimizer)

# If you are restarting, load the weights here
# model.load_weights('...')

# Compile the model
model.compile(optimizer=optimizer,
                 loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[mae,lr_metric])

# Train the model. For now, this goes for 100 epochs regardless of learning rate / error.
# May add a Custom callback in the future to stop when lr = final and mae hasn't changed for 5 epochs
# https://www.tensorflow.org/guide/keras/custom_callback#examples_of_keras_callback_applications
data = model.fit(x=train_spf,
                 y=train_y,
                 epochs=100,
                 validation_data=(val_spf,val_y),
                 batch_size=32,
                 validation_batch_size=32,
                 shuffle=True,
                 callbacks = [reduce_lr,model_chkpt]
                )

# Dump data.history to a pickle file, saves the mae and lr and stuff. If the model crashes,
# change this to history_2.pickle
with open('history_1.pickle', 'wb') as handle:
    pickle.dump(data.history, handle, protocol=4)  

# Save the model at the end of training
tf.keras.models.save_model(model,'example_model_name')
