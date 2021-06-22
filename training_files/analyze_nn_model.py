import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl use, Otherwise mcenv breaks
mpl.use('Agg')

from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *
from pyvibdmc.simulation_utilities.tensorflow_descriptors.cupy_distance import DistIt

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow.keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cupy as cp

# GPU and tensorflow settings 
tf.keras.backend.set_floatx('float64')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Load in data for analysis
train = np.load("train_xy.npz")
train_x = train['train_x']
train_y = train['train_y']

val = np.load("val_xy.npz")
val_x = val['val_x']
val_y = val['val_y']

test = np.load("test_xy.npz")
test_x = test['test_x']
test_y = test['test_y']

#Transform your *_x data to the appropriate descriptor
train_spf = ...
val_spf = ...
test_spf = tf.convert_to_tensor(...)

#Define mae metric for monitoring training+validation while training
def mae(y_true, y_pred):
    return K.mean(K.abs((y_true)-(y_pred)))

# Load finished model
model = tf.keras.models.load_model('...')

# Predict based on energies
v_train = model.predict(train_spf).flatten()
v_val = model.predict(val_spf).flatten()
v_test = model.predict(test_spf).flatten()

# Convert to cm-1
pred_v_train = (10**(v_train)-1)*1000
pred_v_val = (10**(v_val)-1)*1000
pred_v_test = (10**(v_test)-1)*1000

# MAE of different data sets
print(mae(cp.asnumpy(train_y),cp.asnumpy(pred_v_train)))
print(mae(cp.asnumpy(val_y),cp.asnumpy(pred_v_val)))
print(mae(cp.asnumpy(test_y),cp.asnumpy(pred_v_test)))

# Diagonal line plots
plt.scatter(train_y,pred_v_train,label='training data')
plt.legend()
plt.xlabel('MB-pol E')
plt.ylabel('Predicted E')
plt.savefig('training_corr.png',dpi=300,bbox_inches='tight')
plt.close()
plt.scatter(val_y,pred_v_val,label='validation data')
plt.legend()
plt.xlabel('MB-pol E')
plt.ylabel('Predicted E')
plt.savefig('val_corr.png',dpi=300,bbox_inches='tight')
plt.close()
plt.scatter(test_y,pred_v_test,label='test data')
plt.legend()
plt.xlabel('MB-pol E')
plt.ylabel('Predicted E')
plt.savefig('test_corr.png',dpi=300,bbox_inches='tight')
plt.close()
