import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

tf.keras.backend.set_floatx('float64')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#Load in training and validation set - Load in test data later
train = np.load("train_xy.npz")
train_x = train['train_x']
train_y = train['train_y']
val = np.load("val_xy.npz")
val_x = val['val_x']
val_y = val['val_y']

test = np.load("test_xy.npz")
test_x = test['test_x']
test_y = test['test_y']

#Transform your data to the appropriate descriptor
eq_geom = np.load("/gscratch/ilahie/rjdiri/dmc/trimer_training/udu_bohr.npy")
eq_geom = Constants.convert(eq_geom,'angstroms',to_AU=False)
eq_geom = np.array([eq_geom,eq_geom])
dist = DistIt(zs=[8,1,1]*3, sort_mat=False)

req = dist.run(cp.array(eq_geom))[0]
print(req)

train_r = dist.run(cp.array(train_x))
train_spf = tf.convert_to_tensor(cp.asnumpy((train_r - req) / train_r))

val_r = dist.run(cp.array(val_x))
val_spf = tf.convert_to_tensor(cp.asnumpy((val_r - req) / val_r))

test_r = dist.run(cp.array(test_x))
test_spf = tf.convert_to_tensor(cp.asnumpy((test_r - req) / test_r))

#Define mae metric for monitoring training+validation while training
def mae(y_true, y_pred):
    return K.mean(K.abs((y_true)-(y_pred)))

model = tf.keras.models.load_model('trimer_model')
v_train = model.predict(train_spf).flatten()
v_val = model.predict(val_spf).flatten()
v_test = model.predict(test_spf).flatten()
pred_v_train = (10**(v_train)-1)*1000
pred_v_val = (10**(v_val)-1)*1000
pred_v_test = (10**(v_test)-1)*1000
print(train_y)
print(pred_v_train)
print(val_y)
print(pred_v_val)
print(test_y)
print(pred_v_test)

print(mae(cp.asnumpy(train_y),cp.asnumpy(pred_v_train)))
print(mae(cp.asnumpy(val_y),cp.asnumpy(pred_v_val)))
print(mae(cp.asnumpy(test_y),cp.asnumpy(pred_v_test)))

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
