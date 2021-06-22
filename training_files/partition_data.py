import numpy as np
import matplotlib.pyplot as plt
from pyvibdmc.analysis import *
from pyvibdmc.simulation_utilities import *

def get_data_set(flz):
    tot_cds = []
    tot_engs = []
    for fll in flz:
        # SimInfo.get_training returns data in angstroms and cm-1
        cds,engs = SimInfo.get_training(training_file=fll)
        tot_cds.append(cds)
        tot_engs.append(engs)
    tot_cds = np.concatenate(tot_cds)
    tot_engs = np.concatenate(tot_engs)
    return tot_cds, tot_engs

flpath = '/gscratch/ilahie/rjdiri/dmc/trimer_training/hex_training_data'

# Everything remains in Cartesian coordinates until training

train_inds = np.concatenate((np.arange(50, 4000, 100),np.arange(4000, 8000, 50)))
val_inds = np.concatenate((np.arange(70,4070,200),np.arange(4070,8070,100)))
test_inds = np.arange(3025, 4025, 200)

train_fls = [f'{flpath}/tri_dt10_0_training_{x}ts.hdf5' for x in train_inds]
train_x,train_y = get_data_set(train_fls)
print(train_x.shape,train_y.shape)
np.savez("train_xy.npz",train_x=train_x,train_y=train_y)

val_fls = [f'{flpath}/tri_dt10_0_training_{x}ts.hdf5' for x in val_inds]
val_x,val_y = get_data_set(val_fls)
print(val_x.shape,val_y.shape)
np.savez("val_xy.npz",val_x=val_x,val_y=val_y)

test_fls = [f'{flpath}/tri_dt10_0_training_{x}ts.hdf5' for x in test_inds]
test_x,test_y = get_data_set(test_fls)
print(test_x.shape,test_y.shape)
np.savez("test_xy.npz",test_x=test_x,test_y=test_y)

print('done!')

# Look at energy distribution of data
plt.hist(train_y,bins=1000,range=(0,100000),label='train_y')
plt.hist(val_y,bins=1000,range=(0,100000),label='val_y')
plt.hist(test_y,bins=1000,range=(0,100000),label='test_y')
plt.legend()
plt.savefig('energy_histogram.png',dpi=400)
plt.close()
