import numpy as np
import h5py


# f = h5py.File('cofw-mats\COFW_train.mat','r') #<KeysViewHDF5 ['#refs#', 'IsTr', 'bboxesTr', 'phisTr']>
# ds = f.get('IsTr')
# print(type(ds))
#
# name = h5py.h5r.get_name(ds.ref, f.id)
# data = f[name].value
# print(data)
# print(type(data))
#
# data = np.array(data) # For converting to a NumPy array
# print(data.shape)
# print(type(data))
# print(data[0])

def get_name(index, hdf5_data):
    name = hdf5_data['IsTr']
    print(''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value]))


labels_file = 'cofw-mats\COFW_train.mat'
f = h5py.File(labels_file)
for j in range(33402):
    get_name(j, f)
