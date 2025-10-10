import numpy as np
import h5py
import rockverse as rv
import os

shape=(5,2,8)
a = rv.create_array(shape=shape, dtype=np.float32, chunks=(2,2,2))
a[...] = np.random.rand(*shape)
a.zarray[...]
a[...]

filename = os.path.join(os.getenv('USERPROFILE'), "Downloads", "test.h5")
path='/my/awesome/array'
self=a
self.h5_dump(filename, path='/my/awesome/array', mode='w')

with h5py.File(filename, 'r') as fobj:
    b = fobj['/my/awesome/array'][...]
    c = {k: v for k, v in fobj['/my/awesome/array'].attrs.items()}
b == a[...]
c
