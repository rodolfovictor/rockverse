import os
import rockverse as rv
import numpy as np
self = rv.core.create_tensorfield(
    shape=(5, 2, 8),
    tensor_shape=(3, 3),
    chunks=(2,2,2),
    dtype=float,
    #data = np.random.rand(5,2,8),
    store=os.path.join(os.getenv('USERPROFILE'), "Downloads", "test"),
    path="testpath",
    name='test array',
    unit='m/s',
    description="UMA DESC",
    latex_name=r"$ERF$",
    latex_unit="MM",
    coord_data=([1, 2, 4, 7, 9], [2, 2], None),
    coord_names=("QQ", 'y','z'),
    coord_units=('km', "S", "F"),
    coord_descriptions=("UM", "DOIS", "WW"),
    coord_latex_names=(r"$r$", r"$i$", r"$p$"),
    coord_latex_units=('a', '', '.'),
    overwrite=True)
self.validate()
self.components[0, 0][...] = np.random.randn(5,2,8)
self.components[0, 1][...] = np.random.randn(5,2,8)

filename = os.path.join(os.getenv('USERPROFILE'), "Downloads", "test.h5")
self.h5_dump(filename, path='/my/awesome/array', mode='w')

#store=os.path.join(os.getenv('USERPROFILE'), "Downloads", "test2.zarr")
#h5path = '/myawesomearray'
#path=None
#overwrite=True
#kwargs={}
#with h5py.File(filename, mode='r') as fobj:
#    self2 = load_array_from_h5_file(fobj, h5path, store, path=None, overwrite=True)

#self.components[0]
#self.components[0].zarray[...]
#self.components[0][...]
