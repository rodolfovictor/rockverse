import rockverse as rv
import numpy as np
self = rv.core.create_tensorfield(
    data={(0, 0): np.random.rand(5,2,8).astype(float),
            (2, 2): np.random.rand(5,2,8).astype(float),
            },
    chunks=(2,2,2),
    #data = np.random.rand(5,2,8),
    store=r"C:\Users\GOB7\Downloads\test",
    #store='/u/gob7/test.zarr',
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

#filename = '/u/gob7/test.h5'
filename = r"C:\Users\GOB7\Downloads\test.h5"
self.h5_dump(filename, path='/my/awesome/array', mode='w')

#store='/u/gob7/test2.zarr'
#store=r"C:\Users\GOB7\Downloads\test2"
#h5path = '/myawesomearray'
#path=None
#overwrite=True
#kwargs={}
#with h5py.File(filename, mode='r') as fobj:
#    self2 = load_array_from_h5_file(fobj, h5path, store, path=None, overwrite=True)

#self.components[0]
#self.components[0].zarray[...]
#self.components[0][...]
