#%%
import rockverse as rv
import matplotlib.pyplot as plt

img = rv.voxel_image.import_raw(
    rawfile='/estgf_dados/P_D/GOB7/Bentheimer/Cropped_Oxyz_001_001_001_Nxyz_500_500_500.raw',
    store='/estgf_dados/P_D/GOB7/Bentheimer/original_ct.zarr',  #<- path to the imported the voxel image
    shape=(500, 500, 500),           #<- From metadata, image size
    dtype='<u2',                     #<- From metadata, little-endian 16-bit unsigned integer
    offset=0,                        #<- From metadata
    voxel_length=(5, 5, 5),          #<- From metadata
    voxel_unit='um',                 #<- From metadata, micrometer
    raw_file_order='F',              #<- Fortran file order
    chunks=(250, 250, 250),          #<- Our choice of chunk size will give a 2x2x2 chunk grid
    field_name='Attenuation',        #<- Our choice for field name (X-ray attenuation)
    field_unit='a.u.',               #<- Our choice for field units (arbitrary units)
    description='Bentheimer sandstone original X-ray CT',
    overwrite=True                   #<- Overwrite if file exists in disk
    )

viewer = rv.OrthogonalViewer(img)

#%%
x = viewer.histogram.bin_centers
y = viewer.histogram.pdf['full'].values
from rockverse.optimize import multi_gaussian_fit, gaussian_val
c = multi_gaussian_fit(x, y, c0=2)

plt.plot(x, y)
multi_gaussian_plot(c, x)

#%%
