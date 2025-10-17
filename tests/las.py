#%%
import os
import rockverse as rv

self = rv.las_sample1()
#final_data.tree()

#self['Curve'].find('*NM*')

#self['Curve'].find('*M*')
#self['Curve'].tree()

#self['Well'].tree()
self.tree()

a=self['Curve'].data.create_scalarfield('DT')
a.coords[0][...]

#final_data['Well'][0]
# %%
