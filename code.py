#%%
import numpy as np
import matplotlib.pyplot as plt

from data_functions import load_image, show_slices, show_all_slices, show_all_slices_dot, add_channel

# getting our image data 
img_data = load_image('adexample.nii')

# what type is the image
print(type(img_data)) 
# the shape of the image
print(img_data.shape)

#%%
# lets show a slice 
# slice_ex = img_data[:,100,:,0]
# print(slice_ex.shape)
# plt.imshow(slice_ex.T, origin = 'lower',cmap = 'gray')
# plt.show()

#%%
# show the same slice of all dimensions
slice_0 = img_data[90, :, :, 0]
slice_1 = img_data[:, 127, :, 0]
slice_2 = img_data[:, :, 127, 0]
show_slices([slice_0, slice_1, slice_2])

#%%
# adding an extra channel of 1s
img_s = add_channel(img_data)

print(img_s.shape)
# print(img_s[:,:,:,1])

#%%
# lets show a slice 
# slice_ex1 = img_s[:,100,:,0]
# print(slice_ex1.shape)
# plt.imshow(slice_ex1.T, origin = 'lower',cmap = 'gray')
# plt.show()

slice_0 = img_s[90, :, :, 0]
slice_1 = img_s[:, 127, :, 0]
slice_2 = img_s[:, :, 127, 0]
show_slices([slice_0, slice_1, slice_2])

#%%
# adding an extra dimension
# img_s = expand_dims(img_data, axis = -1)
# print(type(img_s))
# arr = np.array(img_s[:,:,:,:,0])
# print(arr.shape)
# s = np.ones_like(arr)
# img_new = np.copy(img_s)
# img_new[:,:,:,:,0]= s

# print(img_new.shape)
# print(img_new[:,:,:,:,0])

#%%
# # lets show a slice 
# slice_ex1 = img_new[:,100,:,0,0]
# print(slice_ex1.shape)
# plt.imshow(slice_ex1[:,:].T, origin = 'lower',cmap = 'gray')
# plt.show()

# slice_0 = img_new[90, :, :, 0]
# slice_1 = img_new[:, 127, :, 0]
# slice_2 = img_new[:, :, 127, 0]
# show_slices_([slice_0, slice_1, slice_2])


