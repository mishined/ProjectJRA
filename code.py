#%%
import numpy as np
import os
import matplotlib.pyplot as plt

from data_functions import load_image, load_images, show_slices, show_all_slices, show_all_slices_dot, add_channel

# data_path = os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"])
files = load_images('/Users/misheton/Downloads/Data/*.nii')

# getting our image data 
img_data = load_image('adexample.nii')

# what type is the image
print(type(img_data)) 
# the shape of the image
print(img_data.shape)

# adding an extra channel of 1s
img_s = add_channel(img_data)

print(img_s.shape)
# print(img_s[:,:,:,1])

# lets show a slice 
slice_ex1 = img_s[:,100,:,0]
print(slice_ex1.shape)
plt.imshow(slice_ex1.T, origin = 'lower',cmap = 'gray')
plt.show()

slice_0 = img_s[90, :, :, 0]
slice_1 = img_s[:, 127, :, 0]
slice_2 = img_s[:, :, 127, 0]
show_slices([slice_0, slice_1, slice_2])


