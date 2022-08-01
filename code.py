#%%
import numpy as np
import os
import matplotlib.pyplot as plt


from data_functions import load_image, load_paths, load_images_from_path, load_images_from_files, show_slices, show_all_slices, show_all_slices_dot, add_channel_ones, add_channel_zeros, random_slice

# data_path = os.sep.join([".", "workspace", "data", "medical", "ixi", "IXI-T1"])

# test the load_path function
# files = load_paths('/Users/misheton/Downloads/Data1/*/*/*/*/*/*.nii')
# for file in files[0]:
    # print(file)

# test the load_images_from_paths function
# change the path to get both AD and CN once the function is ready to tell which one is what
images = load_images_from_path('/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')

# test how the slice cutter works (random slice and add channel with the number of it)
# slice = random_slice(images[10])
# print("slice shape")
# print(slice.shape)
# print("slice 1,1,:")
# print(slice[:,:,2])


slice_0 = slice[:,:,0]
slice_1 = images[0][:, 127, :, 0]
slice_2 = images[0][:, :, 127, 0]
# show_slices([slice_0, slice_1, slice_2])

# getting our image data 
# img_data = load_image('/Users/misheton/Downloads/Data1/AD/006_S_4192/MPRAGE/2011-12-15_14_16_21.0/S133458/ADNI_006_S_4192_MR_MPRAGE_br_raw_20111216104600692_163_S133458_I272410.nii')

# what type is the image
print(type(images)) 
# the shape of the image
print(images[0].shape)

# adding an extra channel of 1s
# img_s = add_channel_ones(img_data)

# print(img_s.shape)
# print(img_s[:,:,:,1])

# lets show a slice 
# slice_ex1 = img_s[:,100,:,0]
# print(slice_ex1.shape)
# plt.imshow(slice_ex1.T, origin = 'lower',cmap = 'gray')
# plt.show()

# slice_0 = img_s[90, :, :, 0]
# slice_1 = img_s[:, 127, :, 0]
# slice_2 = img_s[:, :, 127, 0]
# show_slices([slice_0, slice_1, slice_2])


