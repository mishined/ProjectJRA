#%%
from distutils.fancy_getopt import OptionDummy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib
import ipykernel
import numpy as np
import matplotlib.pyplot as plt
import nilearn
import nibabel as nib

from numpy import expand_dims
from nilearn import plotting
from nilearn import image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# plotting.plot_glass_brain("example.nii")   
mri_file = 'adexample.nii'
img = nib.load(mri_file)

# what type is the image
print(type(img)) 
# the shape of the image
print(img.shape)

# get the information of the header; more info on how to read it: https://brainder.org/2012/09/23/the-nifti-file-format/
hdr = img.header
print(hdr)

# the voxel size (spatial resolution of the image)
print(img.header.get_zooms())

# we have the NiFTI object and now we load the actual data (voxel intensities)
img_data = img.get_fdata()
print(img_data.shape)
print(type(img_data))

#%%
# lets show a slice 
# slice_ex = img_data[:,100,:,0]
# print(slice_ex.shape)
# plt.imshow(slice_ex.T, origin = 'lower',cmap = 'gray')
# plt.show()

#%%
# show the same slice of all dimensions
def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
   plt.suptitle("Center slices for EPI image") 
   plt.show()
# (53, 61, 33)
slice_0 = img_data[90, :, :, 0]
slice_1 = img_data[:, 127, :, 0]
slice_2 = img_data[:, :, 127, 0]
# show_slices([slice_0, slice_1, slice_2])

#%%
# show the same slice of all dimensions
# fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

# ax[0].imshow(img_data[69, :, :, 0].T, origin='lower', cmap='gray')
# ax[0].set_xlabel('Second dim voxel coords.', fontsize=12)
# ax[0].set_ylabel('Third dim voxel coords', fontsize=12)
# ax[0].set_title('First dimension, slice nr. 70', fontsize=15)

# ax[1].imshow(img_data[:, 99, :, 0].T, origin='lower', cmap='gray')
# ax[1].set_xlabel('First dim voxel coords.', fontsize=12)
# ax[1].set_ylabel('Third dim voxel coords', fontsize=12)
# ax[1].set_title('Second dimension, slice nr. 100', fontsize=15)

# ax[2].imshow(img_data[:, :, 99, 0].T, origin='lower', cmap='gray')
# ax[2].set_xlabel('First dim voxel coords.', fontsize=12)
# ax[2].set_ylabel('Second dim voxel coords', fontsize=12)
# ax[2].set_title('Third dimension, slice nr. 100', fontsize=15)

# fig.tight_layout()
# plt.show()

#%%
# # show the same slice and the chosen spot on it
# import matplotlib.patches as patches
# fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

# ax[0].imshow(img_data[90, :, :, 0].T, origin='lower', cmap='gray')
# ax[0].set_xlabel('Second dim voxel coords.', fontsize=12)
# ax[0].set_ylabel('Third dim voxel coords', fontsize=12)
# ax[0].set_title('First dimension, slice nr. 70', fontsize=15)
# rect = patches.Rectangle((119, 109), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
# ax[0].add_patch(rect)

# ax[1].imshow(img_data[:, 127, :, 0].T, origin='lower', cmap='gray')
# ax[1].set_xlabel('First dim voxel coords.', fontsize=12)
# ax[1].set_ylabel('Third dim voxel coords', fontsize=12)
# ax[1].set_title('Second dimension, slice nr. 100', fontsize=15)
# rect = patches.Rectangle((69, 109), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
# ax[1].add_patch(rect)

# ax[2].imshow(img_data[:, :, 127, 0].T, origin='lower', cmap='gray')
# ax[2].set_xlabel('First dim voxel coords.', fontsize=12)
# ax[2].set_ylabel('Second dim voxel coords', fontsize=12)
# ax[2].set_title('Third dimension, slice nr. 100 ', fontsize=15)
# rect = patches.Rectangle((69, 119), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
# ax[2].add_patch(rect)

# fig.tight_layout()
# plt.show()

#%%
# adding an extra channel of 1s
img_data = np.array(img_data)
print(img_data.shape)

data_1 = np.copy(img_data)
b = np.copy(data_1[:,:,:,0])
print(b.shape)
data_1[:,:,:,0] = b

data_2 = np.copy(img_data)
s = np.ones_like(b)
print(s.shape)
data_2[:,:,:,0] = s

img_s = np.concatenate((data_1,data_2), axis=-1)

print(img_s.shape)
# print(img_s[:,:,:,1])

#%%
# lets show a slice 
slice_ex1 = img_s[:,100,:,0]
print(slice_ex1.shape)
plt.imshow(slice_ex1.T, origin = 'lower',cmap = 'gray')
plt.show()

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


