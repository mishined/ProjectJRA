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
import monai
import glob

from nilearn import plotting
from nilearn import image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# function to load data
def load_image(mri_file):
    img = nib.load(mri_file)
    img_data = img.get_fdata()
    return img_data

print('getcwd:      ', os.getcwd())

# def load_images(data_path='/Users/misheton/Downloads/Data/**/*.nii'):
#     files = glob.glob(data_path, 
#                    recursive = True)
#     return files
    

# show the same slice of all dimensions
def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
   plt.suptitle("Center slices for EPI image") 
   plt.show()


# show the same slice of all dimensions
def show_all_slices(slices):
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    ax[0].imshow(slices[0].T, origin='lower', cmap='gray')
    ax[0].set_xlabel('Second dim voxel coords.', fontsize=12)
    ax[0].set_ylabel('Third dim voxel coords', fontsize=12)
    ax[0].set_title('First dimension, slice nr. 70', fontsize=15)

    ax[1].imshow(slices[1].T, origin='lower', cmap='gray')
    ax[1].set_xlabel('First dim voxel coords.', fontsize=12)
    ax[1].set_ylabel('Third dim voxel coords', fontsize=12)
    ax[1].set_title('Second dimension, slice nr. 100', fontsize=15)

    ax[2].imshow(slices[2].T, origin='lower', cmap='gray')
    ax[2].set_xlabel('First dim voxel coords.', fontsize=12)
    ax[2].set_ylabel('Second dim voxel coords', fontsize=12)
    ax[2].set_title('Third dimension, slice nr. 100', fontsize=15)

    fig.tight_layout()
    plt.show()

# show the same slice of all dimensions with a dot at the same place
def show_all_slices_dot(slices):
    import matplotlib.patches as patches
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    ax[0].imshow(slices[0].T, origin='lower', cmap='gray')
    ax[0].set_xlabel('Second dim voxel coords.', fontsize=12)
    ax[0].set_ylabel('Third dim voxel coords', fontsize=12)
    ax[0].set_title('First dimension, slice nr. 70', fontsize=15)
    rect = patches.Rectangle((119, 109), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    ax[1].imshow(slices[1].T, origin='lower', cmap='gray')
    ax[1].set_xlabel('First dim voxel coords.', fontsize=12)
    ax[1].set_ylabel('Third dim voxel coords', fontsize=12)
    ax[1].set_title('Second dimension, slice nr. 100', fontsize=15)
    rect = patches.Rectangle((69, 109), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)

    ax[2].imshow(slices[2].T, origin='lower', cmap='gray')
    ax[2].set_xlabel('First dim voxel coords.', fontsize=12)
    ax[2].set_ylabel('Second dim voxel coords', fontsize=12)
    ax[2].set_title('Third dimension, slice nr. 100 ', fontsize=15)
    rect = patches.Rectangle((69, 119), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
    ax[2].add_patch(rect)

    fig.tight_layout()
    plt.show()


def add_channel(img):
    img_data = np.array(img)

    data_1 = np.copy(img_data)
    b = np.copy(data_1[:,:,:,0])
    print(b.shape)
    data_1[:,:,:,0] = b

    data_2 = np.copy(img_data)
    s = np.ones_like(b)
    print(s.shape)
    data_2[:,:,:,0] = s

    img_s = np.concatenate((data_1,data_2), axis=-1)
    return img_s

