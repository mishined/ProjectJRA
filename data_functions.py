from distutils.fancy_getopt import OptionDummy
import os
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import matplotlib
import ipykernel
import numpy as np
import matplotlib.pyplot as plt
import nilearn
import nibabel as nib
import monai
import glob
import fnmatch
import os.path
import csv
import pandas as pd
import random 

from random import randint
from nilearn import plotting
from nilearn import image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# print('getcwd:      ', os.getcwd())
# function to extract the files needed from a path
def load_paths(data_path):
    files = []
    files.append(glob.glob(data_path, 
                   recursive = True))
    return files

# function to load data from a single file
def load_image(mri_file):
    img = nib.load(mri_file)
    img_data = img.get_fdata()
    return img_data

# # function to load the images from a path
# '/Users/misheton/Downloads/Data1/AD/*.nii'
def load_images_from_path(data_path):
    files = load_paths(data_path + "/*/*/*/*/*/*.nii")
    images = []
    ids = []
    # print(len(files[0]))
    # print(files[0])
    # print(files[0][0])
    # only reading one csv file that contains both AD and CN information
    # manually change the Image Data ID in the file (can do through pandas)
    data = read_csv(data_path + "/*.csv")
    print(len(files[0]))
    for a in range(20):
        print(a)
        print(files[0][a])
        id = files[0][a][-11:-4]
        ids.append(id)
        # print(id)
        if id[0] == "_":
            id = id[1:]
        # print(id)
        index = data.ImageDataID[data.ImageDataID == id].index[0]
        img = load_image(files[0][a])
        if data.Group[index] == "AD":
            img = add_channel_zeros(img)
            print("AD")
        else:
            img = add_channel_ones(img)
            print("CN")
        images.append(img)
    return images,ids

# function to load the images from files
def load_images_from_files(files):
    images = []
    for file in files[0]:
        images.append(load_image(file))
    return images

def open_csv(data_path):
    file = open(data_path)
    csvreader = csv.reader(file)

    rows = []
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
    # print(header)
    # print(rows)
    # print(rows[0])
    return rows

# read a csv file and return the data 
def read_csv(data_path):
    file = glob.glob(data_path, 
                   recursive = True)
    data = pd.read_csv(file[0])
    return data


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

# add extra channel of ones to the colour channels
def add_channel_ones(img):
    img_data = np.array(img)

    data_1 = np.copy(img_data)
    b = np.copy(data_1[:,:,:,0])
    data_1[:,:,:,0] = b

    data_2 = np.copy(img_data)
    s = np.ones_like(b)
    data_2[:,:,:,0] = s

    img_s = np.concatenate((data_1,data_2), axis=-1)
    return img_s

# add extra channel of zeros to the colour channels 
def add_channel_zeros(img):
    img_data = np.array(img)

    data_1 = np.copy(img_data)
    b = np.copy(data_1[:,:,:,0])
    data_1[:,:,:,0] = b

    data_2 = np.copy(img_data)
    s = np.zeros_like(b)
    data_2[:,:,:,0] = s

    img_s = np.concatenate((data_1,data_2), axis=-1)
    return img_s

# def add_channel_slice(img, slice):
#     img_data = np.array(img)
#     # array of the image
#     data_1 = np.copy(img_data)
#     # array of the 0 channel
#     b = np.copy(data_1[:,:,:,0])
#     data_1[:,:,:,0] = b

#     data_2 = np.copy(img_data)
#     a = np.copy(data_1[:,:,:,1])
#     data_2[:,:,:,0] = a

#     data_3 = np.copy(img_data)
#     s = np.full_like(b, slice)
#     data_3[:,:,:,0] = s

#     img_s = np.concatenate((data_1, data_2, data_3), axis=-1)
#     return img_s

def add_channel_slice(img, slice):
    # create three different arrays with the different channels and concatenate them
    img_data = np.array(img)
    # save if the image has AD or HC channel
    hcad = img_data[:,:,:,1]

    # remove the second channel 
    img_del = np.delete(img_data,1,3)

    # array of the image
    data_1 = np.copy(img_del)
    # array of the 0 channel
    b = np.copy(data_1[:,:,:,0])
    data_1[:,:,:,0] = b

    # array with channel 2 - the AD or HC desired output
    data_2 = np.copy(img_del)
    a = np.copy(hcad)
    data_2[:,:,:,0] = a

    # array with channel 3 - the number of the slice
    data_3 = np.copy(img_del)
    s = np.full_like(b, slice)
    data_3[:,:,:,0] = s

    img_s = np.concatenate((data_1, data_2, data_3), axis=-1)
    return img_s

# choose a random slice from an image
def random_slice(img):
    # get the maximum slice number
    a = img.shape[0]
    # get a random number from the number of slices
    n = random.randint(0,a)
    # add a channel with the slice number
    slice_1 = add_channel_slice(img, n)
    # take only the randomly chosen slice
    slice = slice_1[n, :, :, :]
    # slice number
    print(n)
    # slice shape
    return slice



