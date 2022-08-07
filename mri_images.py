
import os
import torch
import torch.nn as nn
import torch.optim as optim
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
from random import shuffle
from nilearn import plotting
from nilearn import image


class MRI_Images:
    def __init__(self, path):
        self.database_path = path
        self.images, self.ids = self.load_images_from_path(self.database_path)

        self.all_image_gotten = self.check_ids_len_and_image(self.images, self.ids)

        self.manipulated_images = self.manipulate_images(self.images)

        self.shuffled_slices, self.sample = self.shuffle_and_batch(self.manipulated_images)

    def check_ids_len_and_image(self, images, ids):
        if len(images) == len(ids):
            return True
    
    def load_images_from_path(self, data_path):
        files = self.load_paths(data_path + "/*/*/*/*/*/*.nii")
        images = []
        ids = []
        # print(len(files[0]))
        # print(files[0])
        # print(files[0][0])
        # only reading one csv file that contains both AD and CN information
        # manually change the Image Data ID in the file (can do through pandas)
        data = self.read_csv(data_path + "/*.csv")
        # print(len(files[0]))
        for a in range(10):
            # print(a)
            # print(files[0][a])
            id = files[0][a][-11:-4]
            # print(id)
            if id[0] == "_":
                id = id[1:]
            ids.append(id)
            # print(id)
            index = data.ImageDataID[data.ImageDataID == id].index[0]
            img = self.load_image(files[0][a])
            if data.Group[index] == "AD":
                img = self.add_channel_zeros(img)
                print("AD")
            else:
                img = self.add_channel_ones(img)
                print("CN")
            images.append(img)
        return images,ids

    # function to extract the files needed from a path
    def load_paths(self, data_path):
        files = []
        files.append(glob.glob(data_path, 
                    recursive = True))
        return files

    # function to load data from a single file
    def load_image(self, mri_file):
        img = nib.load(mri_file)
        img_data = img.get_fdata()
        return img_data

    # read a csv file and return the data 
    def read_csv(self, data_path):
        file = glob.glob(data_path, 
                    recursive = True)
        data = pd.read_csv(file[0])
        return data

    # add extra channel of ones to the colour channels
    def add_channel_ones(self, img):
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
    def add_channel_zeros(self, img):
        img_data = np.array(img)

        data_1 = np.copy(img_data)
        b = np.copy(data_1[:,:,:,0])
        data_1[:,:,:,0] = b

        data_2 = np.copy(img_data)
        s = np.zeros_like(b)
        data_2[:,:,:,0] = s

        img_s = np.concatenate((data_1,data_2), axis=-1)
        return img_s

    def add_channel_slice(self, img, slice):
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
    def random_slice(self, img):
        # print("----------------------------------", img, type(img))
        # get the maximum slice number
        a = img.shape[0]
        # get a random number from the number of slices
        n = random.randint(0,a-1)
        # add a channel with the slice number
        slice_1 = self.add_channel_slice(img, n)
        # take only the randomly chosen slice
        slice = slice_1[n, :, :, :]
        # slice number
        # print(n)
        # slice shape
        return slice

    # get random slice for all images and add the slice number to channels
    def manipulate_images(self, images):
        manipulated_images = []
        for i in range(len(images)):
            manipulated_image = self.random_slice(images[i])
            manipulated_image = np.moveaxis(manipulated_image, -1, 0)
            manipulated_images.append(manipulated_image)
        return manipulated_images
    
    # shuffle images
    def shuffle_and_batch(self,images):
        for item in range(len(images)):
            j = random.randint(0,len(images)-1)
            images[item] = images[j]
            images[j] = images[item]
            batch_beginning=random.randint(0, len(images)-1)
            batch_end=random.randint(batch_beginning, len(images)-1)
            batch0 = images[batch_beginning:batch_end]
        return images, batch0