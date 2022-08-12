import os
import random
import glob

from munch import Munch
from PIL import Image
import numpy as np
import nibabel as nib

import torch
import pandas as pd
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder


# function to extract the paths for files from a path
# listdir() from data_loader
def load_paths(data_path):
    files = []
    files.append(glob.glob(data_path + "/*/*/*/*/*/*.nii", 
                recursive = True))
    return files
    

class MRIDataset(data.Dataset):
    def __init__(self, root, transform=None): # root - directory
        self.samples, self.targets = self._make_dataset(root) #load paths and labels
        self.samples = self.samples[0]
        self.transform = transform

    def __getitem__(self, index, root):
        fname = self.samples[index]
        label = self.targets[index]
        img = self.load_image_from_path(fname, root) # load_images from paths 
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _make_dataset(self, root):
        # get all the files paths
        fnames = load_paths(root)
        labels = []
        # get the csv file
        data = self.read_csv(root + "/*.csv")
        # find the corresponding label for all files and add to labels
        for a in range(len(fnames[0])):
            # print("filename", a)
            id = fnames[0][a][-11:-4]
            if id.__contains__('_'):
                ind = id.find('_')
                id = id[ind+1:]
            # print(id)
            index = data.ImageDataID[data.ImageDataID == id].index[0]
            if data.Group[index] == "AD":
                labels.append("AD")
            else:
                labels.append("CN")
        return fnames, labels
    
    def load_image_from_path(self, file_path, data_path):
        print(file_path)
        img = self.load_image(file_path)
        # only reading one csv file that contains both AD and CN information
        # manually change the Image Data ID in the file (can do through pandas)
        data = self.read_csv(data_path + "/*.csv")
        id = file_path[-11:-4]
        if id[0] == "_":
            id = id[1:]
        index = data.ImageDataID[data.ImageDataID == id].index[0]
        # add target label as a colour channel
        if data.Group[index] == "AD":
            img = self.add_channel_zeros(img)
            print("AD")
        else:
            img = self.add_channel_ones(img)
            print("CN")
        return img

    # read a csv file and return the data 
    def read_csv(self, data_path):
        file = glob.glob(data_path, 
                    recursive = True)
        data = pd.read_csv(file[0])
        return data
    
    # function to load data from a single file
    def load_image(self, mri_file):
        # print(mri_file)
        image = nib.load(mri_file)
        img_data = image.get_fdata()
        return img_data

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

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = MRIDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size)

def get_test_loader(root, img_size=256, batch_size=32):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = MRIDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size)


test = MRIDataset(root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')

data1 = get_train_loader(root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')

img = data1.dataset.__getitem__(0,root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')
lab = data1.dataset.targets[0]
print(img)
print(lab)