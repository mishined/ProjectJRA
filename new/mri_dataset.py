# import os
import random
import glob

from munch import Munch

# from munch import Munch
# from PIL import Image
import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt

import albumentations as A
# conda install -c conda-forge albumentations
# from albumentations import pytorch
from albumentations.pytorch.transforms import ToTensorV2
# import scipy
# from sklearn import datasets

import torch
import pandas as pd
from torch.utils import data
# from torchvision import transforms
# from torchvision.datasets import ImageFolder


# function to extract the paths for files from a path
# listdir() from data_loader
def load_paths(data_path):
    files = []
    files.append(glob.glob(data_path + "/*/*/*/*/*/*.nii", 
                recursive = True))
    return files
    

class MRIDataset(data.Dataset):
    def __init__(self, root, transform=None): # root - directory
        self.data = self.read_csv(root + "/*.csv")
        self.samples, self.targets = self._make_dataset(root, self.data) #load paths and labels
        self.samples = self.samples[0]
        self.transform = transform

    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        img = self.load_image_from_path(fname, label) # load_images from paths 
        img = self.random_slice(img)
        # print(self.transform)
        if self.transform is not None:
            img = self.transform(image=img)
        return img, label

    def _make_dataset(self, root, data):
        # get all the files paths
        fnames = load_paths(root)
        labels = []
        # get the csv file
        # find the corresponding label for all files and add to labels
        for a in range(len(fnames[0])):
            # print("filename", a)
            id = fnames[0][a][-11:-4]
            if id.__contains__('_'):
                ind = id.find('_')
                id = id[ind+1:]
            # print(id)
            # index = data.ImageDataID[data.ImageDataID == id].index[0]
            index=data.loc[data['ImageDataID'] == id].index.values
            if data.Group[index[0]] == "AD":
                labels.append(1)
            else:
                labels.append(0)
        return fnames, labels
    
    def load_image_from_path(self, file_path, label):
        # print(file_path)
        img = self.load_image(file_path)
        # only reading one csv file that contains both AD and CN information
        # manually change the Image Data ID in the file (can do through pandas)
        # add target label as a colour channel
        if label == 1:
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
        df= pd.DataFrame(data)
        return df
    
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

    def __len__(self):
        return len(self.samples)


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, p=0.5):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    transform = A.Compose([
        A.RandomResizedCrop(256,256,scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        A.Resize(img_size,img_size),
        A.HorizontalFlip(p=0.5),
        # A.ToTensor(),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    # crop = transforms.RandomResizedCrop(
    #     img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    # rand_crop = transforms.Lambda(
    #     lambda x: crop(x) if random.random() < prob else x)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     rand_crop,
    #     transforms.Resize([img_size, img_size]),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    # ])

    dataset = MRIDataset(root, transform = transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size)

def get_test_loader(root, img_size=256, batch_size=32):
    print('Preparing DataLoader for the generation phase...')
    # transform = transforms.Compose([
    #     transforms.Resize([img_size, img_size]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    # ])

    transform = A.Compose([
        A.Resize(img_size,img_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    dataset = MRIDataset(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size)


class InputFetcher:
    def __init__(self, loader, latent_dim=16, mode='train'):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        # try:
        #     x, y = next(self.iter)
        # except (AttributeError, StopIteration):
        # self.iter = iter(self.loader)
        # self.iter = tf.convert_to_tensor(self.iter)
        # x, y = next(self.iter)
        x, y = next(iter(self.loader))
        return x, y

    def _fetch_refs(self):
        # try:
        #     x, y = next(self.iter)
        # except (AttributeError, StopIteration):
        self.iter_ref = iter(self.loader)
        x, y = next(self.iter_ref)
        # x, y = next(iter(self.loader))
        return x, y

    def __next__(self):
        x, y = self._fetch_inputs()
        print(y)
        if self.mode == 'train':
            x_ref, y_ref = self._fetch_refs()
            z_trg = torch.randn(len(x), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref,
                           z_trg=z_trg)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = torch.Tensor(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = torch.Tensor(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: torch.tensor(v).to(self.device)
                      for k, v in inputs.items()})

        # return Munch(inputs.x_src, inputs.y_src)

train_transform = A.Compose([
        A.RandomResizedCrop(256,256,scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        A.Resize(256,256),
        A.HorizontalFlip(p=0.5),
        # A.ToTensor(),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])


# test = MRIDataset(root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')

# train_loader = get_train_loader(root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')
# test_loader = get_test_loader(root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')
root1 = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data'
batch_size = 32
train_dataset = MRIDataset(root = root1)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size)

# img = train_loader.dataset.__getitem__(0,root = '/Users/misheton/OneDrive-UniversityofSussex/JRA/Data')
# lab = train_loader.dataset.targets[0]
# print(img)
# print(lab)

# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0]
label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
print(f"Label: {label}")