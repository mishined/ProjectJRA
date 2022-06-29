import os
import numpy as np
import matplotlib.pyplot as plt
import nilearn
import nibabel as nib

from nilearn import plotting
from nilearn import image

# plotting.plot_glass_brain("example.nii")   
mri_file = 'adexample.nii'
img = nib.load(mri_file)

# what type is the image
print(type(img)) 
# the shape of the image
print(img.shape)

# get the information of the header
hdr = img.header
print(hdr)

# the voxel size (spatial resolution of the image)
print(img.header.get_zooms())
