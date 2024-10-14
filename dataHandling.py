import torch
import torchvision
import numpy as np
import os
from PIL import Image
import random
import tifffile
import cv2
from skimage.segmentation import slic

from math import ceil
from segmentation_models_pytorch.encoders import get_preprocessing_fn


import torchvision.transforms.functional as F
import torchvision.transforms as t

def generate_mask(image_shape, num_squares, square_size, mask_color=0):
    mask = np.ones(image_shape, dtype=np.uint8) * 255
    for _ in range(num_squares):
        x = np.random.randint(0, image_shape[1] - square_size - 1)
        y = np.random.randint(0, image_shape[0] - square_size - 1)
        mask[y:y+square_size, x:x+square_size] = mask_color

    return mask


class SingleSceneDataLoader(torch.utils.data.Dataset):
    def __init__(self, path, patchSize = 448, stride = 448, dataset = 'Vaihingen', transform = None):
        self.dataset = dataset

        data = tifffile.imread(path)
        if dataset != 'Vaihingen':
            data = cv2.resize(data, (0, 0), fx = 0.448, fy = 0.448)
            mask = generate_mask(data.shape, 600, 70, 0)
        else:
            mask = generate_mask(data.shape, 400, 70, 0)
        self.masked_img = data.copy()

        data = np.transpose(data, (2, 0, 1))
        self.data = torch.from_numpy(data)

        self.masked_img[np.where(mask == 0)] = 0
        self.masked_img = np.transpose(self.masked_img, (2, 0, 1))
        self.masked_img = torch.from_numpy(self.masked_img)

        self.transform = transform
        self.GaussianBlur = t.GaussianBlur(5, sigma=(0.1, 2.0))
        self.Grayscale = t.Grayscale(num_output_channels=3)
        self.ColorJitter = t.ColorJitter(contrast=0.1, brightness=0.1, saturation=0, hue=0)
        self.RandomErasing = t.RandomErasing(p=1, scale=(0.05, 0.05), ratio=(0.5, 0.5), value=0, inplace=False)

        self.RandomHorizontalFlip = t.RandomHorizontalFlip()
        self.RandomVerticalFlip = t.RandomVerticalFlip()
        self.randcrop = t.RandomCrop(size=(300, 300)) 
        self.resize = t.Resize(size=(448, 448), antialias=True)

        _, rows, cols = self.data.shape
        n1 = ceil((rows - patchSize + 1) / stride)
        n2 = ceil((cols - patchSize + 1) / stride)
        self.n_patches = n1 * n2
        self.patch_coords = []          
        # generate path coordinates
        for i in range(n1):
            for j in range(n2):
                # coordinates in (x1, x2, y1, y2)
                patch_coords = ( 
                                [stride * i, stride * i + patchSize, stride * j, stride * j + patchSize], 
                                [stride * (i + 1), stride * (j + 1)])
                self.patch_coords.append(patch_coords)
        print('no of patches', len(self.patch_coords))

    def __len__(self):
        return self.n_patches
  
    def __getitem__(self, idx):
        limits = self.patch_coords[idx][0]
        
        I1 = self.data[:, limits[0]:limits[1], limits[2]:limits[3]]
        # if random.randint(0,1):
        #     I1 = self.resize(self.randcrop(I1))
            
        # I1 = self.RandomHorizontalFlip(I1)
        # I1 = self.RandomVerticalFlip(I1)
        # I2 = self.RandomErasing(I1)
        # I2 = self.ColorJitter(I2)

        I2 = self.masked_img[:, limits[0]:limits[1], limits[2]:limits[3]]
        for i in range(I2.shape[0]):
            I2[i] = self.ColorJitter((I2[i]).unsqueeze(dim=0)).squeeze()

        sample = {'I1': I1/255, 'I2': I2/255}

        return sample
    




class PatchLoader(torch.utils.data.Dataset):
    def __init__(self, path, test_scene_ids ,patchSize = 448, dataset='Vaihingen', no_patches=20):

        self.dataset = dataset
        self.patchSize = patchSize
        self.path = path
        self.patches = []

        for patch_name in os.listdir(path):
            patch_parent_id = patch_name.split("_")[0]
            if patch_parent_id not in test_scene_ids:
                self.patches.append(patch_name)

        self.patches = self.patches[:no_patches]
        print('no of patches', len(self.patches))

        self.ColorJitter = t.ColorJitter(contrast=0.1, brightness=0.1, saturation=0, hue=0)
        self.GaussianBlur = torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0))
        self.RandomHorizontalFlip = t.RandomHorizontalFlip(p=1)
        self.RandomVerticalFlip = t.RandomVerticalFlip(p=1)
        # self.random_transform = t.RandomChoice([self.ColorJitter, self.Grayscale])


    def __len__(self):
        return len(self.patches)
  
    def __getitem__(self, idx):
        patch_path = os.path.join(self.path, self.patches[idx])
        
        I1 = np.asarray(Image.open(patch_path))
        if self.dataset != 'Vaihingen':
            I1 = cv2.resize(I1, (0, 0), fx = 0.448, fy = 0.448)
        mask = generate_mask(I1.shape, 75, 20, 0)
        I2 = I1.copy()
        I2[np.where(mask == 0)] = 0
        I2 = np.transpose(I2, (2, 0, 1))
        I2 = torch.tensor(I2)
        # I2 = self.GaussianBlur(I2)
        I1 = np.transpose(I1, (2, 0, 1))
        I1 = torch.tensor(I1)


        # if random.randint(0,1): 
        #     I1 = self.RandomHorizontalFlip(I1)
        #     I2 = self.RandomHorizontalFlip(I2)
        # if random.randint(0,1): 
        #     I1 = self.RandomVerticalFlip(I1)
        #     I2 = self.RandomVerticalFlip(I2)

        for i in range(I2.shape[0]):
            I2[i] = self.ColorJitter((I2[i]).unsqueeze(dim=0)).squeeze()

        sample = {'I1': I1/255, 'I2': I2/255}


        return sample
    




class PatchLoader2(torch.utils.data.Dataset):
    def __init__(self, path, test_scene_ids ,patchSize = 448, dataset='Vaihingen', no_patches=20):

        self.dataset = dataset
        self.patchSize = patchSize
        self.path = path
        self.patches = []

        for patch_name in os.listdir(path):
            patch_parent_id = patch_name.split("_")[0]
            if patch_parent_id not in test_scene_ids:
                self.patches.append(patch_name)

        self.patches = self.patches[:no_patches]
        print('no of patches', len(self.patches))

        self.ColorJitter = t.ColorJitter(contrast=0.1, brightness=0.1, saturation=0, hue=0)
        self.GaussianBlur = torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0))
        self.RandomHorizontalFlip = t.RandomHorizontalFlip(p=1)
        self.RandomVerticalFlip = t.RandomVerticalFlip(p=1)
        # self.random_transform = t.RandomChoice([self.ColorJitter, self.Grayscale])


    def __len__(self):
        return len(self.patches)
  
    def __getitem__(self, idx):
        patch_path = os.path.join(self.path, self.patches[idx])
        
        I1 = np.asarray(Image.open(patch_path))
        if self.dataset != 'Vaihingen':
            I1 = cv2.resize(I1, (0, 0), fx = 0.448, fy = 0.448)
        I2 = I1.copy()
        I1 = I1.copy()

        mask1 = generate_mask(I1.shape, 75, 20, 0)
        mask2 = generate_mask(I1.shape, 75, 20, 0)

        I1[np.where(mask1 == 0)] = 0
        I2[np.where(mask2 == 0)] = 0

        I2 = np.transpose(I2, (2, 0, 1))
        I2 = torch.tensor(I2)
        I1 = np.transpose(I1, (2, 0, 1))
        I1 = torch.tensor(I1)


        # if random.randint(0,1): 
        #     I1 = self.RandomHorizontalFlip(I1)
        #     I2 = self.RandomHorizontalFlip(I2)
        # if random.randint(0,1): 
        #     I1 = self.RandomVerticalFlip(I1)
        #     I2 = self.RandomVerticalFlip(I2)

        for i in range(I2.shape[0]):
            I2[i] = self.ColorJitter((I2[i]).unsqueeze(dim=0)).squeeze()
            I1[i] = self.ColorJitter((I1[i]).unsqueeze(dim=0)).squeeze()

        sample = {'I1': I1/255, 'I2': I2/255}


        return sample
    

