import torch
import torchvision
import math
import os
import numpy as np
import torchvision.transforms as transforms
from munch import munchify
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import ImageFolder

class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ImageDataset():
    
    def __init__(self, image_list, num_classes, transform=None, label_map=None):
        self.image_list = image_list
        self.transform = transform
        self.num_classes = num_classes
        
        f = open(image_list,'r')
        self.samples = [item.strip() for item in f.readlines()]
        
    def default_loader(self,path):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)
    
    def accimage_loader(self,path):
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            return self.pil_loader(path)

    def pil_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        image_path = self.samples[index]
        image_id = image_path
        img = self.default_loader(image_path)
        if self.transform is not None:
            sample = self.transform(img)
        return image_id, sample



class TransformImage(object):

    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std

        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))
        self.tf = transforms.Compose(tfs)

    def __call__(self,img):
        tensor = self.tf(img)
        return tensor
