import pretrainedmodels
import sys
import torch
import time
import os
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from DataReader import ImageDataset,TransformImage
from PIL import ImageFile
from utils import eval_utils
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

if len(sys.argv)!=5:
    print ("Usage: python extract.py model_name image_list save_file gpu_id")

model_name = sys.argv[1]
gpu_id = sys.argv[2]
batch_size = int(sys.argv[3])
model_path = sys.argv[4]
num_classes = 403
feature_extract = False
train_list =  './train.lst'
val_list = './test.lst'
label_num = './modified.lst'
label_map = './mapping_list.txt'
save_root = './log'

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model, feature_extract):
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs,num_classes)
    return model

image_loader_dict = eval_utils.get_image_loader(train_list, val_list, model, label_num, num_classes, batch_size)

model_ft = initialize_model(model, feature_extract)
#model_ft = nn.DataParallel(model_ft)
model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load(model_path))

best_model = eval_utils.eval_model(num_classes, model_name, label_map, model_ft, image_loader_dict, device=device, is_inception=(model_name=="inception"))
