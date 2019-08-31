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
from utils import inference_utils
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True


model_name = 'senet154' 
batch_size =16
model_path = 'log/Epoch_14'
num_classes = 399
feature_extract = True
test_list = sys.argv[1]
label_map = './mapping_list.txt'

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(model, feature_extract):
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs,num_classes)
    return model

image_loader_dict = inference_utils.get_image_loader(test_list,  model,  num_classes, batch_size)

model_ft = initialize_model(model, feature_extract)

model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load(model_path))

best_model = inference_utils.inference_model(num_classes, model_name, label_map, model_ft, image_loader_dict, device=device, is_inception=(model_name=="inception"))
