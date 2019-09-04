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
from utils import train_utils
from datetime import datetime
from sync_batchnorm import convert_model
ImageFile.LOAD_TRUNCATED_IMAGES = True

if len(sys.argv)!=4:
    print ("Usage: python train.py model_name gpu_id batch_size")

model_name = sys.argv[1]
gpu_id = sys.argv[2]
batch_size = int(sys.argv[3])
#model_path = sys.argv[4]

num_classes = 403
num_epochs = 30
feature_extract = False
train_list =  './lists/train.lst'
val_list = './lists/test.lst'
label_map = './modified.lst'
save_root = './log'

if feature_extract:
    prefix = 'FE'
else:
    prefix = 'FT'

save_dir = os.path.join(save_root, model_name) + '-' + prefix + '-' + datetime.strftime(datetime.now(),"%m%d%H%M_plain")
if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class CrossEntropy_onehot():
    def __init__(self, size_average=True):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        self.size_average = size_average
    
    def calc(self, input, target):
        
        logsoftmax = nn.LogSoftmax()
        if self.size_average:
            return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        else:
            return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

def initialize_model(model, feature_extract):
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs,num_classes)
    return model

image_loader_dict = data_utils.get_image_loader(train_list, val_list, model, label_map, num_classes, batch_size)

model_ft = initialize_model(model, feature_extract)

# handle BN 
#model_ft = convert_model(model_ft)
model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = CrossEntropy_onehot(size_average=True)

#model_ft.load_state_dict(torch.load(model_path))
model_ft, hist = data_utils.train_model(model_ft, image_loader_dict, criterion, optimizer_ft, save_dir=save_dir, device=device, num_epochs=num_epochs, is_inception=(model_name=="inception"))
