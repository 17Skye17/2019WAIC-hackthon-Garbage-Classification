# -*- coding: UTF-8 -*-
import pretrainedmodels
import sys
import torch
import time
import os
import copy
import numpy as np
import torchvision
import torch.nn as nn
from DataReader import ImageDataset,TransformImage
from PIL import ImageFile

def accuracy(output, target,  topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = []
        for k in topk:
            #maxk = max(topk)
            batch_size = target.size(0)
            #print (output)
            _, pred = output.topk(k, 1, True, True)
            #correct = pred.eq(target.view(1, -1).expand_as(pred))
            onehot_pred = torch.zeros(target.size())
            
            for i in range(batch_size):
                onehot_pred[i][pred[i]] = 1

            correct_map = onehot_pred.double().cuda() * target
            correct = torch.nonzero(correct_map.sum(1)).size(0)
            res.append(correct * 100.0 / batch_size)
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def val(self):
        """
        Return: return the average accuracy until the current batch
        """
        return self.avg

class record():
    def __init__(self, model_name, map_path, num_classes, topk=5):
        self.f = open(model_name+'.lst','w',encoding='utf-8')
        self.map_path = os.path.abspath(map_path)
        
        print (self.map_path) 
        
        #self.lf = open(self.map_path,'r',encoding='cp936')
        self.lf = open("../modified.lst",'r')
        self.topk = topk
        
        #self.label_map = np.empty(num_classes,dtype='object')
        self._dict = {}
        for line in self.lf.readlines():
            item = line.strip().split(' ')
            self._dict[item[0]] = item[2].split(',')
        #for line in self.lf.readlines():
            # line is string
        #    item = line.strip().split(',')
            # save Chinese characters as bytes
        #    self.label_map[int(item[2])] = item[0].encode('cp936')

    def write(self, ids, outputs):
        batch_size = outputs.size(0)
        _, pred_indices = outputs.topk(self.topk,1,True,True)
        
        pred_indices = pred_indices.cpu().numpy()
        for i in range(batch_size):
            pred = pred_indices[i].tolist()
            line = ids[i] + ' '
            for j in range(len(pred)):
                line += str(pred[j]) + ','
            line = line + ' '
            for l in self._dict[ids[i].split('/')[-1].split('.')[0]]:
                line += l + ','
            self.f.write(line+'\n')
            #line = ids[i] + ' ' + str(gt) + ' ' + str(pred) +'\n'
            #self.f.write(line)

    def close(self):
        self.f.close()

class rank_record():
    def __init__(self, model_name, map_path, num_classes, topk=5):
        self.f = open('result.txt','w',encoding='utf-8')
        self.map_path = os.path.abspath(map_path)

        self.lf = open(self.map_path,'r',encoding='cp936')
        self.topk = topk
        #self.label_map = np.empty(num_classes,dtype='object')
        self.label_map = {}
        for line in self.lf.readlines():
            # line is string
            item = line.strip().split(',')
            self.label_map[item[2]] = int(item[1])
            #self.label_map[int(item[2])] = item[0].encode('cp936')
    
    def write(self, ids, outputs, topk=5):
        """
        if topk has 
        """
        
        with torch.no_grad():
            batch_size = outputs.size(0)
            _, pred = outputs.topk(topk, 1, True, True)
            for j in range(batch_size):
                hash_table = np.zeros([4])
                single_pred = pred[j]
                for i in range(len(single_pred)):
                    if single_pred[i].item() not in [0,1,2,3]:
                        hash_table[self.label_map[str(single_pred[i].item())]] += 1
                    else:
                        hash_table[single_pred[i].item()] += 1

                index = np.argmax(hash_table)

    def write1(self, ids, outputs):
        with torch.no_grad():
            batch_size = outputs.size(0)
            #_, pred_indices = outputs.topk(self.topk,1,True,True)
            valid_indices = [0,1,2,3]
            valid_pred = torch.zeros(outputs.size())
            for i in range(batch_size):
                valid_pred[i][valid_indices] = outputs[i][valid_indices].cpu().float()
            _, pred_indices = valid_pred.topk(1,1,True,True)
            for i in range(batch_size):
                #pred = self.label_map[pred_indices[i].tolist()]
                
                line = ids[i] + ','
                line += str(pred_indices[i].tolist()[0]) + '\n'
                self.f.write(line)
                #for j in range(len(pred)):
                #    line += str(pred[j],encoding='utf-8') + '/'
                #self.f.write(line+'\n')
                #line = ids[i] + ' ' + str(gt) + ' ' + str(pred) +'\n'
                #self.f.write(line)

    def close(self):
        self.f.close()

def inference_model(num_classes, model_name, label_map, model, dataloaders, device, is_inception=False):
    since = time.time()

    # Each epoch has a training and validation phase
    print("Enter Inference Mode...\n")
    model.eval()   # Set model to evaluate mode
    #writer = rank_record(model_name,label_map,num_classes,topk=5)
    writer = record(model_name, label_map, num_classes, topk=5)
    # Iterate over data.
    count = 0
    for ids, inputs in dataloaders['inference']:
        inputs = inputs.to(device)
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.double()
            writer.write(ids, outputs)
            #  topK accuracy
            count += 1
            print ("processed {}".format(count*inputs.size(0)))
    writer.close()
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_image_loader(test_list,model,num_classes,batch_size):
# Load Image dataset
    image_datasets = {
                    "inference":ImageDataset(test_list, transform=TransformImage(model, scale=0.875, random_crop=False, random_hflip=False, random_vflip=False, preserve_aspect_ratio=True), \
                    num_classes=num_classes)
                    }

    image_loader_dict = {x:torch.utils.data.DataLoader(
                dataset=image_datasets[x], 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True) for x in ['inference']}

    return image_loader_dict
