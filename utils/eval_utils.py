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

def rank_acc(output, target):
    #valid_indices = [66,100,106,121]
    valid_indices = [0,1,2,3]
    batch_size = output.size(0)

    #top1 = output[:][valid_indices].topk(1, 1, True, True)
    #top1 = output.topk(1,1,True,True)
    with torch.no_grad():
        # find top1 
        valid_pred = torch.zeros(target.size())
        for i in range(batch_size):
            valid_pred[i][valid_indices] = output[i][valid_indices].cpu().float()

        _, top1 = valid_pred.topk(1, 1, True, True)
        top1_pred = torch.zeros(target.size())
        for i in range(batch_size):
            top1_pred[i][top1[i]] = 1
        #valid_map = valid_pred.double().cuda() * target
        correct_map = top1_pred.double().cuda() * target
        correct = torch.nonzero(correct_map.sum(1)).size(0)
        return correct * 100.0 / batch_size
        #print (valid_pred)

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
        # not valid, if this is set to n, the result is larger than expected.
        # set to sum*n + val
        self.sum += val * n
        #self.sum = self.sum * self.count + val
        self.count += n
        self.avg = self.sum / self.count

class record():
    def __init__(self, model_name, map_path, num_classes, topk=5):
        self.f = open(model_name+'.lst','w',encoding='utf-8')
        self.map_path = os.path.abspath(map_path)
        
        print (self.map_path) 
        
        self.lf = open(self.map_path,'r',encoding='utf-8')
        self.topk = topk
        self.label_map = np.empty(num_classes,dtype='object')
        for line in self.lf.readlines():
            # line is string
            item = line.strip().split(',')
            # save Chinese characters as bytes
            self.label_map[int(item[2])] = item[0].encode('utf-8')

    def write(self, ids, outputs, labels):
        with torch.no_grad():
            batch_size = outputs.size(0)
            np_labels = labels.cpu().numpy()
            _, pred_indices = outputs.topk(self.topk,1,True,True)
            
            pred_indices = pred_indices.cpu().numpy()
            for i in range(batch_size):
                gt_indices = np.nonzero(np_labels[i])[0]
                #print (self.label_map[[1,2,3]])
                gt = self.label_map[gt_indices.tolist()]
                pred = self.label_map[pred_indices[i].tolist()]
                line = ids[i] + ' '
                for j in range(len(gt)):
                    line += str(gt[j],encoding='utf-8') + '/'
                line += ' '
                for j in range(len(pred)):
                    line += str(pred[j],encoding='utf-8') + '/'
                self.f.write(line+'\n')
                #line = ids[i] + ' ' + str(gt) + ' ' + str(pred) +'\n'
                #self.f.write(line)

    def close(self):
        self.f.close()

class rank_record():
    def __init__(self, model_name, map_path, num_classes, topk=5):
        self.f = open(model_name+'.lst','w',encoding='utf-8')
        self.map_path = os.path.abspath(map_path)
        
        print (self.map_path) 
        
        self.lf = open(self.map_path,'r',encoding='utf-8')
        self.topk = topk
        self.label_map = np.empty(num_classes,dtype='object')
        for line in self.lf.readlines():
            # line is string
            item = line.strip().split(',')
            # save Chinese characters as bytes
            self.label_map[int(item[2])] = item[0].encode('utf-8')

    def write(self, ids, outputs, labels):
        with torch.no_grad():
            batch_size = outputs.size(0)
            np_labels = labels.cpu().numpy()
            #_, pred_indices = outputs.topk(self.topk,1,True,True)
            #valid_indices = [66,100,106,121]
            valid_indices = [0,1,2,3]
            valid_pred = torch.zeros(labels.size())
            for i in range(batch_size):
                valid_pred[i][valid_indices] = outputs[i][valid_indices].cpu().float()
            _, pred_indices = valid_pred.topk(1,1,True,True)
            #pred_indices = pred_indices.cpu().numpy()
            for i in range(batch_size):
                gt_indices = np.nonzero(np_labels[i])[0]
                #print (self.label_map[[1,2,3]])
                gt = self.label_map[gt_indices.tolist()]
                pred = self.label_map[pred_indices[i].tolist()]
                line = ids[i] + ' '
                for j in range(len(gt)):
                    line += str(gt[j],encoding='utf-8') + '/'
                line += ' '
                for j in range(len(pred)):
                    line += str(pred[j],encoding='utf-8') + '/'
                self.f.write(line+'\n')
                #line = ids[i] + ' ' + str(gt) + ' ' + str(pred) +'\n'
                #self.f.write(line)

    def close(self):
        self.f.close()

def eval_model(num_classes, model_name, label_map, model, dataloaders, device, is_inception=False):
    since = time.time()

    # Each epoch has a training and validation phase
    print("Enter Evaluation Mode...\n")
    model.eval()   # Set model to evaluate mode
    #writer = record(model_name,label_map,num_classes,topk=5)
    writer = rank_record(model_name,label_map,num_classes,topk=5)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    rank1 = AverageMeter('Rank1', ':6.2f')
    # Iterate over data.
    for ids, inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs.double()
            writer.write(ids, outputs, labels)
            #  topK accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            rank1_acc = rank_acc(outputs, labels)

            top1.update(acc1, n=outputs.size(0))
            top5.update(acc5, n=outputs.size(0))
            rank1.update(rank1_acc, outputs.size(0))
            print ("Batch Acc@1:{:.4f} Acc@5:{:.4f} Rank@1:{:.4f}".format(top1.avg, top5.avg, rank1.avg))
    
    writer.close()
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print ("Total Acc@1:{:.4f} Acc@5:{:.4f} Rank@1:{:.4f}".format(top1.avg, top5.avg, rank1.avg))

    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_image_loader(train_list,val_list,model,label_map,num_classes,batch_size):
# Load Image dataset
    image_datasets = {
                    "train":ImageDataset(train_list, transform=TransformImage(model, scale=0.875, random_crop=True, random_hflip=True, random_vflip=True, preserve_aspect_ratio=True), \
                    label_map=label_map, num_classes=num_classes),
                    "val":ImageDataset(val_list, transform=TransformImage(model, scale=0.875, random_crop=False, random_hflip=False, random_vflip=False, preserve_aspect_ratio=True), \
                    label_map=label_map, num_classes=num_classes)
                    }

    image_loader_dict = {x:torch.utils.data.DataLoader(
                dataset=image_datasets[x], 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True) for x in ['train', 'val']}

    return image_loader_dict
