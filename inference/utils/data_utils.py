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
            maxk = k
            batch_size = target.size(0)
            #print (output)
            _, pred = output.topk(maxk, 1, True, True)
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

    #def __str__(self):
    #    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #    return fmtstr.format(**self.__dict__)
    def val(self):
        """
        Return: return the average accuracy until the current batch
        """
        return self.avg

def train_model(model, dataloaders, criterion, optimizer, save_dir, device, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), save_dir+'/'+'Epoch_'+str(epoch))
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # Just set a flag, so that dropout and batch normailization can behave well
                print("Enter Training Mode...\n")
                model.train()  # Set model to training mode
            else:
                print("Enter Evaluation Mode...\n")
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            # Iterate over data.
            for ids, inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        outputs = outputs.double()
                        aux_outputs = aux_outputs.double()
                        loss1 = criterion.calc(outputs, labels)
                        loss2 = criterion.calc(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        outputs = outputs.double()
                        loss = criterion.calc(outputs, labels)
                    
                    #  topK accuracy
                    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                    top1.update(acc1, outputs.size(0))
                    top5.update(acc5, outputs.size(0))
                    print ("BestEpoch:{} Batch Loss:{:.4f} Acc@1:{:.4f} Acc@5:{:.4f}".format(best_epoch, loss, top1.val, top5.val))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
           
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss:{:.4f} Acc@1:{:.4f} Acc@5:{:.4f}'.format(phase, epoch_loss, top1.val, top5.val))
            epoch_acc = top1.val
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            top1.reset()
            top5.reset()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_dir+'/'+'Best_model_epoch_'+str(best_epoch))
    return model, val_acc_history

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
