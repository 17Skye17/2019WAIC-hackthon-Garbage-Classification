# -*- coding: UTF-8 -*- 
import os
import csv
import codecs
import random
import string
import multiprocessing
from tqdm import tqdm
from PIL import Image


failed_f = 'failed.lst'
valid_f = 'data.lst'
label_f = 'label_num.lst'
map_f = 'label_map.lst'
save_dir = '../data'
data_dir = '../garbage'

def check_pic(path, random_id):
    basename = os.path.basename(path)
    try:
        pil_image = Image.open(path)
    except:
        print ("Warning: Failed to parse image{}".format(basename))
        f = open(failed_f,'a')
        f.write(path+'\n')
        f.close()
        return False
    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print ("Warning: Failed to convert image {} to RGB".format(basename))
        f = open(failed_f,'a')
        f.write(path+'\n')
        f.close()
        return False
    try:
        pil_image_rgb.save(os.path.join(save_dir, random_id+'.jpg'), format='JPEG', quality=100)
    except:
        print ("Warning: Failed to save image {}".format(path))
        f = open(failed_f,'a')
        f.write(path+'\n')
        f.close()
        return False
    return True

def gen_random_id():
    _id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    return _id

def get_image_list():
    f = open(valid_f,'w')
    for root, dirs, files in os.walk(data_dir):
        for _file in tqdm(files):
            path = os.path.join(root,_file)
            random_id = gen_random_id()
            valid_pic = check_pic(path, random_id)
            if  valid_pic:
                # write label
                line = random_id + ':'
                line += path + ':'
                labels = path.split('/')[2:-1]
                for label in labels:
                    line += label + ',' 
                f.write(line+'\n')
    f.close()

def get_image_dict():
    f = open(valid_f,'r',encoding='utf-8')
    lf = open(label_f,'w', encoding='utf-8')
    mf = open(map_f,'w',encoding='utf-8')
    
    all_labels = []
    label_dict = {}
    for line in f.readlines():
        item = line.strip().split(':')
        id = item[0]
        labels = item[2].split(',')[:-1]
        label_dict[id] = labels
        for l in labels:
            all_labels.append(l)
    f.close()
    unique_label = list(set(all_labels))
    print ("Total unique label = {}".format(len(unique_label)))
    _dict = {}
    
    for i in range(len(unique_label)):
        _dict[unique_label[i]] = i
        mf.write(unique_label[i] + ' ' + str(i) + '\n')
    mf.close()

    for key in label_dict.keys():
        #onehot = np.zeros(len(unique_label))
        labels = label_dict[key]
        #for l in labels:
        #    onehot[_dict[l]] = 1
        line = key + ' '
        for l in labels:
            line += str(_dict[l]) + ','
        line += '\n'
        lf.write(line)
    lf.close()

if __name__ == '__main__':
    get_image_list() 
    get_image_dict()
