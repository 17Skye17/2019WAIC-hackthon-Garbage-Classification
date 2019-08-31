import sys
from tqdm import tqdm
_file = sys.argv[1]
f = open(_file,'r')
valid = ['recyclable','kitchen','harmful','other']
correct = 0
lines = f.readlines()
for line in tqdm(lines):
    item = line.strip().split(' ')
    gt = item[1].split('/')
    pred = item[2].split('/')
    temp_gt = []
    temp_pred = []
    for g in gt:
        if g in valid:
            temp_gt.append(g)
    for p in pred:
        if p in valid:
            temp_pred.append(p)
    if temp_pred == []:
        continue
    if temp_pred[0] in temp_gt:
        correct += 1

print ("Total Acc@1 = {}".format(correct*100.0/len(lines)))
