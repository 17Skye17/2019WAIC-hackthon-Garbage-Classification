f = open('senet154.lst','r')
wf = open('wrongs.lst','w')
correct = 0
lines = f.readlines()

def get_dict():
    #f = open("success.lst",'r',encoding='cp936')
    f = open("success.lst",'r')
    _dict = {}
    for line in f.readlines():
        item = line.strip().split(' ')
        _dict[item[0]] = item[2].split(',')[0]
    f.close()
    return _dict

gt_dict = get_dict()

for line in lines:
    item = line.strip().split(' ')
    _id = item[0].split('/')[-1].split('.')[0]
    gt = gt_dict[_id]
    # pred
    preds = item[1].split(',')
    valid = None
    for pred in preds:
        if int(pred) in [0,1,2,3]:
           valid = pred
           break
    
    if valid == gt:
        correct += 1
    else:
        wf.write(line)

#for line in lines:
#    item = line.strip().split(',')
#    _id = item[0].split('/')[-1].split('.')[0]
#    pred = item[1]
#    gt = gt_dict[_id]
#    if pred == gt:
#        correct += 1
wf.close()
f.close()
print  (correct*100.0/len(lines))
