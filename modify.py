import os
_map = {"0":399, "1":400, "2":401, "3":402}
f = open("success.lst",'r')
sf = open("modified.lst",'w')

for line in f.readlines():
    item = line.strip().split(' ')
    labels = item[2].split(',')
    label2 = labels[1]
    if label2 in _map.keys():
        label2 = str(_map[label2])
    newline = item[0] + ' ' + item[1] + ' ' + labels[0] + ',' + label2 + '\n'
    sf.write(newline)

f.close()
sf.close()
