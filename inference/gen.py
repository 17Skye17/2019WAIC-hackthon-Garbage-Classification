import os
files = os.listdir('eval')
f = open('test.txt','w')
for _file in files:
    f.write(os.path.join('./eval',_file)+'\n')
