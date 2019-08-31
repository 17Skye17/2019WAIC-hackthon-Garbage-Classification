import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="The tfrecords dataset path")
parser.add_argument("train_ratio", type=float, help="Train dataset ratio")
args = parser.parse_args()

train_file = 'train.lst'
test_file = 'test.lst'

train_ratio = args.train_ratio

data_dir = args.data_path

trainf = open(train_file,'w')
testf = open(test_file,'w')

data = os.listdir(data_dir)
random.shuffle(data)

train_data = data[:int(len(data)*train_ratio)]
test_data = data[int(len(data)*train_ratio):]

for data in train_data:
    trainf.write(os.path.join(data_dir,data)+"\n")

for data in test_data:
    testf.write(os.path.join(data_dir,data)+"\n")

trainf.close()
testf.close()
