import numpy as np
from dataloader import LT_Dataset

DATA_ROOT= '/vilsrv-storage/open_set_trax/lt_data'


TRAIN_PATH = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_train.txt'
VAL_PATH = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_val.txt'
TEST_PATH = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_test.txt'


TRAIN_PATH_NEW = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_train2.txt'
VAL_PATH_NEW = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_val2.txt'
TEST_PATH_NEW = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_test2.txt'
OPEN_PATH_NEW = '/home/dinari/OpenLongTailRecognition-OLTR/data/Trax_LT/Trax_LT_open2.txt'





open_set_precentile = 0.15

train_data = LT_Dataset(DATA_ROOT,TRAIN_PATH)

labels = np.unique(train_data.labels)

open_labels = np.random.choice(labels,int(np.floor(len(labels) * open_set_precentile)), replace=False)



def change_to_open(open_file,old_cur_file,new_cur_file,open_labels):
    for line in old_cur_file.readlines():
        l = line.split()
        if int(l[1]) in open_labels:
            open_file.write(line)
        else:
            new_cur_file.write(line)

open_file = open(OPEN_PATH_NEW,'w+')
old_cur_file = open(TRAIN_PATH,'r')
new_cur_file = open(TRAIN_PATH_NEW,'w+')

change_to_open(open_file,old_cur_file,new_cur_file,open_labels)

new_cur_file.close()

old_cur_file = open(VAL_PATH,'r')
new_cur_file = open(VAL_PATH_NEW,'w+')

change_to_open(open_file,old_cur_file,new_cur_file,open_labels)

new_cur_file.close()

old_cur_file = open(TEST_PATH,'r')
new_cur_file = open(TEST_PATH_NEW,'w+')

change_to_open(open_file,old_cur_file,new_cur_file,open_labels)

new_cur_file.close()

open_file.close()




print("blob")
