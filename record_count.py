import glob
import sys
import os
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import cv2

from collections import defaultdict

def usage():
    print('Usage: ' + sys.argv[0] + ' <records_dir_path>')

if __name__ == '__main__' :
    argc = len(sys.argv)
    if argc < 2:
        usage()
        quit()
    
    input_path = sys.argv[1]
    
    files = glob.glob(input_path + '/*.tfrecord')
    objs = defaultdict(int)
    if len(files) == 0:
        print('invalid input path : ' + input_path)
        quit(-1)
    print('input files  : {}'.format(input_path + '/*.tfrecord'))

    
    for record_file in files:
        record_iterator = tf.python_io.tf_record_iterator(record_file)
        
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            labels = example.features.feature["image/object/class/label"].int64_list.value

            for label in labels:
                objs[label] += 1
    
    print(" --- object count --- ")
    for k, v in objs.items() :
        print("{} : {} ".format(k,v))
