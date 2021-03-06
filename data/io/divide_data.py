# -*- coding: utf-8 -*-
# @Time    : 2020/11/29 16:24
# @Author  : Peng Miao
# @File    : divide_data.py
# @Intro   : 分割数据集，将原始的图片数据集和标注数据集，按照一定的比例，分别对应地划分为train和test数据集

from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import shutil
import os
import random
import math


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

divide_rate = 0.9

root_path = 'D:/ideaWorkPlace/Pycharm/graduation_project/R2CNN-DFPN_RPN_HEAD_AROI/data'

image_path = root_path + '/VOCdevkit/JPEGImages'
xml_path = root_path + '/VOCdevkit/Annotations'

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_train = os.path.join(root_path, 'VOCdevkit_train/JPEGImages')
mkdir(image_output_train)
image_output_test = os.path.join(root_path, 'VOCdevkit_test/JPEGImages')
mkdir(image_output_test)

xml_train = os.path.join(root_path, 'VOCdevkit_train/Annotations')
mkdir(xml_train)
xml_test = os.path.join(root_path, 'VOCdevkit_test/Annotations')
mkdir(xml_test)


count = 0
for i in train_image:
    shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_train)
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_train)
    if count % 100 == 0:
        print("process step {}".format(count))
    count += 1

for i in test_image:
    shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_test)
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_test)
    if count % 100 == 0:
        print("process step {}".format(count))
    count += 1








