# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os

# root path
ROO_PATH = os.path.abspath('/home/20184868@software.com/PM/pycharmProjects/R2CNN-DFPN_RPN_HEAD_AROI-Linux')

# 预训练权重文件路径
MODEL_PATH = ROO_PATH + '/output/model'
SUMMARY_PATH = ROO_PATH + '/output/summary'

TEST_SAVE_PATH = ROO_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROO_PATH + '/tools/inference_image'  # 测试输入图片路径
INFERENCE_SAVE_PATH = ROO_PATH + '/tools/inference_result'  # 测试结果保存路径

NET_NAME = 'resnet_v1_101'  # share net的name
VERSION = 'ship_head_2021_3_12'
CLASS_NUM = 1  # exclude background
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]  # P越高，RPN生成的anchor数越少，对应在原图中anchor的size也应该越大
STRIDE = [4, 8, 16, 32, 64]  # 5个stride，对应5个Feature Map的缩放比例(例如：stride=4表示P1相对于原图，size缩小为原来的1/4)
ANCHOR_SCALES = [1.]
ANCHOR_RATIOS = [1 / 2., 1 / 3., 1., 3., 2.]
SCALE_FACTORS = [10., 10., 5., 5., 10.]
OUTPUT_STRIDE = 16
SHORT_SIDE_LEN = 600  # 宽600（较短的一边）
DATASET_NAME = 'ship'

BATCH_SIZE = 1
WEIGHT_DECAY = {'vggnet16': 0.0005, 'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001}
EPSILON = 1e-5
MOMENTUM = 0.9
MAX_ITERATION = 60000  # 40000
GPU_GROUP = "0"

# rpn
SHARE_HEAD = False
RPN_NMS_IOU_THRESHOLD = 0.7
MAX_PROPOSAL_NUM = 300
RPN_IOU_POSITIVE_THRESHOLD = 0.6
RPN_IOU_NEGATIVE_THRESHOLD = 0.25
RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5  # RPN生成的minibatch anchor samples中，正样本所占比例（这里正:负=1:1）
IS_FILTER_OUTSIDE_BOXES = True
RPN_TOP_K_NMS = 3000
FEATURE_PYRAMID_MODE = 0  # {0: 'feature_pyramid', 1: 'dense_feature_pyramid'}

# fast rcnn
ROTATE_NMS_USE_GPU = True
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 0.5
FAST_RCNN_NMS_IOU_THRESHOLD = 0.15
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FINAL_SCORE_THRESHOLD = 0.8
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_MINIBATCH_SIZE = 128
FAST_RCNN_POSITIVE_RATE = 0.5
