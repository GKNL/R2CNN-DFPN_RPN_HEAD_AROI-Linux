# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
from data.io import image_preprocess
from libs.configs import cfgs


def read_single_example_and_decode(filename_queue):

    # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # reader = tf.TFRecordReader(options=tfrecord_options)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 11])

    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, img, gtboxes_and_label, num_objects


def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):
    """
    读取图片，并对图像进行处理与变换从而进行数据增强
    :param filename_queue: tf内部的queue类型，存放着全部的文件名
    :param shortside_len: 图像较短一边（宽）的长度（这里为600）
    :param is_training: 训练or测试
    :return:
    """

    img_name, img, gtboxes_and_label, num_objects = read_single_example_and_decode(filename_queue)
    # img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    img = img - tf.constant([103.939, 116.779, 123.68])
    if is_training:
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)
        img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img, gtboxes_and_label=gtboxes_and_label)

    else:
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)

    return img_name, img, gtboxes_and_label, num_objects


def next_batch(dataset_name, batch_size, shortside_len, is_training):
    """
    读出tfrecords中的图片等信息，并分割为若干个batch
    :param dataset_name:
    :param batch_size:
    :param shortside_len:
    :param is_training:
    :return:
    """
    if dataset_name not in ['UAV', 'shapan', 'airplane', 'SHIP', 'ship', 'spacenet', 'pascal', 'coco']:
        raise ValueError('dataSet name must be in pascal or coco')

    if is_training:  # 训练模式读取训练集
        pattern = os.path.join('../data/tfrecords', dataset_name + '_train*')
    else:  # 测试模式读取测试集
        pattern = os.path.join('../data/tfrecords', dataset_name + '_test*')

    print('tfrecord path is -->', os.path.abspath(pattern))
    filename_tensorlist = tf.train.match_filenames_once(pattern)  # 判断是否读取到文件
    # 使用tf.train.string_input_producer函数把我们需要的全部文件打包为一个tf内部的queue类型，之后tf开文件就从这个queue中取目录了（要注意一点的是这个函数的shuffle参数默认是True）
    filename_queue = tf.train.string_input_producer(filename_tensorlist)

    # 这里对图像进行处理与变换从而进行数据增强 ，返回的是文件名，坐标及标签，以及物体的个数。
    img_name, img, gtboxes_and_label, num_obs = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                              is_training=is_training)
    # 这里产生batch，队列最大等待数为100，多线程处理
    img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch = \
        tf.train.batch(
                       [img_name, img, gtboxes_and_label, num_obs],
                       batch_size=batch_size,
                       capacity=100,
                       num_threads=16,
                       dynamic_pad=True)
    return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch


if __name__ == "__main__":
    img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
        next_batch(dataset_name=cfgs.DATASET_NAME,
                   batch_size=cfgs.BATCH_SIZE,
                   shortside_len=cfgs.SHORT_SIDE_LEN,
                   is_training=True)

    with tf.Session() as sess:
        print(gtboxes_and_label_batch)  # Tensor("batch:2", shape=(1, ?, 11), dtype=int32)
        print(tf.squeeze(gtboxes_and_label_batch, 0))  # Tensor("Squeeze_1:0", shape=(?, 11), dtype=int32)

