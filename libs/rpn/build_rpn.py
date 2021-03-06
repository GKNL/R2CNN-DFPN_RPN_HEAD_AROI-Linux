# # -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from libs.box_utils import anchor_utils_pyfunc, make_anchor, nms
from libs.box_utils import boxes_utils
from libs.box_utils import iou
from libs.box_utils import encode_and_decode
from libs.box_utils.show_box_in_tensor import draw_box_with_color
from libs.losses import losses
DEBUG = True


class RPN(object):
    def __init__(self, net_name, inputs, gtboxes_and_label,
                 is_training,
                 share_net,
                 anchor_ratios,
                 anchor_scales,
                 scale_factors,
                 base_anchor_size_list,  # P2, P3, P4, P5, P6
                 stride,  # 每一层Pi的步幅stride
                 level,
                 top_k_nms,
                 share_head=False,  # 模块间是否共享权重变量
                 rpn_nms_iou_threshold=0.7,
                 max_proposals_num=300,
                 rpn_iou_positive_threshold=0.7,
                 rpn_iou_negative_threshold=0.3,  # iou>=0.7 is positive box, iou< 0.3 is negative
                 rpn_mini_batch_size=256,
                 rpn_positives_ratio=0.5,
                 remove_outside_anchors=False,  # whether remove anchors outside
                 rpn_weight_decay=0.0001,
                 ):

        self.net_name = net_name
        self.img_batch = inputs
        self.gtboxes_and_label = gtboxes_and_label  # shape is [M. 5],

        self.base_anchor_size_list = base_anchor_size_list

        self.anchor_ratios = tf.constant(anchor_ratios, dtype=tf.float32)
        self.anchor_scales = tf.constant(anchor_scales, dtype=tf.float32)
        self.share_head = share_head
        self.num_of_anchors_per_location = len(anchor_scales) * len(anchor_ratios)

        self.scale_factors = scale_factors
        self.stride = stride  # 5个stride，对应5个Feature Map的缩放比例(例如：stride=4表示P1相对于原图，size缩小为原来的1/4)
        self.level = level
        self.top_k_nms = top_k_nms

        self.rpn_nms_iou_threshold = rpn_nms_iou_threshold
        self.max_proposals_num = max_proposals_num

        self.rpn_iou_positive_threshold = rpn_iou_positive_threshold
        self.rpn_iou_negative_threshold = rpn_iou_negative_threshold
        self.rpn_mini_batch_size = rpn_mini_batch_size
        self.rpn_positives_ratio = rpn_positives_ratio  # # RPN生成的minibatch anchor samples中，正样本所占比例（这里正:负=1:1）
        self.remove_outside_anchors = remove_outside_anchors
        self.rpn_weight_decay = rpn_weight_decay
        self.is_training = is_training
        self.share_net = share_net

        self.feature_maps_dict = self.get_feature_maps()  # 特征图s(C2, C3, C4, C5)
        if cfgs.FEATURE_PYRAMID_MODE == 0:
            self.feature_pyramid = self.build_feature_pyramid()
        else:
            self.feature_pyramid = self.build_dense_feature_pyramid()

        self.anchors, self.rpn_encode_boxes, self.rpn_scores = self.get_anchors_and_rpn_predict()

    def get_feature_maps(self):

        '''
            Compared to https://github.com/KaimingHe/deep-residual-networks,the implementation of resnet_50 in slim
            subsample the output activations in the last residual unit of each block,
            instead of subsampling the input activations in the first residual unit of each block.
            The two implementations give identical results but the implementation of slim is more memory efficient.

            SO, when we build feature_pyramid, we should modify the value of 'C_*' to get correct spatial size feature maps.
            :return: feature maps
        '''

        with tf.variable_scope('get_feature_maps'):
            if self.net_name == 'resnet_v1_50':
                feature_maps_dict = {
                    'C2': self.share_net['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # [56, 56]
                    'C3': self.share_net['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # [28, 28]
                    'C4': self.share_net['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # [14, 14]
                    'C5': self.share_net['resnet_v1_50/block4']  # [7, 7]
                }
            elif self.net_name == 'resnet_v1_101':
                feature_maps_dict = {
                    'C2': self.share_net['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # [56, 56]
                    'C3': self.share_net['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # [28, 28]
                    'C4': self.share_net['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # [14, 14]
                    'C5': self.share_net['resnet_v1_101/block4']  # [7, 7]
                }
            else:
                raise Exception('get no feature maps')

            return feature_maps_dict

    def build_dense_feature_pyramid(self):
        ''' 作者提出的DFPN：对ResNet中的各层Feature Map进行处理，得到P2,P3,P4,P5,P6
        reference: DenseNet
        build P2, P3, P4, P5, P6
        :return: multi-scale feature map
        '''

        feature_pyramid = {}
        with tf.variable_scope('dense_feature_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                # C5层先经过1 x 1卷积，将通道数改为256，得到P5
                feature_pyramid['P5'] = slim.conv2d(self.feature_maps_dict['C5'],
                                                    num_outputs=256,
                                                    kernel_size=[1, 1],
                                                    stride=1,
                                                    scope='build_P5')

                # 对P5进行max pooling 得到P6
                feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                        kernel_size=[2, 2], stride=2, scope='build_P6')
                # P6 is down sample of P5

                for layer in range(4, 1, -1):  # 依次对C4, C3, C2进行处理，得到P4, P3, P2
                    c = self.feature_maps_dict['C' + str(layer)]
                    # 以layer = 3为例，对C3进行1*1卷积，改变通道数
                    c_conv = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                         scope='build_P%d/reduce_dimension' % layer)
                    p_concat = [c_conv]
                    up_sample_shape = tf.shape(c)
                    # 下面的代码是DFPN的创新点，区别于普通的FPN
                    for layer_top in range(5, layer, -1):  # 对P5和P4分别进行上采样（以layer = 3为例时）
                        p_temp = feature_pyramid['P' + str(layer_top)]
                        # 对P_temp进行上采样
                        p_sub = tf.image.resize_nearest_neighbor(p_temp, [up_sample_shape[1], up_sample_shape[2]],
                                                                 name='build_P%d/up_sample_nearest_neighbor' % layer)
                        p_concat.append(p_sub)

                    p = tf.concat(p_concat, axis=3)  # 将P5_sub, P4_sub, C3_conv进行拼接（以layer = 3为例时）
                    # 对拼接后的结果再进行3*3的卷积（减轻最近邻近插值带来的混叠影响，周围的数都相同）
                    p_conv = slim.conv2d(p, 256, kernel_size=[3, 3], stride=[1, 1],
                                         padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                    feature_pyramid['P' + str(layer)] = p_conv

        return feature_pyramid

    def build_feature_pyramid(self):

        ''' 普通FPN
        reference: https://github.com/CharlesShang/FastMaskRCNN
        build P2, P3, P4, P5
        :return: multi-scale feature map
        '''

        feature_pyramid = {}
        with tf.variable_scope('build_feature_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                feature_pyramid['P5'] = slim.conv2d(self.feature_maps_dict['C5'],
                                                    num_outputs=256,
                                                    kernel_size=[1, 1],
                                                    stride=1,
                                                    scope='build_P5')

                feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                        kernel_size=[2, 2], stride=2, scope='build_P6')
                # P6 is down sample of P5

                for layer in range(4, 1, -1):  # 依次对C4, C3, C2进行处理，得到P4, P3, P2
                    p, c = feature_pyramid['P' + str(layer + 1)], self.feature_maps_dict['C' + str(layer)]
                    up_sample_shape = tf.shape(c)
                    # 以layer = 4为例: 对P5进行上采样使之与C4的size相同
                    up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                                 name='build_P%d/up_sample_nearest_neighbor' % layer)
                    # 对C4进行1*1卷积，改变通道数
                    c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                    scope='build_P%d/reduce_dimension' % layer)
                    p = up_sample + c  # 拼接upsample和c，得到M4特征图
                    #  对M4特征图再经过3 x 3卷积(减轻最近邻近插值带来的混叠影响，周围的数都相同)
                    p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                    padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                    feature_pyramid['P' + str(layer)] = p

        return feature_pyramid

    def make_anchors(self):
        """
        针对金字塔的每一层即相对应的feature-map生成anchors
        :return:
        """
        with tf.variable_scope('make_anchors'):
            anchor_list = []
            level_list = self.level
            with tf.name_scope('make_anchors_all_level'):
                for level, base_anchor_size, stride in zip(level_list, self.base_anchor_size_list, self.stride):
                    '''
                    (level, base_anchor_size) tuple:
                    (P2, 32), (P3, 64), (P4, 128), (P5, 256), (P6, 512)
                    '''
                    featuremap_height, featuremap_width = tf.shape(self.feature_pyramid[level])[1], \
                                                          tf.shape(self.feature_pyramid[level])[2]
                    # stride = base_anchor_size / 8.

                    # tmp_anchors = tf.py_func(
                    #     anchor_utils_pyfunc.make_anchors,
                    #     inp=[base_anchor_size, self.anchor_scales, self.anchor_ratios,
                    #          featuremap_height, featuremap_width, stride],
                    #     Tout=tf.float32
                    # )

                    # 在原图上生成anchors
                    tmp_anchors = make_anchor.make_anchors(base_anchor_size, self.anchor_scales, self.anchor_ratios,
                                                           featuremap_height,  featuremap_width, stride,
                                                           name='make_anchors_{}'.format(level))
                    # 返回的anchors集合格式：[w * h * len(anchor_scales) * len(anchor_ratios), 4]
                    tmp_anchors = tf.reshape(tmp_anchors, [-1, 4])
                    anchor_list.append(tmp_anchors)

                all_level_anchors = tf.concat(anchor_list, axis=0)
            return all_level_anchors

    def rpn_net(self):

        rpn_encode_boxes_list = []  # 所有层Feature Map经回归分支的输出
        rpn_scores_list = []
        with tf.variable_scope('rpn_net'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                # 对于DFPN的每一层的Feature Map进行操作
                for level in self.level:

                    if self.share_head:  # 初始化P2模块各层权重变量，其他模块的相应层之间共享P2的变量
                        reuse_flag = None if level == 'P2' else True
                        scope_list = ['conv2d_3x3', 'rpn_classifier', 'rpn_regressor']
                    else:  # 模块之间的相应层不共享变量
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_'+level, 'rpn_classifier_'+level, 'rpn_regressor_'+level]

                    rpn_conv2d_3x3 = slim.conv2d(inputs=self.feature_pyramid[level],
                                                 num_outputs=256,
                                                 kernel_size=[3, 3],
                                                 stride=1,
                                                 scope=scope_list[0],  # 共享变量所指的variable_scope
                                                 reuse=reuse_flag)  # 指定是否共享层或者和变量
                    """
                    接下来对整个Feature Map进行卷积操作（分类和回归分支）[注意是对整个Feature Map进行操作，而不是对Anchor进行操作]
                    因为Feature Map上每个点对应原图一个区域(k个anchor)，相当于对原图上的每个anchor进行分类和回归分支了
                    """
                    # 分类分支：对rpn_conv2d_3x3进行分类（前景/非前景分数值）
                    rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                                 num_outputs=2 * self.num_of_anchors_per_location,
                                                 kernel_size=[1, 1],
                                                 stride=1,
                                                 scope=scope_list[1],
                                                 activation_fn=None,
                                                 reuse=reuse_flag)
                    # 回归分支（移变换t_x*,t_y*和缩放尺度t_w*,t_h*）
                    rpn_encode_boxes = slim.conv2d(rpn_conv2d_3x3,
                                                   num_outputs=4 * self.num_of_anchors_per_location,  # （tx,ty,tw,th）
                                                   kernel_size=[1, 1],
                                                   stride=1,
                                                   scope=scope_list[2],
                                                   activation_fn=None,
                                                   reuse=reuse_flag)

                    rpn_box_scores = tf.reshape(rpn_box_scores, [-1, 2])  # 该卷积层输出含义：表示当前位置是否含有目标 [background score、object score]
                    rpn_encode_boxes = tf.reshape(rpn_encode_boxes, [-1, 4])  # 该卷积层输出含义：移变换t_x*,t_y*和缩放尺度t_w*,t_h*

                    rpn_scores_list.append(rpn_box_scores)
                    rpn_encode_boxes_list.append(rpn_encode_boxes)

                rpn_all_encode_boxes = tf.concat(rpn_encode_boxes_list, axis=0)
                rpn_all_boxes_scores = tf.concat(rpn_scores_list, axis=0)

            return rpn_all_encode_boxes, rpn_all_boxes_scores

    def get_anchors_and_rpn_predict(self):
        """
        获得RPN网络输出的anchors，以及分类分支和回归分支（框回归坐标）的计算值
        :return:
        """

        anchors = self.make_anchors()  # 计算anchors
        rpn_encode_boxes, rpn_scores = self.rpn_net()  # 在Feature Map上计算分类与回归分支

        with tf.name_scope('get_anchors_and_rpn_predict'):
            if self.is_training:
                if self.remove_outside_anchors:  # 若True，则过滤掉超出边界的anchor
                    valid_indices = boxes_utils.filter_outside_boxes(boxes=anchors,
                                                                     img_h=tf.shape(self.img_batch)[1],
                                                                     img_w=tf.shape(self.img_batch)[2])
                    valid_anchors = tf.gather(anchors, valid_indices)
                    rpn_valid_encode_boxes = tf.gather(rpn_encode_boxes, valid_indices)
                    rpn_valid_scores = tf.gather(rpn_scores, valid_indices)

                    return valid_anchors, rpn_valid_encode_boxes, rpn_valid_scores
                else:
                    return anchors, rpn_encode_boxes, rpn_scores
            else:
                return anchors, rpn_encode_boxes, rpn_scores

    def rpn_find_positive_negative_samples(self, anchors):
        ''' 对输入的anchor区分正负样本
        assign anchors targets: object or background.
        :param anchors（在所有level的Feature Map上生成的anchor）: [valid_num_of_anchors, 4]. use N to represent valid_num_of_anchors

        :return:labels. anchors_matched_gtboxes, object_mask

        labels shape is [N, ].  positive is 1, negative is 0, ignored is -1
        anchor_matched_gtboxes. each anchor's gtbox(only positive box has gtbox)shape is [N, 4]
        object_mask. tf.float32. 1.0 represent box is object, 0.0 is others. shape is [N, ]
        '''
        with tf.variable_scope('rpn_find_positive_negative_samples'):
            gtboxes = tf.reshape(self.gtboxes_and_label[:, :-1], [-1, 4])  # 训练集中标注好的ground true anchors
            gtboxes = tf.cast(gtboxes, tf.float32)

            # 计算anchor与gt的交并比IOU
            ious = iou.iou_calculate(anchors, gtboxes)  # [N, M]  N代表valid_num_of_anchors，M代表gtboxes的个数

            max_iou_each_row = tf.reduce_max(ious, axis=1)  # [1,N] 找出ious矩阵中每一行上的最大值 => 每个anchor与M个gt中最大的交并比

            labels = tf.ones(shape=[tf.shape(anchors)[0], ], dtype=tf.float32) * (-1)  # [N, ] # ignored is -1

            matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)  # [1,N] 比较每一行的元素，将每一行最大元素所在的索引记录下来

            anchors_matched_gtboxes = tf.gather(gtboxes, matchs)  # [N, 4]  每个anchor所对应的gt（两者IOU计算值最大）

            # 计算正样本 an anchor that has an IoU overlap higher than 0.7 with any ground-truth box
            positives1 = tf.greater_equal(max_iou_each_row, self.rpn_iou_positive_threshold)  # iou >= 0.7 is positive （[1,N]的bool矩阵）

            # to avoid none of boxes iou >= 0.7, use max iou boxes as positive
            max_iou_each_column = tf.reduce_max(ious, 0)  # [1,M] 找出ious矩阵每一列中的最大值 => 每个gt与若干个anchor相交，其中IOU最大的值
            # the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box
            positives2 = tf.reduce_sum(tf.cast(tf.equal(ious, max_iou_each_column), tf.float32), axis=1)  # [1,N]的bool数组 => 前面找出了每个gt有anchor存在的最大IOU，这个数组表示 这些最大IOU是由哪个anchor产生的

            positives = tf.logical_or(positives1, tf.cast(positives2, tf.bool))  # 求正样本并集

            labels += 2 * tf.cast(positives, tf.float32)  # Now, positive is 1, ignored and background is -1

            # 计算负样本
            negatives = tf.less(max_iou_each_row, self.rpn_iou_negative_threshold)
            negatives = tf.logical_and(negatives, tf.greater_equal(max_iou_each_row, 0.1))

            labels = labels + tf.cast(negatives, tf.float32)  # [N, ] positive is >=1.0, negative is 0, ignored is -1.0
            '''
                Need to note: when positive, labels may >= 1.0.
                Because, when all the iou< 0.7, we set anchors having max iou each column as positive.
                these anchors may have iou < 0.3.
                In the begining, labels is [-1, -1, -1...-1]
                then anchors having iou<0.3 as well as are max iou each column will be +1.0.
                when decide negatives, because of iou<0.3, they add 1.0 again.
                So, the final result will be 2.0
    
                So, when opsitive, labels may in [1.0, 2.0]. that is labels >=1.0
            '''
            positives = tf.cast(tf.greater_equal(labels, 1.0), tf.float32)
            ignored = tf.cast(tf.equal(labels, -1.0), tf.float32) * -1

            labels = positives + ignored
            object_mask = tf.cast(positives, tf.float32)  # 1.0 is object, 0.0 is others

            return labels, anchors_matched_gtboxes, object_mask

    def make_minibatch(self, valid_anchors):
        """
        在所有anchor中选出小部分样本（正负比为1:1）作为mini batch
        :param valid_anchors: RPN产生的anchors（如果remove_outside_anchors=True,则是移除超出边界的anchor）
        :return: minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, labels_one_hot

        """
        with tf.variable_scope('rpn_minibatch'):

            # in labels(shape is [N, ]): 1 is positive, 0 is negative, -1 is ignored
            labels, anchor_matched_gtboxes, object_mask = \
                self.rpn_find_positive_negative_samples(valid_anchors)  # [num_of_valid_anchors, ]

            positive_indices = tf.reshape(tf.where(tf.equal(labels, 1.0)), [-1])  # use labels is same as object_mask

            # 正样本占整个minibatch size的1/2
            num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                          tf.cast(self.rpn_mini_batch_size * self.rpn_positives_ratio, tf.int32))

            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices,
                                        begin=[0],
                                        size=[num_of_positives])

            negatives_indices = tf.reshape(tf.where(tf.equal(labels, 0.0)), [-1])
            num_of_negatives = tf.minimum(self.rpn_mini_batch_size - num_of_positives,
                                          tf.shape(negatives_indices)[0])

            negatives_indices = tf.random_shuffle(negatives_indices)
            negatives_indices = tf.slice(negatives_indices, begin=[0], size=[num_of_negatives])

            minibatch_indices = tf.concat([positive_indices, negatives_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_anchor_matched_gtboxes = tf.gather(anchor_matched_gtboxes, minibatch_indices)
            object_mask = tf.gather(object_mask, minibatch_indices)
            labels = tf.cast(tf.gather(labels, minibatch_indices), tf.int32)
            labels_one_hot = tf.one_hot(labels, depth=2)
            return minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, labels_one_hot

    def rpn_losses(self):
        """
        Bounding Box Regression实现部分！！！
        1.读取小批量anchor样本。然后根据gt和RPN输出的anchors计算平移和缩放因子，与RPN回归分支输出结果一起，计算回归loss
        2.画出图片中的正样本和负样本
        3.根据回归分支输出的平移和缩放因子，与RPN输出anchor进行计算，得到pred box，并画出分数TopK的pred box
        4.计算分类loss
        (注：encode_***代表平移和缩放尺度因子、decode_***代表框坐标)

        :return:
        """
        with tf.variable_scope('rpn_losses'):
            minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, minibatch_labels_one_hot = \
                self.make_minibatch(self.anchors)

            minibatch_anchors = tf.gather(self.anchors, minibatch_indices)
            minibatch_encode_boxes = tf.gather(self.rpn_encode_boxes, minibatch_indices)  # 回归分支输出：平移和缩放尺度因子
            minibatch_boxes_scores = tf.gather(self.rpn_scores, minibatch_indices)

            # encode gtboxes
            # 根据anchor box的[x,y,w,h]和gt box的[x*,y*,w*,h*]，计算得到平移变换t_x*,t_y*和缩放尺度t_w*,t_h*(RPN回归分支学习的目标)
            minibatch_encode_gtboxes = encode_and_decode.encode_boxes(unencode_boxes=minibatch_anchor_matched_gtboxes,
                                                                      reference_boxes=minibatch_anchors,
                                                                      scale_factors=self.scale_factors)
            # 画出图片中的正样本
            positive_anchors_in_img = draw_box_with_color(self.img_batch,
                                                          minibatch_anchors * tf.expand_dims(object_mask, 1),
                                                          text=tf.shape(tf.where(tf.equal(object_mask, 1.0)))[0])
            # 画出图片中的负样本
            negative_mask = tf.cast(tf.logical_not(tf.cast(object_mask, tf.bool)), tf.float32)
            negative_anchors_in_img = draw_box_with_color(self.img_batch,
                                                          minibatch_anchors * tf.expand_dims(negative_mask, 1),
                                                          text=tf.shape(tf.where(tf.equal(object_mask, 0.0)))[0])

            # 根据anchor box的[x,y,w,h]和RPN回归分支输出的平移变换t_x*,t_y*和缩放尺度t_w*,t_h*，计算得出pred box
            minibatch_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=minibatch_encode_boxes,
                                                                    reference_boxes=minibatch_anchors,
                                                                    scale_factors=self.scale_factors)

            tf.summary.image('/positive_anchors', positive_anchors_in_img)
            tf.summary.image('/negative_anchors', negative_anchors_in_img)
            # 取分数最高的前20
            top_k_scores, top_k_indices = tf.nn.top_k(minibatch_boxes_scores[:, 1], k=20)

            # 在图中画出分数最高的前topK个回归框
            top_detections_in_img = draw_box_with_color(self.img_batch,
                                                        tf.gather(minibatch_decode_boxes, top_k_indices),
                                                        text=tf.shape(top_k_scores)[0])
            tf.summary.image('/top_20', top_detections_in_img)

            # losses
            with tf.variable_scope('rpn_location_loss'):
                location_loss = losses.l1_smooth_losses(predict_boxes=minibatch_encode_boxes,  # 回归分支输出的平移和缩放尺度因子
                                                        gtboxes=minibatch_encode_gtboxes,  # 由anchor和gt anchor计算得到的平移和缩放尺度因子
                                                        object_weights=object_mask)
                slim.losses.add_loss(location_loss)  # add smooth l1 loss to losses collection

            with tf.variable_scope('rpn_classification_loss'):
                classification_loss = slim.losses.softmax_cross_entropy(logits=minibatch_boxes_scores,  # RPN分类分支的输出
                                                                        onehot_labels=minibatch_labels_one_hot)  # 训练集gt的label

            return location_loss, classification_loss

    def rpn_proposals(self):
        """
        根据scores返回最高的几个框，然后对这几个框根据IOU（大于0.7的视为不错）进行NMS处理，返回index，
        然后根据Index挑选框（优秀选手）,返回proposals(优秀选手)及他们的scores(成绩)
        :return:
        """
        with tf.variable_scope('rpn_proposals'):
            # RPN回归分支得到的pred box
            rpn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=self.rpn_encode_boxes,  # 回归分支输出的四个因子
                                                              reference_boxes=self.anchors,
                                                              scale_factors=self.scale_factors)

            if not self.is_training:  # when test, clip proposals to img boundaries  根据image边界对proposal的超出部分进行裁剪
                img_shape = tf.shape(self.img_batch)
                rpn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(rpn_decode_boxes, img_shape)

            rpn_softmax_scores = slim.softmax(self.rpn_scores)  # 对分类分支的输出分数通过softmax进行归一化处理
            rpn_object_score = rpn_softmax_scores[:, 1]  # [background score、object score] => second column represent object

            if self.top_k_nms:  # 根据score返回object score最高的几个框
                rpn_object_score, top_k_indices = tf.nn.top_k(rpn_object_score, k=self.top_k_nms)
                rpn_decode_boxes = tf.gather(rpn_decode_boxes, top_k_indices)

            # 根据IOU（大于0.7的视为不错）进行NMS处理，返回留下的box的index
            valid_indices = nms.non_maximal_suppression(boxes=rpn_decode_boxes,
                                                        scores=rpn_object_score,
                                                        max_output_size=self.max_proposals_num,
                                                        iou_threshold=self.rpn_nms_iou_threshold)

            valid_boxes = tf.gather(rpn_decode_boxes, valid_indices)
            valid_scores = tf.gather(rpn_object_score, valid_indices)
            # print_tensors(valid_scores, 'rpn_score')
            rpn_proposals_boxes, rpn_proposals_scores = tf.cond(
                tf.less(tf.shape(valid_boxes)[0], self.max_proposals_num),
                lambda: boxes_utils.padd_boxes_with_zeros(valid_boxes, valid_scores,
                                                          self.max_proposals_num),
                lambda: (valid_boxes, valid_scores))

            return rpn_proposals_boxes, rpn_proposals_scores
