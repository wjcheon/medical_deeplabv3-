import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import string

def weight_variable(shape_, name=None):
    #print(np.shape(shape_))

    #initial = tf.truncated_normal(shape_, stddev=0.6)
    #initial = tf.random_normal(shape_, stddev=0.5)
    if np.ndim(shape_) is 1:
        initial = tf.contrib.layers.xavier_initializer()
    else:
        initial = tf.constant_initializer(0.0)

    variable_return = tf.get_variable(name, shape=shape_, initializer=initial)
    #return tf.Variable(initial, name=name)
    return variable_return

def softmax_layer(inpt_, shape_):
    fc_w = weight_variable(shape_)
    fc_b = tf.Variable(tf.zeros([shape_[1]]))
    fc_h = tf.nn.softmax(tf.matmul(inpt_, fc_w) + fc_b)

    return fc_h

def conv_layer(inpt_, filter_shape_, stride_, af_='relu', name=None):
    out_channels = filter_shape_[3]
    #
    name_filter = name + '_filter'
    name_bias = name + '_bias'
    filter_ = weight_variable(filter_shape_, name=name_filter)
    conv = tf.nn.conv2d(inpt_, filter=filter_, strides=[1, stride_, stride_, 1], padding="SAME")
    conv_bias = weight_variable([out_channels], name=name_bias)
    # Batch-normalization
    # mean, var = tf.nn.moments(conv, axes=[0,1,2])
    # beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    # gamma = weight_variable([out_channels], name="gamma")
    #
    # batch_norm = tf.nn.batch_norm_with_global_normalization(
    #     conv, mean, var, beta, gamma, 0.001,
    #     scale_after_normalization=True)
    # out = tf.nn.relu(batch_norm)
    if af_ is 'relu':
        out = tf.nn.relu(conv+conv_bias)
        print('activation function is ReLu')
    elif af_ is 'sigmoid':
        out = tf.nn.sigmoid(conv+conv_bias)
        print('activation function is sigmoid')
    elif af_ is 'None':
        out=conv+conv_bias
        print('activation function is None')

    return out


def trans_conv_layer(inpt_, output_shape_, kernel_size_= [2, 2], strides_= [2, 2], af_='relu', name=None):
    # conv1 (64 x 64, 128) -> (128 x 128, 64)
    # filter_shape_ =
    # original: [kernel, kernel, deconv_output_channel, deconv_input_channel]
    inpt_channel_batch = tf.shape(inpt_)[0]
    inpt_channel =  inpt_.get_shape().as_list()[3]
    kernel_shape = [kernel_size_[0], kernel_size_[1], output_shape_[2], inpt_channel]

    name_filter = name + '_filter'
    name_bias = name + '_bias'
    deconv_filter = tf.get_variable(name_filter, shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
    deconv_bias = tf.get_variable(name_bias, shape=kernel_shape[2], initializer=tf.contrib.layers.xavier_initializer())
    deconv_temp = tf.nn.conv2d_transpose(inpt_, deconv_filter, [inpt_channel_batch, output_shape_[0], output_shape_[1], output_shape_[2]], strides=[1, strides_[0], strides_[1], 1], padding='SAME')

    if af_ is 'relu':
        out = tf.nn.relu(deconv_temp + deconv_bias)
        print('activation function is ReLu')
    elif af_ is 'sigmoid':
        out = tf.nn.sigmoid(deconv_temp + deconv_bias)
        print('activation function is sigmoid')
    elif af_ is 'None':
        out=(deconv_temp + deconv_bias)
        print('activation function is None')

    return out


def residual_block(inpt_, output_depth_, down_sample_, projection=False, name=None):
    input_depth_ = inpt_.get_shape().as_list()[3]
    #
    name_residual_conv1 = name + '_residual_conv1'
    name_residual_conv2 = name + '_residual_conv2'
    #
    if down_sample_:
        # max pooling with stride 'two'
        filter_ = [1,2,2,1]
        inpt_ = tf.nn.max_pool(inpt_, ksize=filter_, strides=filter_, padding='SAME')
    # Convolution with 3 x 3 filter
    conv1 = conv_layer(inpt_, [3, 3, input_depth_, output_depth_], 1, name= name_residual_conv1)
    #print(np.shape(conv1))
    conv2 = conv_layer(conv1, [3, 3, output_depth_, output_depth_], 1, name=name_residual_conv2)

    if input_depth_ != output_depth_:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt_, [1, 1, input_depth_, output_depth_], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt_, [[0,0], [0,0], [0,0], [0, output_depth_ - input_depth_]])
    else:
        input_layer = inpt_

    res = tf.nn.relu(conv2 + input_layer)
    return res


def atrous_spatial_pyramid_pooling(net, depth=256, reuse=None):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """
    feature_map_size = tf.shape(net)
    # apply global average pooling
    image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
    image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1_in_ASPP",
                                       activation_fn=None)
    image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

    at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0_in_ASPP", activation_fn=None)

    at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1_in_ASPP", rate=6, activation_fn=None)

    at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2_in_ASPP", rate=12, activation_fn=None)

    at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3_in_ASPP", rate=18, activation_fn=None)

    net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                    name="concat_in_ASPP")
    net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
    return net
