##
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.contrib.training.python.training.sequence_queueing_state_saver import _padding
print('Successfully import the pakages')

## Load DICOM image and RT-Structure
#root = tk.Tk()
#root.withdraw()
# file_path_CT = filedialog.askopenfilename(filetypes=[("Select CT images","*.mat")])
# file_path_brain = filedialog.askopenfilename(filetypes=[("Select BRAIN OAR","*.mat")])
# file_path_rt_lung = filedialog.askopenfilename(filetypes=[("Select RT-LUNG OAR","*.mat")])
# file_path_lt_lung = filedialog.askopenfilename(filetypes=[("Select LT-LUNG OAR","*.mat")])

file_path_CT = '/home/user/DB2T/OAR/CSI_1/mat-file-resized/RS1.2.752.243.1.1.20180720132437372.4240.73430_CT_img_resized.mat'
file_path_brain = '/home/user/DB2T/OAR/CSI_1/mat-file-resized/RS1.2.752.243.1.1.20180720132437372.4240.73430_BRAIN_resized.mat'
file_path_rt_lung = '/home/user/DB2T/OAR/CSI_1/mat-file-resized/RS1.2.752.243.1.1.20180720132437372.4240.73430_RT LUNG_resized.mat'
file_path_lt_lung = '/home/user/DB2T/OAR/CSI_1/mat-file-resized/RS1.2.752.243.1.1.20180720132437372.4240.73430_LT LUNG_resized.mat'
#
import h5py
with h5py.File(file_path_CT, 'r',  libver='latest') as file:
    print(list(file.keys()))
    CT_img = list(file['CT_img'])

with h5py.File(file_path_brain, 'r') as file:
    print(list(file.keys()))
    RT_Structure_Brain = list(file['target_GT_mask'])

with h5py.File(file_path_rt_lung, 'r') as file:
    print(list(file.keys()))
    RT_Structure_rt_lung = list(file['target_GT_mask'])

with h5py.File(file_path_lt_lung, 'r') as file:
    print(list(file.keys()))
    RT_Structure_lt_lung = list(file['target_GT_mask'])

#
#
#
mean_val = 950
std_val = 250
mean_val = 0.0
std_val = 1.0
# CT image normalization
CT_img = np.asanyarray(CT_img)
CT_img_standardization = (CT_img-mean_val)/std_val
CT_img_standardization_expand_dim = np.expand_dims(CT_img_standardization,3)
CT_img_standardization_expand_dim_float32 = np.float32(CT_img_standardization_expand_dim)

# Brain
RT_Structure_Brain = np.asanyarray(RT_Structure_Brain)
RT_Structure_Brain_expand_dim = np.expand_dims(RT_Structure_Brain , axis = 3)

# LUNG
RT_Structure_rt_lung  = np.asanyarray(RT_Structure_rt_lung )
RT_Structure_lt_lung  = np.asanyarray(RT_Structure_lt_lung )
RT_Structure_Lung = [sum(x) for x in zip(RT_Structure_rt_lung, RT_Structure_lt_lung)]
RT_Structure_Lung = np.asanyarray(RT_Structure_Lung)
# VOC 2012 format
RT_Structure_Lung[RT_Structure_Lung==1.0] = 2
RT_Structure_Lung_expand_dim = np.expand_dims(RT_Structure_Lung, axis=3)
RT_Structure_BG_Brain_Lung = [sum(x) for x in zip(RT_Structure_Brain, RT_Structure_Lung)]
RT_Structure_BG_Brain_Lung_expand_dim = np.expand_dims(RT_Structure_BG_Brain_Lung, axis=3)
##
train_X = CT_img_standardization_expand_dim_float32
train_Y = RT_Structure_BG_Brain_Lung_expand_dim
# train_Y =  []
# tarin_Y_slice =[]
# for x in zip(RT_Structure_Brain_expand_dim, RT_Structure_BG_Brain_Lung_expand_dim):
#     print(np.shape(x[0]))
#     print(np.shape(x[1]))
#     tarin_Y_slice = np.squeeze(np.stack((x[0],x[1]), axis=2),axis=3)
#     train_Y.append(tarin_Y_slice)
#train_Y = np.asanyarray(train_Y)


# sample_train_Y = train_Y[130,:,:, 1]
# plt.figure()
# plt.imshow(sample_train_Y)
print('CT images and RT-Structure were successfully loaded!')
print("The shape of trainX {}".format(np.shape(train_X)))
print("The shape of trainY {}".format(np.shape(train_Y)))
print("The trainY consist of {}".format(np.unique(train_Y)))

## Image debug using my-eyes
# size_RT_Structure = np.shape(RT_Structure_Lung )
# Check_target = CT_img_standardization_expand_dim_float32
# Check_target = np.squeeze(Check_target )
# Check_target = RT_Structure_Lung
# # You probably won't need this if you're embedding things in a tkinter plot...
# plt.ion()
# slice_mean = []
# for num_slice_iter in range(0,size_RT_Structure[0]):
#     target_slice = Check_target[num_slice_iter];
#     plt.imshow(target_slice )
#     slice_mean.append(np.mean(target_slice ))
#     plt.draw()
#     plt.pause(0.0000001)
#     plt.clf
#     print(num_slice_iter)
##

import tensorflow as tf
import tensorflow.contrib.slim as slim
from resnet_module import softmax_layer, conv_layer, residual_block, trans_conv_layer

is_train = True
learning_rate = 0.001
training_epoches = 15
batch_size = 10
n = 20

if n < 20 or (n - 20) % 12 != 0:
    print ("ResNet depth invalid.")


num_conv = int((n - 20) / 12 + 1)

X_tensor = tf.placeholder(tf.float32, [None, 256, 256, 1])
Y_tensor = tf.placeholder(tf.float32, [None, 256, 256, 1])
##
conv1_channel_num = 16
conv2_channel_num = 16
conv3_channel_num = 64
conv4_channel_num = 128
conv5_channel_num = 256

with tf.variable_scope('conv1_var_scope'):
    conv1 = conv_layer(X_tensor, [7, 7, 1, conv1_channel_num], 2, af_='relu',name='conv1')


for i in range(num_conv):
    with tf.variable_scope('conv2_%d_var_scope' % (i + 1)):
        conv2_1 = residual_block(conv1, conv2_channel_num, down_sample_=True, name = 'conv2_1')
        conv2 = residual_block(conv2_1, conv2_channel_num, False, name = 'conv2')

for i in range(num_conv):
    down_sample = True if i == 0 else False
    print(down_sample)
    with tf.variable_scope('conv3_%d' % (i + 1)):
        conv3_x = residual_block(conv2, conv3_channel_num, down_sample, name='conv3_1')
        conv3 = residual_block(conv3_x, conv3_channel_num, False, name='conv3')


for i in range(num_conv):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i + 1)):
        conv4_x = residual_block(conv3, conv4_channel_num, down_sample, name='conv4_1')
        conv4 = residual_block(conv4_x, conv4_channel_num, False, name='conv4')



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



with tf.variable_scope('ASPP'):
    conv4_ASSP =  atrous_spatial_pyramid_pooling(conv4)


# with tf.variable_scope('Decoder_step_1'):
#     # Decoder_bilinear_up_output_stride_4
#     conv5_bilinear_upsampling = tf.image.resize_bilinear(conv5, [64, 64])
#     # Dimension reduction of Conv2 -> 48 or 32
#     conv2_dim_reduction = slim.conv2d(conv2, 48, [1, 1],padding='SAME', activation_fn=None)
#     # Concat
#     conv_6_concat = tf.concat((conv5_bilinear_upsampling, conv2_dim_reduction), axis=3)
#     # Conv (3x3, 256) x 2
#     conv_6_concat_conv1 = slim.conv2d(conv_6_concat , 256, [3, 3])
#     conv_6= slim.conv2d(conv_6_concat_conv1 , 256, [3, 3])
#
#
# with tf.variable_scope('Decoder_step_2'):
#     #conv7_dim_reduction = slim.conv2d(conv_6, 2, [1, 1], padding='SAME')
#     conv7_dim_reduction = slim.conv2d(conv_6, 3, [1, 1], padding='SAME')
#
#     conv7 = tf.image.resize_bilinear(conv7_dim_reduction, [256, 256])
#
# logits = conv7


# %%
# with tf.variable_scope("deconv5") as scope:
wd5 = tf.get_variable('wd5', shape=[2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
bd5 = tf.get_variable('bd5', shape=[128], initializer=tf.contrib.layers.xavier_initializer())
tmp5 = tf.nn.conv2d_transpose(conv4_ASSP, wd5, [tf.shape(X_tensor)[0], 32, 32, 128], strides=[1, 2, 2, 1], padding='SAME')
deconv5 = tf.nn.relu(tmp5 + bd5)  # (?, 64, 64, 128)
deconv5_concat = tf.concat([conv3, deconv5], 3)  # (?, 64, 64, 256)

w5 = tf.get_variable('w5', shape=[3, 3, 192, 128], initializer=tf.contrib.layers.xavier_initializer())
w5_ = tf.get_variable('w5_', shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.get_variable('b5', shape=[128], initializer=tf.constant_initializer(0.0))
b5_ = tf.get_variable('b5_', shape=[128], initializer=tf.constant_initializer(0.0))
conv5 = tf.nn.conv2d(deconv5_concat, w5, strides=[1, 1, 1, 1], padding='SAME')  # (?,64,64,128)
rconv5 = tf.nn.relu(conv5 + b5)
conv5_ = tf.nn.conv2d(rconv5, w5_, strides=[1, 1, 1, 1], padding='SAME')  # (?,64,64,128)
rconv5_ = tf.nn.relu(conv5_ + b5_)

# with tf.variable_scope("deconv6") as scope:
wd6 = tf.get_variable('wd6', shape=[2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
bd6 = tf.get_variable('bd6', shape=[64], initializer=tf.contrib.layers.xavier_initializer())
tmp6 = tf.nn.conv2d_transpose(rconv5_, wd6, [tf.shape(X_tensor)[0], 64, 64, 64], strides=[1, 2, 2, 1], padding='SAME')
deconv6 = tf.nn.relu(tmp6 + bd6)  # (?,128,128,64)
deconv6_concat = tf.concat([conv2, deconv6], 3)  # (?,128,128,128)
w6 = tf.get_variable('w6', shape=[3, 3, 80, 64], initializer=tf.contrib.layers.xavier_initializer())
w6_ = tf.get_variable('w6_', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.get_variable('b6', shape=[64], initializer=tf.constant_initializer(0.0))
b6_ = tf.get_variable('b6_', shape=[64], initializer=tf.constant_initializer(0.0))
conv6 = tf.nn.conv2d(deconv6_concat, w6, strides=[1, 1, 1, 1], padding='SAME')  # (?,128,128,64)
rconv6 = tf.nn.relu(conv6 + b6)
conv6_ = tf.nn.conv2d(rconv6, w6_, strides=[1, 1, 1, 1], padding='SAME')  # (?,128,128,64)
rconv6_ = tf.nn.relu(conv6_ + b6_)

# with tf.variable_scope("deconv6") as scope:
wd7 = tf.get_variable('wd7', shape=[2, 2, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
bd7 = tf.get_variable('bd7', shape=[32], initializer=tf.contrib.layers.xavier_initializer())
tmp7 = tf.nn.conv2d_transpose(rconv6_, wd7, [tf.shape(X_tensor)[0], 128, 128, 32], strides=[1, 2, 2, 1], padding='SAME')
deconv7 = tf.nn.relu(tmp7 + bd7)  # (?,256,256,32)
deconv7_concat = tf.concat([conv1, deconv7], 3)  # (?,256,256,64)
w7 = tf.get_variable('w7', shape=[3, 3, 48, 32], initializer=tf.contrib.layers.xavier_initializer())
w7_ = tf.get_variable('w7_', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.get_variable('b7', shape=[32], initializer=tf.constant_initializer(0.0))
b7_ = tf.get_variable('b7_', shape=[32], initializer=tf.constant_initializer(0.0))
conv7 = tf.nn.conv2d(deconv7_concat, w7, strides=[1, 1, 1, 1], padding='SAME')  # (?,256,256,32)
rconv7 = tf.nn.relu(conv7 + b7)
conv7_ = tf.nn.conv2d(rconv7, w7_, strides=[1, 1, 1, 1], padding='SAME')  # (?,256,256,32)
rconv7_ = tf.nn.relu(conv7_ + b7_)

#
wd8 = tf.get_variable('wd8', shape=[2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
bd8 = tf.get_variable('bd8', shape=[16], initializer=tf.contrib.layers.xavier_initializer())
tmp8 = tf.nn.conv2d_transpose(rconv7_, wd8, [tf.shape(X_tensor)[0], 256, 256, 16], strides=[1, 2, 2, 1], padding='SAME')
deconv8 = tf.nn.relu(tmp8 + bd8)  # (?,512,512,16)
#deconv8_concat = tf.concat([rconv0_, deconv8], 3)  # (?,512,512,32)
w8 = tf.get_variable('w8', shape=[3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
w8_ = tf.get_variable('w8_', shape=[3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.get_variable('b8', shape=[16], initializer=tf.constant_initializer(0.0))
b8_ = tf.get_variable('b8_', shape=[16], initializer=tf.constant_initializer(0.0))
conv8 = tf.nn.conv2d(deconv8, w8, strides=[1, 1, 1, 1], padding='SAME')
rconv8 = tf.nn.relu(conv8 + b8)  # (?,512,512,16)
conv8_ = tf.nn.conv2d(rconv8, w8_, strides=[1, 1, 1, 1], padding='SAME')
rconv8_ = tf.nn.relu(conv8_ + b8_)  # (?,512,512,16)

#
#w = tf.get_variable('w', shape=[1, 1, 16, 1], initializer=tf.contrib.layers.xavier_initializer())
#b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer(0.0))
w = tf.get_variable('w', shape=[1, 1, 16, 3], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b', shape=[3], initializer=tf.constant_initializer(0.0))
conv8 = tf.nn.conv2d(rconv8_, w, strides=[1, 1, 1, 1], padding='SAME') + b

logits = conv8
##
Y_tensor_int32 = tf.cast(Y_tensor, dtype=tf.int32)
Y_squeeze = tf.squeeze(Y_tensor_int32 ,axis=3)
#loss = - dice_coef(Y, logits)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_squeeze, logits=logits)
loss_sum = tf.reduce_sum(loss )
#
logits_softmax = tf.nn.softmax(logits, axis=3)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss_sum)

##
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

accuracy = dice_coef(Y_tensor, logits)


batch_size = 10
train_X_batch = train_X[60:140, :, :, :]
train_Y_batch = train_Y[60:140, :, :, :]


# initialize
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess= tf.Session(config=config)
##
sess.run(tf.global_variables_initializer())
training_epochs = 1000
avg_cost = 0
Epoch_total = 0
c_list = []
##
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):

    # total_batch = int(mnist.train.num_examples / batch_size)
    Epoch_total = Epoch_total + 1
    total_batch = training_epoches
    for i in range(total_batch):
        batch_xs = train_X_batch
        batch_ys = train_Y_batch
        feed_dict = {X_tensor: batch_xs, Y_tensor: batch_ys}
        c, _ = sess.run([loss_sum, train], feed_dict=feed_dict)
        c_list.append(c)
        # print(c)
        # avg_cost += c / total_batch

    print('Epoch:', '%04d' % (Epoch_total + 1), 'cost =', '{:.9f}'.format(c))

print('Learning Finished!')

##
out_img = sess.run(logits_softmax, feed_dict={X_tensor: batch_xs, Y_tensor: batch_ys})
out_img_sel = np.squeeze(out_img[0])
out_img_sel_bg = out_img_sel[:,:,0]
out_img_sel_brain = out_img_sel[:,:,1]
out_img_sel_lung = out_img_sel[:,:,2]
fig = plt.figure()
ax1 = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6 = fig.add_subplot(336)
ax7 = fig.add_subplot(337)
ax8 = fig.add_subplot(338)
ax9 = fig.add_subplot(339)
ax1.set_title('Region_Brain')
ax1.imshow(out_img_sel_bg)
ax4.imshow(out_img_sel_brain)
ax7.imshow(out_img_sel_lung)
ax2.set_title('Region: No OAR')
out_img_sel = np.squeeze(out_img[40])
out_img_sel_bg = out_img_sel[:,:,0]
out_img_sel_brain = out_img_sel[:,:,1]
out_img_sel_lung = out_img_sel[:,:,2]
ax2.imshow(out_img_sel_bg)
ax5.imshow(out_img_sel_brain)
ax8.imshow(out_img_sel_lung)
ax3.set_title('Region: Lung')
out_img_sel = np.squeeze(out_img[79])
out_img_sel_bg = out_img_sel[:,:,0]
out_img_sel_brain = out_img_sel[:,:,1]
out_img_sel_lung = out_img_sel[:,:,2]
ax3.imshow(out_img_sel_bg)
ax6.imshow(out_img_sel_brain)
ax9.imshow(out_img_sel_lung)

##
np.shape(batch_ys)
target_slice_number= 79
target_slice = np.squeeze(batch_ys[target_slice_number, :, :])
plt.figure()
plt.imshow(target_slice )
