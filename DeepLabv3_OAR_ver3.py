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
from resnet_module import conv_layer, residual_block, trans_conv_layer, atrous_spatial_pyramid_pooling

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


with tf.variable_scope('ASPP'):
    conv4_ASSP =  atrous_spatial_pyramid_pooling(conv4)


# with tf.variable_scope("deconv6") as scope:
deconv5 = trans_conv_layer(conv4_ASSP, [32, 32, 128], name='deconv5')
deconv5_concat = tf.concat([conv3, deconv5], 3)  # (?, 64, 64, 256)
deconv5_conv1 = conv_layer(deconv5_concat, [3, 3, 192, 128], 1, name='decon5_concat_conv1')
deconv5_conv2 = conv_layer(deconv5_conv1 , [3, 3, 128, 128], 1, name='decon5_concat_conv2')

# with tf.variable_scope("deconv6") as scope:
deconv6 = trans_conv_layer(deconv5_conv2, [64, 64, 64], name='deconv6')
deconv6_concat = tf.concat([conv2, deconv6], 3)  # (?, 64, 64, 256)
deconv6_conv1 = conv_layer(deconv6_concat, [3, 3, 80, 64], 1, name='decon6_concat_conv1')
deconv6_conv2 = conv_layer(deconv6_conv1 , [3, 3, 64, 64], 1, name='decon6_concat_conv2')

# with tf.variable_scope("deconv7") as scope:
deconv7 = trans_conv_layer(deconv6_conv2, [128, 128, 32], name='deconv7')
deconv7_concat = tf.concat([conv1, deconv7], 3)  # (?, 64, 64, 256)
deconv7_conv1 = conv_layer(deconv7_concat, [3, 3, 48, 32], 1, name='deconv7_concat_conv1')
deconv7_conv2 = conv_layer(deconv7_conv1 , [3, 3, 32, 32], 1, name='deconv7_concat_conv2')


# with tf.variable_scope("deconv8") as scope:
deconv8 = trans_conv_layer(deconv7_conv2, [256, 256, 16], name='deconv8')
deconv8_conv1 = conv_layer(deconv8 , [3, 3, 16, 3], 1, af_='None', name='deconv8_concat_conv1')
# deconv8_conv2 = conv_layer(deconv8_conv1 , [3, 3, 8, 4], 1, name='decon8_concat_conv2')
# deconv8_conv3 = conv_layer(deconv8_conv2, [1, 1, 4, 3], 1, af_ = 'None', name='logits')

logits = deconv8_conv1

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
training_epochs = 500
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

##
plt.figure()
plt.plot(c_list)