##
import tensorflow as tf
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
train_Y =  []
tarin_Y_slice =[]
for x in zip(RT_Structure_Brain_expand_dim, RT_Structure_BG_Brain_Lung_expand_dim):
    print(np.shape(x[0]))
    print(np.shape(x[1]))
    tarin_Y_slice = np.squeeze(np.stack((x[0],x[1]), axis=2),axis=3)
    train_Y.append(tarin_Y_slice)

#
#train_Y = np.asanyarray(train_Y)
train_Y = RT_Structure_BG_Brain_Lung_expand_dim

# sample_train_Y = train_Y[130,:,:, 1]
# plt.figure()
# plt.imshow(sample_train_Y)
print('CT images and RT-Structure were successfully loaded!')
print("The shape of trainX {}".format(np.shape(train_X)))
print("The shape of trainY {}".format(np.shape(train_Y)))
print("tarinY consist of {}".format(np.unique(train_Y)))
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

# %% function that it'll be used
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# %% build network
X = tf.placeholder(tf.float32, [None, 256, 256, 1])
Y = tf.placeholder(tf.float32, [None, 256, 256, 1])
keep_prob = tf.placeholder(tf.float32)

# with tf.variable_scope("conv0") as scope:
w0 = tf.get_variable('w0', shape=[3, 3, 1, 16], initializer=tf.contrib.layers.xavier_initializer())
w0_ = tf.get_variable('w0_', shape=[3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
b0 = tf.get_variable('b0', shape=[16], initializer=tf.constant_initializer(0.0))
b0_ = tf.get_variable('b0_', shape=[16], initializer=tf.constant_initializer(0.0))
conv0 = tf.nn.conv2d(X, w0, strides=[1, 1, 1, 1], padding='SAME')  # (?,512,512,16)
rconv0 = tf.nn.relu(conv0 + b0)
conv0_ = tf.nn.conv2d(rconv0, w0_, strides=[1, 1, 1, 1], padding='SAME')  # (?,512,512,16)
rconv0_ = tf.nn.relu(conv0_ + b0_)
pool0 = tf.nn.max_pool(rconv0_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,254,254,16)
# print(pool0.shape)

# with tf.variable_scope("conv1") as scope:
w1 = tf.get_variable('w1', shape=[3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
w1_ = tf.get_variable('w1_', shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('b1', shape=[32], initializer=tf.constant_initializer(0.0))
b1_ = tf.get_variable('b1_', shape=[32], initializer=tf.constant_initializer(0.0))
conv1 = tf.nn.conv2d(pool0, w1, strides=[1, 1, 1, 1], padding='SAME')  # (?,256,256,32)
rconv1 = tf.nn.relu(conv1 + b1)
conv1_ = tf.nn.conv2d(rconv1, w1_, strides=[1, 1, 1, 1], padding='SAME')  # (?,256,256,32)
rconv1_ = tf.nn.relu(conv1_ + b1_)
pool1 = tf.nn.max_pool(rconv1_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,128,128,32)
# print(pool1.shape)

# with tf.variable_scope("conv2") as scope:
w2 = tf.get_variable('w2', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
w2_ = tf.get_variable('w2_', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2', shape=[64], initializer=tf.constant_initializer(0.0))
b2_ = tf.get_variable('b2_', shape=[64], initializer=tf.constant_initializer(0.0))
conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME')  # (?,128,128,64)
rconv2 = tf.nn.relu(conv2 + b2)
conv2_ = tf.nn.conv2d(rconv2, w2_, strides=[1, 1, 1, 1], padding='SAME')  # (?,128,128,64)
rconv2_ = tf.nn.relu(conv2_ + b2_)
pool2 = tf.nn.max_pool(rconv2_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,64,64,64)
# print(pool2.shape)

# with tf.variable_scope("conv3") as scope:
w3 = tf.get_variable('w3', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
w3_ = tf.get_variable('w3_', shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('b3', shape=[128], initializer=tf.constant_initializer(0.0))
b3_ = tf.get_variable('b3_', shape=[128], initializer=tf.constant_initializer(0.0))
conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='SAME')  # (?,64,64,128)
rconv3 = tf.nn.relu(conv3 + b3)
conv3_ = tf.nn.conv2d(rconv3, w3_, strides=[1, 1, 1, 1], padding='SAME')  # (?,64,64,128)
rconv3_ = tf.nn.relu(conv3_ + b3_)
pool3 = tf.nn.max_pool(rconv3_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (?,32,32,128)
# print(pool3.shape)

# with tf.variable_scope("conv4") as scope:
w4 = tf.get_variable('w4', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
w4_ = tf.get_variable('w4_', shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable('b4', shape=[256], initializer=tf.constant_initializer(0.0))
b4_ = tf.get_variable('b4_', shape=[256], initializer=tf.constant_initializer(0.0))
conv4 = tf.nn.conv2d(pool3, w4, strides=[1, 1, 1, 1], padding='SAME')  # (?,32,32,256)
rconv4 = tf.nn.relu(conv4 + b4)
conv4_ = tf.nn.conv2d(rconv4, w4_, strides=[1, 1, 1, 1], padding='SAME')  # (?,32,32,256)
rconv4_ = tf.nn.relu(conv4_ + b4_)
# print(rconv4_.shape)

# %%
# with tf.variable_scope("deconv5") as scope:
wd5 = tf.get_variable('wd5', shape=[2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
bd5 = tf.get_variable('bd5', shape=[128], initializer=tf.contrib.layers.xavier_initializer())
tmp5 = tf.nn.conv2d_transpose(rconv4_, wd5, [tf.shape(X)[0], 32, 32, 128], strides=[1, 2, 2, 1], padding='SAME')
deconv5 = tf.nn.relu(tmp5 + bd5)  # (?, 64, 64, 128)
deconv5_concat = tf.concat([rconv3_, deconv5], 3)  # (?, 64, 64, 256)
w5 = tf.get_variable('w5', shape=[3, 3, 256, 128], initializer=tf.contrib.layers.xavier_initializer())
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
tmp6 = tf.nn.conv2d_transpose(rconv5_, wd6, [tf.shape(X)[0], 64, 64, 64], strides=[1, 2, 2, 1], padding='SAME')
deconv6 = tf.nn.relu(tmp6 + bd6)  # (?,128,128,64)
deconv6_concat = tf.concat([rconv2_, deconv6], 3)  # (?,128,128,128)
w6 = tf.get_variable('w6', shape=[3, 3, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
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
tmp7 = tf.nn.conv2d_transpose(rconv6_, wd7, [tf.shape(X)[0], 128, 128, 32], strides=[1, 2, 2, 1], padding='SAME')
deconv7 = tf.nn.relu(tmp7 + bd7)  # (?,256,256,32)
deconv7_concat = tf.concat([rconv1_, deconv7], 3)  # (?,256,256,64)
w7 = tf.get_variable('w7', shape=[3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer())
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
tmp8 = tf.nn.conv2d_transpose(rconv7_, wd8, [tf.shape(X)[0], 256, 256, 16], strides=[1, 2, 2, 1], padding='SAME')
deconv8 = tf.nn.relu(tmp8 + bd8)  # (?,512,512,16)
deconv8_concat = tf.concat([rconv0_, deconv8], 3)  # (?,512,512,32)
w8 = tf.get_variable('w8', shape=[3, 3, 32, 16], initializer=tf.contrib.layers.xavier_initializer())
w8_ = tf.get_variable('w8_', shape=[3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.get_variable('b8', shape=[16], initializer=tf.constant_initializer(0.0))
b8_ = tf.get_variable('b8_', shape=[16], initializer=tf.constant_initializer(0.0))
conv8 = tf.nn.conv2d(deconv8_concat, w8, strides=[1, 1, 1, 1], padding='SAME')
rconv8 = tf.nn.relu(conv8 + b8)  # (?,512,512,16)
conv8_ = tf.nn.conv2d(rconv8, w8_, strides=[1, 1, 1, 1], padding='SAME')
rconv8_ = tf.nn.relu(conv8_ + b8_)  # (?,512,512,16)

#
#w = tf.get_variable('w', shape=[1, 1, 16, 1], initializer=tf.contrib.layers.xavier_initializer())
#b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer(0.0))
w = tf.get_variable('w', shape=[1, 1, 16, 3], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b', shape=[3], initializer=tf.constant_initializer(0.0))
conv8 = tf.nn.conv2d(rconv8_, w, strides=[1, 1, 1, 1], padding='SAME') + b
##

# rconv8 = tf.nn.relu(conv8 + b8)
#logits = tf.nn.sigmoid(conv8)
logits = conv8
# %% cost
Y = tf.cast(Y, dtype=tf.int32)
Y_squeeze = tf.squeeze(Y,axis=3)
#loss = - dice_coef(Y, logits)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_squeeze, logits=logits)
loss_2 = tf.reduce_sum(loss )
logits_softmax = tf.nn.softmax(logits, axis=3)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss_2)

# accuracy
accuracy = dice_coef(Y, logits)

## %% train
batch_size = 10
train_X_batch = train_X[60:140, :, :, :]
train_Y_batch = train_Y[60:140, :, :, :]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
##
# f = open(full_filename, 'w')
# aa = range(1,11)
# bb = range(11,21)
# for a_val, b_val in zip(aa,bb):
#    line_buffer = '{0:.6f}, {1:.6f}\n'.format(a_val,b_val)
#    f.write(line_buffer)
# f.close()
full_train_filename = './train_log.txt'
full_test_filename = './test_log.txt'
#f_training = open(full_train_filename, 'wb+')
#f_test = open(full_test_filename, 'wb+')
total_epoch = 0
for epoch in range(1000):

    # batch_x = x_data[step * batch_size:step * batch_size + batch_size]
    # batch_x = np.expand_dims(batch_x, 3)
    # batch_y = y_data[step * batch_size:step * batch_size + batch_size]
    # batch_y = np.expand_dims(batch_y, 3)
    #
    batch_x = train_X_batch
    batch_y = train_Y_batch
    l, _ = sess.run([loss_2, train], feed_dict={X: batch_x, Y: batch_y})
    print('{:}, training cost = {:.4f}'.format(total_epoch + 1, l))
    total_epoch = total_epoch+1
    #a, out_img = sess.run([accuracy, logits], feed_dict={X: batch_x, Y: batch_y})
    out_img = sess.run(logits, feed_dict={X: batch_x, Y: batch_y})
#    print('accuracy = {:.4f}'.format(a))

    #f_training.write('{0:4d}, {1:.4f}, {2:.4f}\n'.format(epoch, l, a))
    #f_training.flush()
    #        logger.info(l)
    #        logger.info(a)


print('End epoch')
print("Finish all!")
## 0 40 79
out_img = sess.run(logits_softmax, feed_dict={X: batch_x, Y: batch_y})
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


