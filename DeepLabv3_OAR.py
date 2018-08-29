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
root = tk.Tk()
root.withdraw()
file_path_CT = filedialog.askopenfilename(filetypes=[("Select CT images","*.mat")])
file_path_brain = filedialog.askopenfilename(filetypes=[("Select BRAIN OAR","*.mat")])
file_path_rt_lung = filedialog.askopenfilename(filetypes=[("Select RT-LUNG OAR","*.mat")])
file_path_lt_lung = filedialog.askopenfilename(filetypes=[("Select LT-LUNG OAR","*.mat")])

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
RT_Structure_Lung_expand_dim = np.expand_dims(RT_Structure_Lung, axis=3)

#
train_X = CT_img_standardization_expand_dim_float32
train_Y = []
tarin_Y_slice =[]
for x in zip(RT_Structure_Brain_expand_dim, RT_Structure_Lung_expand_dim):
    print(np.shape(x[0]))
    print(np.shape(x[1]))
    tarin_Y_slice = np.squeeze(np.stack((x[0],x[1]), axis=2),axis=3)
    train_Y.append(tarin_Y_slice)

#
train_Y = np.asanyarray(train_Y)
# plt.figure()
# plt.imshow(train_Y[30,:,:,1])
print('CT images and RT-Structure were successfully loaded!')
print("The shape of trainX {}".format(np.shape(train_X)))
print("The shape of trainY {}".format(np.shape(train_Y)))

## Image debug using my-eyes
size_RT_Structure = np.shape(RT_Structure_Lung )
Check_target = CT_img_standardization_expand_dim_float32
Check_target = np.squeeze(Check_target )
Check_target = RT_Structure_Lung
# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()
slice_mean = []
for num_slice_iter in range(0,size_RT_Structure[0]):
    target_slice = Check_target[num_slice_iter];
    plt.imshow(target_slice )
    slice_mean.append(np.mean(target_slice ))
    plt.draw()
    plt.pause(0.0000001)
    plt.clf
    print(num_slice_iter)
##
##
import tensorflow as tf
import tensorflow.contrib.slim as slim
from resnet_module import softmax_layer, conv_layer, residual_block

learning_rate = 0.001
training_epoches = 15
batch_size = 10
n = 20

if n < 20 or (n - 20) % 12 != 0:
    print ("ResNet depth invalid.")


num_conv = int((n - 20) / 12 + 1)


layers = []
train_X_batch = train_X[121:161, :, :, :]
train_Y_batch = train_Y[121:161, :, :, :]

X_tensor = tf.placeholder(tf.float32, [None, 256, 256, 1])
Y_tensor = tf.placeholder(tf.float32, [None, 256, 256, 2])

layers = []
with tf.variable_scope('conv1'):
    conv1 = conv_layer(X_tensor, [7, 7, 1, 16], 2)
    layers.append(conv1)

for i in range(num_conv):
    with tf.variable_scope('conv2_%d' % (i + 1)):
        conv2_first = residual_block(layers[-1], 16, down_sample=True)
        conv2 = residual_block(conv2_first, 16, False)
        layers.append(conv2_first)
        layers.append(conv2)

for i in range(num_conv):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv3_%d' % (i + 1)):
        conv3_x = residual_block(layers[-1], 32, down_sample)
        conv3 = residual_block(conv3_x, 32, False)
        layers.append(conv3_x)
        layers.append(conv3)


for i in range(num_conv):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i + 1)):
        conv4_x = residual_block(layers[-1], 64, down_sample)
        conv4 = residual_block(conv4_x, 64, False)
        layers.append(conv4_x)
        layers.append(conv4)



def atrous_spatial_pyramid_pooling(net, depth=256, reuse=None):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """
    feature_map_size = tf.shape(net)
    depth = 12
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
    conv5 =  atrous_spatial_pyramid_pooling(conv4)

conv5_for_concat_1x1 = slim.conv2d(conv5, 1, [1, 1], activation_fn=None )
conv5_for_concat_upsampling = slim.conv2d_transpose(conv5_for_concat_1x1, 1, [3, 3], [4, 4], padding="SAME")

conv_6  = slim.conv2d(conv5, 1, [1, 1], scope="conv_1x1_to_channel_1", activation_fn=None)
conv_7 = slim.conv2d_transpose(conv_6, 1, [3, 3], [4, 4],padding='SAME')
conv_7_concat = tf.concat((conv_7, conv5_for_concat_upsampling), axis=3)

conv_8 = slim.conv2d(conv_7_concat, 1, [3, 3], activation_fn=None)
conv_9_lastlayer = slim.conv2d_transpose(conv_8, 2, [3, 3], [4, 4], padding='SAME')


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_tensor, logits=conv_9_lastlayer))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay= 0.999).minimize(cost)


##
# initialize
sess = tf.Session()
##
sess.run(tf.global_variables_initializer())
training_epochs = 6000
avg_cost = 0
Epoch_total = 0
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
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        # print(c)
        # avg_cost += c / total_batch

    print('Epoch:', '%04d' % (Epoch_total + 1), 'cost =', '{:.9f}'.format(c))

print('Learning Finished!')

##

logits_val = sess.run(conv_9_lastlayer, feed_dict=feed_dict)
plt.figure()
plt.imshow(logits_val[10,:,:,1])
