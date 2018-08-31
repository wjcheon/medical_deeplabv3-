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
train_Y = RT_Structure_Lung_expand_dim
# sample_train_Y = train_Y[130,:,:, 1]
# plt.figure()
# plt.imshow(sample_train_Y)
print('CT images and RT-Structure were successfully loaded!')
print("The shape of trainX {}".format(np.shape(train_X)))
print("The shape of trainY {}".format(np.shape(train_Y)))

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
from resnet_module import softmax_layer, conv_layer, residual_block

is_train = True
learning_rate = 0.001
training_epoches = 15
batch_size = 10
n = 20

if n < 20 or (n - 20) % 12 != 0:
    print ("ResNet depth invalid.")


num_conv = int((n - 20) / 12 + 1)



train_X_batch = train_X[130:135, :, :, :]
train_Y_batch = train_Y[130:135, :, :, :]

X_tensor = tf.placeholder(tf.float32, [None, 256, 256, 1])
#Y_tensor = tf.placeholder(tf.float32, [None, 256, 256, 2])
Y_tensor = tf.placeholder(tf.float32, [None, 256, 256, 1])
##

with tf.variable_scope('conv1'):
    conv1 = conv_layer(X_tensor, [7, 7, 1, 64], 2, af_='sigmoid')


for i in range(num_conv):
    with tf.variable_scope('conv2_%d' % (i + 1)):
        conv2_first = residual_block(conv1, 64, down_sample_=True)
        conv2 = residual_block(conv2_first, 64, False)

for i in range(num_conv):
    down_sample = True if i == 0 else False
    print(down_sample)
    with tf.variable_scope('conv3_%d' % (i + 1)):
        conv3_x = residual_block(conv2, 128, down_sample)
        conv3 = residual_block(conv3_x, 128, False)


for i in range(num_conv):
    down_sample = True if i == 0 else False
    with tf.variable_scope('conv4_%d' % (i + 1)):
        conv4_x = residual_block(conv3, 256, down_sample)
        conv4 = residual_block(conv4_x, 256, False)



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
    conv5 =  atrous_spatial_pyramid_pooling(conv4)


with tf.variable_scope('Decoder_step_1'):
    # Decoder_bilinear_up_output_stride_4
    conv5_bilinear_upsampling = tf.image.resize_bilinear(conv5, [64, 64])
    # Dimension reduction of Conv2 -> 48 or 32
    conv2_dim_reduction = slim.conv2d(conv2, 32, [1, 1],padding='SAME', activation_fn=None)
    # Concat
    conv_6_concat = tf.concat((conv5_bilinear_upsampling, conv2_dim_reduction), axis=3)
    # Conv (3x3, 256) x 2
    conv_6_concat_conv1 = slim.conv2d(conv_6_concat , 256, [3, 3])
    conv_6= slim.conv2d(conv_6_concat_conv1 , 256, [3, 3])


with tf.variable_scope('Decoder_step_2'):
    #conv7_dim_reduction = slim.conv2d(conv_6, 2, [1, 1], padding='SAME')
    conv7_dim_reduction = slim.conv2d(conv_6, 1, [1, 1], padding='SAME')

    conv7 = tf.image.resize_bilinear(conv7_dim_reduction, [256, 256])

logits = conv7

##
def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#
# n_class= 2
# cost = cross_entropy(tf.reshape(Y_tensor, [-1, n_class]), tf.reshape(pixel_wise_softmax(logits), [-1, n_class]))

def _get_cost(n_class_, gt_, logits_, cost_name, cost_kwargs):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are:
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """

    with tf.name_scope("cost"):
        flat_logits = tf.reshape(logits_, [-1, n_class_])
        flat_logits = tf.reshape(logits, [-1])
        flat_labels = tf.reshape(gt_, [-1, n_class_])
        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                      labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                 labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax(logits_)
            intersection = tf.reduce_sum(prediction * gt_)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(gt_)
            loss = -(2 * intersection / (union))

        else:
            raise ValueError("Unknown cost function: " % cost_name)


        return loss


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

cost_dice = -dice_coef(Y_tensor, logits)
cost_kwargs={}
n_class = 2
#cost = _get_cost(n_class, Y_tensor, logits, "cross_entropy", cost_kwargs)
#cost = _get_cost(n_class, Y_tensor, logits, "dice_coefficient", cost_kwargs)
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_tensor, logits=logits))
#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_dice)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay= 0.999).minimize(cost_dice)


##
# initialize
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess= tf.Session(config=config)
##
sess.run(tf.global_variables_initializer())
training_epochs = 10000
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
        c, _ = sess.run([cost_dice, optimizer], feed_dict=feed_dict)
        c_list.append(c)
        # print(c)
        # avg_cost += c / total_batch

    print('Epoch:', '%04d' % (Epoch_total + 1), 'cost =', '{:.9f}'.format(c))

print('Learning Finished!')

##

logits_val = sess.run(logits, feed_dict=feed_dict)
plt.figure()
plt.imshow(logits_val[10,:,:,1])

##
index_temp = 3
img_tmp = np.squeeze(train_Y_batch[index_temp])
img_tmp2 = np.squeeze(train_X_batch[index_temp])
plt.figure()
plt.imshow(img_tmp)
plt.figure()
plt.imshow(img_tmp2)