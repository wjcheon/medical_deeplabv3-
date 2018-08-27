##
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
print('Successfully import the pakages')

##
root = tk.Tk()
root.withdraw()
file_path_CT = filedialog.askopenfilename(filetypes=[("Select CT images","*.mat")])
file_path_brain = filedialog.askopenfilename(filetypes=[("Select OAR","*.mat")])

##
import h5py
with h5py.File(file_path_CT, 'r') as file:
    print(list(file.keys()))
    CT_img = list(file['CT_img'])

with h5py.File(file_path_brain, 'r') as file:
    print(list(file.keys()))
    RT_Structure = list(file['target_GT_mask'])
print('CT images and RT-Structure were successfully loaded!')
##
CT_img = np.asanyarray(CT_img)
RT_Structure = np.asanyarray(RT_Structure)
size_RT_Structure = np.shape(RT_Structure)
##
Check_target = CT_img
# You probably won't need this if you're embedding things in a tkinter plot...
plt.ion()

for num_slice_iter in range(0,size_RT_Structure[0]):
    plt.imshow(Check_target[num_slice_iter,:,:])
    plt.draw()
    plt.pause(0.0000001)
    plt.clf
    print(num_slice_iter)
##

