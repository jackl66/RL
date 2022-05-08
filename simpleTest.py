# Make sure to have the server side running in CoppeliaSim: 
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!
import time
import math

try:
    import sim
    import numpy as np
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torch as T
    import cv2 as cv
    import scipy.misc
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time

print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections

clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
# returnCode = sim.simxStartSimulation(clientID,sim.simx_opmode_oneshot )
if clientID != -1:
    print('Connected to remote API server')
    out = nn.AdaptiveAvgPool2d(16)
    res, box = sim.simxGetObjectHandle(clientID, 'box', sim.simx_opmode_blocking)  # senor under the box

    res, camDepth = sim.simxGetObjectHandle(clientID, 'depth', sim.simx_opmode_blocking)
    print(camDepth)
    res, rgb = sim.simxGetObjectHandle(clientID, 'rgb', sim.simx_opmode_blocking)

    returnCode1, resolution1, depthImage1 = sim.simxGetVisionSensorDepthBuffer(clientID, camDepth,
                                                                            sim.simx_opmode_streaming)

    returnCode, resolution, depthImage = sim.simxGetVisionSensorDepthBuffer(clientID, camDepth,
                                                                            sim.simx_opmode_buffer)

    print(returnCode)
    depthImage1 = (np.array(depthImage).reshape((128, 128)))
    # depthImage1 = np.flip(np.array(depthImage).reshape((128, 128)))

    # print(depthImage.shape)
    depthImage = np.array(depthImage).reshape((1, resolution[0], resolution[1]))
    print(type(depthImage), 78)
    de = T.tensor(depthImage)
    t = out(de)
    print(t.size())
    depth_filtered = np.squeeze(t)

    fig = plt.figure()
    fig.set_size_inches(9.5, 6.5)

    plt.subplot(2, 2, 1)
    plt.imshow(depth_filtered)

    plt.subplot(2, 2, 2)
    plt.imshow(depthImage1)

    # --------------------------------------#

    depth_filtered3 = depthImage1[10:100, 10:100]
    plt.subplot(2, 2, 3)
    plt.imshow(depth_filtered3)

    depthImage2 = np.array(depth_filtered3).reshape((1, 90, 90))
    print(type(depthImage), 78)
    de = T.tensor(depthImage2)
    t = out(de)
    print(t.size())

    depth_filtered2 = np.squeeze(t)
    plt.subplot(2, 2, 4)

    plt.imshow(depth_filtered2)
    plt.show()
else:
    print('Failed connecting to remote API server')
print('Program ended')

# print(depthImage.shape)
# out = nn.AdaptiveAvgPool2d((9, 9))
# depth_filtered = np.squeeze(out(depthImage))
# print(depth_filtered.shape)
sim.simxFinish(clientID)
