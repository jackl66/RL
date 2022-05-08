# This small example illustrates how to use the remote API
# synchronous mode. The synchronous mode needs to be
# pre-enabled on the server side. You would do this by
# starting the server (e.g. in a child script) with:
#
# simRemoteApi.start(19999,1300,false,true)
#
# But in this example we try to connect on port
# 19997 where there should be a continuous remote API
# server service already running and pre-enabled for
# synchronous mode.
#
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

try:
    import sim
    import numpy as np
    import math
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import sys

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # enable the synchronous mode on the client:
    sim.simxSynchronous(clientID,True)

    # start the simulation:
    sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)
    res, j6 = sim.simxGetObjectHandle(clientID, 'UR5_joint6', sim.simx_opmode_blocking)
    ret, position = sim.simxGetJointPosition(clientID, j6, sim.simx_opmode_streaming)
    ret, position6 = sim.simxGetJointPosition(clientID, j6, sim.simx_opmode_buffer)
    emptyBuff = bytearray()
    sim.simxSynchronousTrigger(clientID)

    errorCode = sim.simxSetJointTargetPosition (clientID, j6, position6+(1*math.pi/180),
                                               sim.simx_opmode_oneshot)

    # Now step a few times:
    for i in range(1,20):
        if sys.version_info[0] == 3:
            input('Press <enter> key to step the simulation!')
        else:
            raw_input('Press <enter> key to step the simulation!')
        time1 = sim.simxGetLastCmdTime(clientID)

        # ret, position6 = sim.simxGetJointPosition(clientID, j6, sim.simx_opmode_buffer)(
        # errorCode = sim.simxSetJointVelocity(clientID, j6, -0.5,
        #                                            sim.simx_opmode_oneshot)

        # position6-=0.5*math.pi/180
        # res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(clientID, 'UR5',
        #                                                                             sim.sim_scripttype_childscript,
        #                                                                             'mymove',
        #                                                                             [],
        #                                                                             [0, 0, 0, 0, 0, position6],
        #                                                                             [], emptyBuff,
        #                                                                             sim.simx_opmode_blocking)
        #
        # if res == sim.simx_return_ok:
        #     print("results:", retStrings)
        # else:
        #     print("remote function call failed")
        if abs(position6 - (0.5 * math.pi / 180)) > 0.001:
            print("not there yet")
        sim.simxSynchronousTrigger(clientID)
        # todo -0.2, dt=50,10ms in coppelia dp=0.02 *math.pi/180
        # todo -0.2, dt=10,2ms in coppelia  dp=0.004
        # todo -0.2, dt=25,5ms in coppelia  dp=0.01
        ret, position6 = sim.simxGetJointPosition(clientID, j6, sim.simx_opmode_buffer)


        time2 = sim.simxGetLastCmdTime(clientID)
        sim.simxSynchronousTrigger(clientID)
        print('%.10f' %(position6))
        print(time2-time1,99)
        print(time2)
        print(i)
    # stop the simulation:
    sim.simxStopSimulation(clientID,sim.simx_opmode_blocking)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')




