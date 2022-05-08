import sim
import time
import argparse as ap
import numpy as np
import csv
import ast

import matplotlib.pyplot as plt
from scipy.stats import norm
import math


class coppelia:
    def __init__(self, port=19997, num_obj=5, mass=14.375e-03, size=0.025):
        super(coppelia, self).__init__()
        self.port = port
        self.mass = mass
        self.init_error = 0
        self.emptyBuff = bytearray()
        self.object_handles = []
        self.obj_string = ['Cuboid', 'Cylinder', 'Sphere']
        self.num_object_pool = [30, 35, 40, 45, 50]
        self.obj_size_pool = [0.025, 0.019, 0.016]
        self.radius = 0.123198
        self.offset = -0.611446
        self.single_block_weight = mass * 9.81

    def reset(self, height=0.0, velocity=0.0, offset=0.0, num_obj=1, size=1):

        self.object = self.obj_string[0]
        self.num_object = self.num_object_pool[num_obj]
        # set to one when collect the data
        # self.num_object = 1
        self.size = self.obj_size_pool[size]
        self.total_block_weight = self.num_object * self.mass * 9.81

        self.clientID = sim.simxStart('127.0.0.1', self.port, True, True, 5000, 5)
        if self.clientID != -1:
            print('connected to remote API server')
        else:
            print('fail')
            exit(0)
        returnCode = sim.simxSynchronous(self.clientID, True)
        print(returnCode, "synch")
        returnCode = sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        if returnCode != 0 and returnCode != 1:
            print('fail to start')
            exit(0)

        self.triggerSim()
        # obtain all the handles and signal
        ret, signalValue = sim.simxGetFloatSignal(self.clientID, 'init_done', sim.simx_opmode_streaming)
        ret, error = sim.simxGetFloatSignal(self.clientID, 'exception', sim.simx_opmode_streaming)
        # only for rim to rim case
        ret0, signalValue = sim.simxGetFloatSignal(self.clientID, 'first_arm_done', sim.simx_opmode_streaming)

        # sensor under the container
        res, self.target = sim.simxGetObjectHandle(self.clientID, '/Floor/f2/Box/f1', sim.simx_opmode_blocking)
        returnCode, state, forceVector, toequeVector = sim.simxReadForceSensor(self.clientID, self.target,
                                                                               sim.simx_opmode_streaming)

        # get handles
        res, self.joint6 = sim.simxGetObjectHandle(self.clientID, 'UR5_joint6', sim.simx_opmode_blocking)
        res, self.cup = sim.simxGetObjectHandle(self.clientID, 'source', sim.simx_opmode_blocking)
        res, self.plane = sim.simxGetObjectHandle(self.clientID, 'Plane', sim.simx_opmode_blocking)
        res, self.rim = sim.simxGetObjectHandle(self.clientID, 'rim', sim.simx_opmode_blocking)

        # position steaming init
        ret, position = sim.simxGetJointPosition(self.clientID, self.joint6, sim.simx_opmode_streaming)
        ret, position = sim.simxGetObjectPosition(self.clientID, self.cup, -1, sim.simx_opmode_streaming)
        ret, position = sim.simxGetObjectPosition(self.clientID, self.plane, -1, sim.simx_opmode_streaming)
        ret, position = sim.simxGetObjectPosition(self.clientID, self.rim, -1, sim.simx_opmode_streaming)

        ret0, signalValue1 = sim.simxGetFloatSignal(self.clientID, 'first_arm_done', sim.simx_opmode_buffer)
        while signalValue1 != 99:
            self.triggerSim()
            ret0, signalValue1 = sim.simxGetFloatSignal(self.clientID, 'first_arm_done', sim.simx_opmode_buffer)
        self.py_get_pose()

        self.py_moveToPose(offset, height)
        loop = 20
        while loop > 0:
            self.triggerSim()
            loop -= 1

        sim.simxSetIntegerSignal(self.clientID, 'arm_done', 99, sim.simx_opmode_oneshot)
        # ret, position = sim.simxGetObjectPosition(self.clientID, self.plane, self.cup, sim.simx_opmode_streaming)
        self.setNumberOfBlock()

        # wait for objects to be created
        while True:
            self.triggerSim()
            # signal to mark the creation is finished
            ret0, signalValue = sim.simxGetFloatSignal(self.clientID, 'init_done', sim.simx_opmode_buffer)
            ret, error = sim.simxGetFloatSignal(self.clientID, 'exception', sim.simx_opmode_blocking)

            if signalValue == 99:
                loop = 20
                # small stall to make sure everything falls in the container
                while loop > 0:
                    self.triggerSim()
                    loop -= 1
                break
            if error == 99:
                self.init_error += 1

        # print("init done")
        # object handles array
        idx = 0
        while idx < self.num_object:
            self.object_handles.append(
                sim.simxGetObjectHandle(self.clientID, self.object + str(idx), sim.simx_opmode_blocking))
            sim.simxGetObjectPosition(self.clientID, self.object_handles[idx][1], -1, sim.simx_opmode_streaming)
            idx += 1

        # all positions are relative to the world
        ret, self.plane_position = sim.simxGetObjectPosition(self.clientID, self.plane, -1,
                                                             sim.simx_opmode_buffer)
        ret, self.cup_position = sim.simxGetObjectPosition(self.clientID, self.cup, -1,
                                                           sim.simx_opmode_buffer)
        ret, self.rim_position = sim.simxGetObjectPosition(self.clientID, self.rim, -1,
                                                           sim.simx_opmode_buffer)
        self.height_diff = abs(self.plane_position[2] - self.cup_position[2])
        # print(f'cup original height {self.cup_position[2]}. original relative height diff {self.height_diff}')
        # print(f'rim position {self.rim_position}')

        sim.simxSetJointTargetVelocity(self.clientID, self.joint6, velocity, sim.simx_opmode_oneshot_wait)
        self.triggerSim()

        ret, pos = sim.simxGetObjectPosition(self.clientID, self.object_handles[0][1], -1,
                                             sim.simx_opmode_buffer)
        # print(pos[2], ' height')

        # self.target_box_weight = -1 * np.mean(filtered_reading)
        self.target_box_weight = 1.963962

        # print(self.target_box_weight, 'box weight')

    # call Lua script to spawn the objects the in scene
    def setNumberOfBlock(self):
        emptyBuff = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, 'target',
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'setNumberOfBlocks',
                                                                                    [self.num_object],
                                                                                    [self.mass, self.size],
                                                                                    [self.object], emptyBuff,
                                                                                    sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print("remote call results:", retStrings[0])
        else:
            print("init function call failed", res)
            self.finish()
            exit(0)

    # get the pose of the end-effector
    def py_get_pose(self):
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, 'UR5',
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'get_pose',
                                                                                    [],
                                                                                    [],
                                                                                    [], self.emptyBuff,
                                                                                    sim.simx_opmode_blocking)
        self.old_z = retFloats[1]
        self.old_y = retFloats[0]

        if res != sim.simx_return_ok:
            print("get pose is wrong", res)
            self.finish()
            exit(0)

    # call Lua script to control the end-effector
    def py_moveToPose(self, displacement_y, displacement_z):

        new_y = self.old_y + displacement_y

        new_z = self.old_z + displacement_z
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, 'UR5',
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'py_moveToPose',
                                                                                    [],
                                                                                    [new_y, new_z],
                                                                                    [], self.emptyBuff,
                                                                                    sim.simx_opmode_blocking)

        if res != sim.simx_return_ok:
            print("moveTopose function call failed", res)
            self.finish()
            exit(0)

    # manually control the scene to go to next time step
    def triggerSim(self):
        e = sim.simxSynchronousTrigger(self.clientID)

    def finish(self):
        x = sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)
        self.object_handles = []


def collect():
    parser = ap.ArgumentParser()
    parser.add_argument("port", help="choose a port",
                        choices={'19990', '19991', '19992', '19993', '19994', '19995', '19996', '19997', '19998',
                                 '19999'})
    # parser.add_argument("num", help="number of objects",
    #                     choices={'1', '5', '10', '15', '25', '20', '30', '40', '50', '35'})

    args = parser.parse_args()
    coppelia_port = int(args.port)
    # num_obj = int(args.num)
    num_obj = 1
    env = coppelia(port=coppelia_port, num_obj=1)

    velocity_pool = [-0.75, - 0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3]
    height_change = np.arange(start=-0.24, stop=0.24, step=0.02)

    trajectory = [[[] for _ in range(num_obj)] for _ in range(240)]

    i = 0
    for velocity in velocity_pool:
        for height in height_change:
            done = False
            env.reset(height, velocity, 0, 1, 0)
            # print(height,velocity)
            above = 0

            while not done:
                ret, position = sim.simxGetJointPosition(env.clientID, env.joint6, sim.simx_opmode_buffer)
                ret, rim_position = sim.simxGetObjectPosition(env.clientID, env.rim, -1, sim.simx_opmode_buffer)

                if position < -2.5:
                    done = True
                # don't need to record the early stage
                if position > -1:
                    env.triggerSim()
                    continue

                for obj in range(len(env.object_handles)):
                    _, obj_position = sim.simxGetObjectPosition(env.clientID, env.object_handles[obj][1], -1,
                                                                sim.simx_opmode_buffer)
                    if abs(obj_position[2] - rim_position[2]) < 0.05:
                        # object is higher than the rim
                        if obj_position[2] - rim_position[2] >= 0.005:
                            above = 1

                        # once the object is lower than the rim,
                        # meaning it falls out of the cup
                        if above:
                            if 0.007 >= rim_position[2] - obj_position[2] >= 0.0:
                                x = obj_position[0] - env.cup_position[0]
                                y = obj_position[1] - env.cup_position[1]
                                z = obj_position[2]

                                trajectory[i][obj].append([0, x, y, z, height, velocity])

                    if abs(obj_position[2] - env.plane_position[2]) < 0.007:
                        x = obj_position[0]
                        y = obj_position[1] - env.cup_position[1]
                        z = obj_position[2]
                        trajectory[i][obj].append([1, x, y, z, height, velocity])

                env.triggerSim()
            env.finish()

            i += 1
        #     if i == 2:
        #         break
        # print(trajectory)
        # break
    file_path = './trajectory/{}/02step_height.npy'.format(num_obj)

    with open(file_path, 'wb') as f:
        np.save(f, np.array(trajectory, dtype=object))


def read():
    data = np.load('./trajectory/1/larger.npy', allow_pickle=True).squeeze()
    print(data.shape)

    fall_result = {}
    land_result = {}
    for trial in data:
        if not trial:
            continue
        falling_height = []
        landing_position = []
        key = str(trial[0][4:])

        fall_result[key] = [0, 0]
        land_result[key] = [0, 0]
        for info in trial:
            if info[0] == 0:
                falling_height.append(info[3])
            else:
                landing_position.append(info[2])

        if len(falling_height) != 0:
            if len(falling_height) != 1:
                falling_mean_coordinate = np.mean(falling_height, axis=0)
                falling_std_coordinate = np.std(falling_height, axis=0)
            else:
                falling_mean_coordinate = falling_height[0]
                falling_std_coordinate = 0
            fall_result[key] = [falling_mean_coordinate, falling_std_coordinate]
        if len(landing_position) != 0:
            if len(landing_position) != 1:
                landing_mean_coordinate = np.mean(landing_position, axis=0)
                landing_std_coordinate = np.std(landing_position, axis=0)
            else:

                landing_mean_coordinate = landing_position[0]
                landing_std_coordinate = 0
            land_result[key] = [landing_mean_coordinate, landing_std_coordinate]

    with open('falling.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['height', 'velocity', 'avg falling height', 'std']
        writer.writerow(header)
        for key, val in fall_result.items():
            output = ast.literal_eval(key)
            h, v, m, s = output[0], output[1], val[0], val[1]
            row = [h, v, m, s]
            writer.writerow(row)

    with open('landing.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['height', 'velocity', 'avg landing position', 'std']
        writer.writerow(header)
        for key, val in land_result.items():
            output = ast.literal_eval(key)
            h, v, m, s = output[0], output[1], val[0], val[1]
            row = [h, v, m, s]
            writer.writerow(row)


def test():
    parser = ap.ArgumentParser()
    parser.add_argument("port", help="choose a port",
                        choices={'19990', '19991', '19992', '19993', '19994', '19995', '19996', '19997', '19998',
                                 '19999'})
    parser.add_argument("part", help="choose a part",
                        choices={'0', '2', '4', '6', '8'})

    args = parser.parse_args()
    coppelia_port = int(args.port)
    part = int(args.part)
    # the target container is 0.3m away in the negative y direction to the cup
    original_y_offset = -0.21261

    # the target container is 0.41m lower than the cup
    original_z_offset = 0.41292
    env = coppelia(port=coppelia_port)
    velocity_pool = [-0.75, - 0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3]
    velocity_pool = velocity_pool[part:part + 2]
    height_change = np.arange(start=-0.24, stop=0.24, step=0.02)

    result = np.zeros((48, 2, 5))
    avg_out = np.zeros((48, 2))

    # sol1: tabular
    # with open("./240trial_0.007reading/falling.csv", newline='') as fall:
    #
    #     falling_reader = csv.reader(fall, delimiter=' ', quotechar='|')
    #     with open("./240trial_0.007reading/landing.csv", newline='') as land:
    #         landing_reader = csv.reader(land, delimiter=' ', quotechar='|')
    #         # skip the header
    #         next(falling_reader)
    #         next(landing_reader)
    #         combination = 0
    #         for falling_row, landing_row in zip(falling_reader, landing_reader):
    #
    #             falling_row = ast.literal_eval(falling_row[0])
    #             landing_row = ast.literal_eval(landing_row[0])
    #
    #             height, velocity, falling_height, landing_offset = falling_row[0], falling_row[1], falling_row[2], \
    #                                                                landing_row[2]
    #             # we want the cube to fall into the target
    #             # therefore if the landing offset is smaller than the original distance between cup the target
    #             # we will move the cup towards target
    #             # if the offset is greater than the distance, we will move the cup away from the target
    #             # both value a negative
    #             if landing_offset < original_y_offset:
    #                 y_displacement = abs(landing_offset - original_y_offset)
    #             else:
    #                 y_displacement = original_y_offset - landing_offset
    # sol2: linear regression
    # y = -0.00919x - 0.1439
    combination = 0
    for velocity in velocity_pool:
        x = 1
        for height in height_change:
            # for each combination, we test with different number of objects, and we repeat this twice
            for rep in range(2):
                diff_num = 0
                for num_obj in range(5):
                    done = False
                    size = np.random.randint(3)
                    y_displacement = -0.00919 * x - 0.1439
                    # we want the cube to fall into the target
                    # therefore if the landing offset is smaller than the original distance between cup the target
                    # we will move the cup towards target
                    # if the offset is greater than the distance, we will move the cup away from the target
                    # both value a negative
                    if y_displacement < original_y_offset:
                        y_displacement = abs(y_displacement - original_y_offset)
                    else:
                        y_displacement = original_y_offset - y_displacement
                    env.reset(height, velocity, y_displacement, num_obj, size)

                    while not done:
                        # determine termination
                        ret, position6 = sim.simxGetJointPosition(env.clientID, env.joint6, sim.simx_opmode_buffer)
                        ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(env.clientID, env.target,
                                                                                         sim.simx_opmode_buffer)
                        # when we poured everything out
                        if position6 < -2.4:
                            done = True
                        env.triggerSim()

                    # make sure everything setting down before we read outliers
                    temp = 0
                    filtered_reading = []
                    while temp < 20:
                        env.triggerSim()
                        temp += 1
                        ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(env.clientID, env.target,
                                                                                         sim.simx_opmode_buffer)
                        filtered_reading.append(forceVector2[2])
                    final_reading = np.mean(filtered_reading)
                    target_weight = -1 * final_reading - env.target_box_weight
                    outlier_weight = env.total_block_weight - target_weight
                    num_outlier = 0
                    print(outlier_weight)
                    # if we have any outlier
                    if outlier_weight > 0:
                        num_outlier = round(outlier_weight / env.single_block_weight)
                        # at least we don't want impossible number
                        if num_outlier > env.num_object:
                            num_outlier = env.num_object
                        result[combination][rep][diff_num] = num_outlier
                    print(f'combination {combination}, rep {rep}, round {diff_num}, outlier{num_outlier}')
                    print('---------------------------------')
                    env.finish()

                    time.sleep(1)
                    diff_num += 1
            combination += 1
            x += 1
    file_name = 'raw_result' + str(part) + '.csv'
    # print(result[:3])
    with open(file_name, 'w') as f:
        i = 0
        writer = csv.writer(f)
        for combination in result:
            rep1 = combination[0]
            rep2 = combination[1]
            temp = np.concatenate((rep1, rep2))
            avg = np.mean(temp)
            std = np.std(temp)
            avg_out[i][0], avg_out[i][1] = avg, std
            i += 1
            writer.writerow(rep1)
            writer.writerow(rep2)

    file_name2 = 'result' + str(part) + '.csv'
    with open(file_name2, 'w') as f:
        writer = csv.writer(f)
        header = ['avg number of outliers', 'std']
        writer.writerow(header)
        for val in avg_out:
            writer.writerow(val)


def rim_collect():
    parser = ap.ArgumentParser()
    parser.add_argument("port", help="choose a port",
                        choices={'19990', '19991', '19992', '19993', '19994', '19995', '19996', '19997', '19998',
                                 '19999'})

    args = parser.parse_args()
    coppelia_port = int(args.port)
    print('collecting')
    # the offset to move the cup in order to make it that rim to rim
    original_y_offset = -(9.65 - 0.75 - 0.1) / 10

    # height of the rim of the target
    original_z_offset = 5.1949e-01 - 0.04

    env = coppelia(port=coppelia_port)
    num_obj = 1
    velocity_pool = [-0.75, - 0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3]
    height_change = np.arange(start=0, stop=0.22, step=0.02)

    trajectory = [[[] for _ in range(num_obj)] for _ in range(110)]

    i = 0
    for velocity in velocity_pool:
        for height in height_change:
            done = False
            env.reset(height=height, velocity=velocity, offset=0, num_obj=1, size=0)
            above = 0
            below = 0
            while not done:
                ret, position = sim.simxGetJointPosition(env.clientID, env.joint6, sim.simx_opmode_buffer)
                ret, rim_position = sim.simxGetObjectPosition(env.clientID, env.rim, -1, sim.simx_opmode_buffer)

                if position < -3:
                    done = True
                # don't need to record the early stage
                if position > -1.8:
                    env.triggerSim()
                    continue

                for obj in range(len(env.object_handles)):
                    _, obj_position = sim.simxGetObjectPosition(env.clientID, env.object_handles[obj][1], -1,
                                                                sim.simx_opmode_buffer)
                    if abs(obj_position[2] - rim_position[2]) < 0.05:
                        # object is higher than the rim
                        if obj_position[2] - rim_position[2] >= 0.005:
                            above = 1

                        # once the object is lower than the rim,
                        # meaning it falls out of the cup
                        if above:
                            if 0.01 >= rim_position[2] - obj_position[2] >= 0.0:
                                x = obj_position[0] - rim_position[0]
                                y = obj_position[1] - rim_position[1]
                                z = obj_position[2]

                                trajectory[i][obj].append([0, x, y, z, height, velocity])
                                below = 1
                    if below:
                        if abs(obj_position[2] - env.plane_position[2]) < 0.01:
                            x = obj_position[0]
                            y = obj_position[1] - rim_position[1]
                            z = obj_position[2]
                            trajectory[i][obj].append([1, x, y, z, height, velocity])

                env.triggerSim()
            env.finish()

            i += 1
        #     if i == 2:

        # print(trajectory)

    file_path = './trajectory/{}/rim.npy'.format(num_obj)

    with open(file_path, 'wb') as f:
        np.save(f, np.array(trajectory, dtype=object))


def rim_test():
    parser = ap.ArgumentParser()
    parser.add_argument("port", help="choose a port",
                        choices={'19990', '19991', '19992', '19993', '19994', '19995', '19996', '19997', '19998',
                                 '19999'})
    parser.add_argument("part", help="choose a part",
                        choices={'0', '2', '4', '6', '8'})

    args = parser.parse_args()
    coppelia_port = int(args.port)
    part = int(args.part)
    print('testing')
    # the offset to move the cup in order to make it that rim to rim
    original_y_offset = -0.088

    # height of the rim of the target
    original_z_offset = 5.1949e-01 - 0.04

    env = coppelia(port=coppelia_port)

    velocity_pool = [-0.75, - 0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3]

    velocity_pool = velocity_pool[part:part + 2]
    height_change = np.arange(start=0, stop=0.22, step=0.02)
    regressions = [[-0.01381495062794, -0.0314944365194866], [0.00762091092174016, -0.143133371358826],
                   [0.0153275217328753, -0.180639958097821], [-0.0117075835665067, -0.0696382679873043],
                   [-0.0139727824264103, -0.0461796994010607], [0.00325178829106418, -0.12277129157023],
                   [0.0085472219520145, -0.163080342389919], [0.00899469057718913, -0.152634034554164],
                   [0.0108376805981, -0.170764420043539], [-0.00290714353322983, -0.105323242644469]]

    result = np.zeros((22, 2, 5))
    avg_out = np.zeros((22, 2))

    combination = 0
    reg_idx = 0
    for velocity in velocity_pool:
        x = 1
        for height in height_change:
            # for each combination, we test with different number of objects, and we repeat this twice
            for rep in range(2):
                diff_num = 0
                for num_obj in range(5):
                    done = False
                    size = np.random.randint(3)
                    y_displacement = regressions[reg_idx][0] * x + regressions[reg_idx][1]
                    # we want the cube to fall into the target
                    # therefore if the landing offset is smaller than the original distance between cup the target
                    # we will move the cup towards target
                    # if the offset is greater than the distance, we will move the cup away from the target
                    # both value a negative
                    if y_displacement < original_y_offset:
                        y_displacement = abs(y_displacement - original_y_offset)
                    else:
                        y_displacement = original_y_offset - y_displacement

                    env.reset(height, velocity, y_displacement, num_obj, size)

                    while not done:
                        # determine termination
                        ret, position6 = sim.simxGetJointPosition(env.clientID, env.joint6, sim.simx_opmode_buffer)
                        ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(env.clientID, env.target,
                                                                                         sim.simx_opmode_buffer)
                        # when we poured everything out
                        if position6 < -2.8:
                            done = True
                        env.triggerSim()

                    # make sure everything setting down before we read outliers
                    temp = 0
                    filtered_reading = []
                    while temp < 20:
                        env.triggerSim()
                        temp += 1
                        ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(env.clientID, env.target,
                                                                                         sim.simx_opmode_buffer)
                        filtered_reading.append(forceVector2[2])
                    final_reading = np.mean(filtered_reading)
                    target_weight = -1 * final_reading - env.target_box_weight
                    outlier_weight = env.total_block_weight - target_weight
                    num_outlier = 0
                    # print(outlier_weight)
                    # if we have any outlier
                    if outlier_weight > 0:
                        num_outlier = round(outlier_weight / env.single_block_weight)
                        # at least we don't want impossible number
                        if num_outlier > env.num_object:
                            num_outlier = env.num_object
                        result[combination][rep][diff_num] = num_outlier
                    print(f'combination {combination}, rep {rep}, round {diff_num}, outlier{num_outlier}')
                    print('---------------------------------')
                    env.finish()

                    time.sleep(1)
                    diff_num += 1
            combination += 1
            x += 1
        reg_idx += 1
    file_name = 'raw_rim_result' + str(part) + '.csv'
    # print(result[:3])
    with open(file_name, 'w') as f:
        i = 0
        writer = csv.writer(f)
        for combination in result:
            rep1 = combination[0]
            rep2 = combination[1]
            temp = np.concatenate((rep1, rep2))
            avg = np.mean(temp)
            std = np.std(temp)
            avg_out[i][0], avg_out[i][1] = avg, std
            i += 1
            writer.writerow(rep1)
            writer.writerow(rep2)

    file_name2 = 'rim_result' + str(part) + '.csv'
    with open(file_name2, 'w') as f:
        writer = csv.writer(f)
        header = ['avg number of outliers', 'std']
        writer.writerow(header)
        for val in avg_out:
            writer.writerow(val)


# read()
# collect()
# test()
rim_collect()
# rim_test()
