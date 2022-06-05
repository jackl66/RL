import sim
import numpy as np
import torch as T
import torch.nn as nn
import math
import random


# class to handle all interaction with coppelia
class sin_coppelia:
    def __init__(self, depth, token, mass=14.375e-03, object_type="cube", size=0.025, num_object=50, port=19997,
                 model=0, eval=0, same='1'):
        self.num_pour_out = 0
        self.mass = mass
        self.object = object_type
        self.size = size
        self.token = token
        self.num_object = num_object
        self.single_block_weight = mass * 9.81
        self.total_block_weight = num_object * mass * 9.81
        self.port = port
        self.pouring_speed = -0.4
        self.out = nn.AdaptiveAvgPool2d(16)
        self.time_factor = 0.05
        self.using_depth = depth
        self.done = False
        self.final_reward = False
        self.weight_history = []
        self.num_outlier = 0
        self.big_box_weight = 0
        self.target_box_weight = 0
        self.reward_history = np.zeros(10)
        self.new_pose = np.zeros(2)

        # min_y, max_y,min_z,max_z
        self.bound = np.array([-0.66745, -0.85798, 0.66901, 0.85954])
        self.larger_bound = np.array([-0.56745, -0.95798, 0.56901, 0.95954])

        self.old_z = 0
        self.old_y = 0
        self.ori_z = 0
        self.emptyBuff = bytearray()
        self.init_amount = [30, 35, 40, 45, 50]

        # smaller to accommodate the shrink factor
        # self.init_amount = [20, 25, 30, 35, 40]

        self.obj_string = ['Cuboid', 'Cylinder', 'Sphere']
        self.obj_size = [0.025, 0.019, 0.016]

        # self.eval_init_amount = [32, 36, 43, 46, 51]
        self.eval_init_amount = [18, 22, 29, 34, 42]
        self.eval_obj_size = [0.027, 0.020, 0.014]
        self.init_history = np.zeros(9)
        self.init_error = 0

        # action space for DQN
        self.action = [[-0.02, 0.02], [-0.02, -0.02], [-0.02, 0],
                       [0, 0.02], [0, -0.02], [0, 0],
                       [0.02, 0.02], [0.02, -0.02], [0.02, 0], ]

        self.action_3 = [[-0.02, -0.02, -0.05], [-0.02, -0.02, 0], [-0.02, -0.02, 0.05], [-0.02, 0, -0.05],
                         [-0.02, 0, 0], [-0.02, 0, 0.05],
                         [-0.02, 0.02, -0.05], [-0.02, 0.02, 0], [-0.02, 0.02, 0.05], [0, -0.02, -0.05], [0, -0.02, 0],
                         [0, -0.02, 0.05],
                         [0, 0, -0.05], [0, 0, 0], [0, 0, 0.05], [0, 0.02, -0.05], [0, 0.02, 0], [0, 0.02, 0.05],
                         [0.02, -0.02, -0.05],
                         [0.02, -0.02, 0], [0.02, -0.02, 0.05], [0.02, 0, -0.05], [0.02, 0, 0], [0.02, 0, 0.05],
                         [0.02, 0.02, -0.05],
                         [0.02, 0.02, 0], [0.02, 0.02, 0.05]]

        self.model = model
        self.eval = eval

        self.interval = 1 / 7
        self.img_bound = np.arange(start=0, stop=8 / 7, step=1 / 7, dtype=float)
        self.pdf = np.zeros(7)
        self.same = int(same)
        # self.target_container_scale_factor_pool = [0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.velocity_pool = [-0.75, - 0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3]

        # shrink to only test hard cases
        self.height_change_pool = np.arange(start=0.02, stop=0.22, step=0.02)
        # self.height_change_pool = np.array([0.18, 0.2])

        self.target_container_scale_factor_pool = [0.55, 0.65]
        self.height_scale_factor_pool = [2.25, 2]

        self.ori_rim = [-(9.65 - 0.75) / 10, (5.1949 + 0.80183 / 2) / 10]

        # linear regression model to pick the starting point for exploration
        # dim 0 is the index of velocity pool
        self.regressions = [[-0.0114008425763159, -0.0404692104368499], [-0.0130502221620444, -0.0321985361070344],
                            [-0.0133292214436965, -0.0328235057267276], [-0.0131503900548972, -0.0488519692059719],
                            [-0.0139437097238772, -0.0313131784020048], [-0.0144071032603582, -0.0427860965331396],
                            [-0.0130329799471479, -0.0359688442764861], [-0.0131965924393047, -0.0297948994419791],
                            [-0.0128810468948249, -0.0373327018636646], [-0.0136431304794369, -0.03255832213344]]
        self.time = np.arange(0, 127 * 4)

        self.period_adjustor = [[1.29, 1.31, 1.33], [1.29, 1.31, 1.35], [1.31, 1.34, 1.35], [1.33, 1.36, 1.39]]
        self.sin = [[0.83, -1.2], [0.63, -0.8], [0.53, -0.6], [0.315, -0.4]]

    def reset(self, num_object, obj_shape, size):

        self.done = False
        self.final_reward = False
        self.weight_history = []

        # if we want to evaluate the model
        # use a different set of sizes
        if self.eval and self.same == 1:
            self.num_object = self.eval_init_amount[num_object]
            self.size = self.eval_obj_size[size]
        else:
            self.num_object = self.init_amount[num_object]
            self.size = self.obj_size[size]
        # self.num_object = 40

        # todo need to change in the future
        self.size = self.obj_size[1]
        self.iteration = 0
        self.object = self.obj_string[obj_shape]
        self.total_block_weight = self.num_object * self.mass * 9.81
        self.backward = False
        # record the distribution of different init combination
        self.init_history[num_object] += 1
        self.init_history[5 + size] += 1
        self.init_history[8 + obj_shape] += 1

        # randomly makes the target smaller and taller
        self.width_scale = np.random.choice(self.target_container_scale_factor_pool, 1)[0]
        self.height_scale = np.random.choice(self.height_scale_factor_pool, 1)[0]
        # self.width_scale = 0.55
        # self.height_scale = 2.25

        self.target_container_left_rim = -9.6500e-01 + (1.5007e-01) / 2 * self.width_scale
        self.target_container_right_rim = -9.6500e-01 - (1.5007e-01) / 2 * self.width_scale
        self.target_container_rim_hight = +5.1949e-01 + (
            8.0195e-02) / 2 * self.width_scale + 9.4118e-02 * self.height_scale
        # the offset to move the cup in order to make it that rim to rim
        self.original_y_offset = -0.085 * self.width_scale

        # construct different sin wave velocity
        # Amplitude of the sine wave is sine of a variable like time
        # a*b*time * c
        # a is the frequency, controls Period
        # b is for smoothing it
        # c is Amplitude

        # todo test again,
        # todo need to move the arm much closer to the rim
        # todo overwrite the offset when cup has rotated enough and move it towards the rim
        # the bigger the value, the shorter the period
        self.period_adjustor = [[1.43, 1.45, 1.49], [1.29, 1.31, 1.35], [1.31, 1.34, 1.35], [1.41, 1.43, 1.45]]
        self.sin = [[0.83, -1.2], [0.63, -0.8], [0.53, -0.6], [0.73, -1]]
        amp_idx = np.random.randint(0, high=4)
        # amp_idx = 3
        period_scale_idx = np.random.randint(0, high=3)
        # period_scale_idx = 0
        period = self.sin[amp_idx][0]
        amp = self.sin[amp_idx][1]
        period_scale = self.period_adjustor[amp_idx][period_scale_idx]
        self.amplitude = np.sin(period * period_scale * 1 / 20 * self.time) * amp

        # the height of the rim is fixed for each episode
        height_idx = np.random.randint(8, high=10)
        height = self.height_change_pool[height_idx]
        old_height = height
        height += 9.4118e-02 * self.height_scale
        large_velocity_bound = 0
        # now the max speed is chosen by amp
        # pouring_idx = np.random.randint(0, high=10)
        # self.pouring_speed = self.velocity_pool[pouring_idx]
        if amp != -0.8 and amp != -1.2 and amp != -1:
            pouring_idx = self.velocity_pool.index(amp)
        else:
            pouring_idx = 0
            large_velocity_bound = 0.01
        # the offset is the maximum offset it can go
        # but since the velocity is much slower than the max for the most of the time
        # the offset for the warm-up episode will be a linear transformation
        # that start from left rim + 0.005 to offset

        regression = self.regressions[pouring_idx]
        y_displacement = regression[0] * (height_idx + 1) + regression[1]

        if y_displacement < self.original_y_offset:
            y_displacement = abs(y_displacement - self.original_y_offset)
            # temp = abs(temp-self.original_y_offset)
        else:
            y_displacement = 0

            # old_y = y_displacement

            # # todo 0.01 diff between 1.40 and 1.48
            # -0.7882819175720215
            # 0.13544883242797845

            y_displacement -= (1.5007e-01) * 0.5 * (1 - self.width_scale) + 0.05

        print(f'old height {old_height}, h {height}, amp {amp}, period {period}, period_scale {period_scale},',
              f'shrink factor {self.width_scale}, height scale {self.height_scale}, num obj {self.num_object}')

        # update the bound based on the scale
        self.bound = np.array(
            [self.target_container_left_rim + abs(y_displacement) + 0.02 + (1.5007e-01) * 0.5 * (1 - self.width_scale),
             self.target_container_left_rim + large_velocity_bound, 0.66901, 0.85954])

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
        ret, pack_img = sim.simxGetStringSignal(self.clientID, 'pack_img', sim.simx_opmode_streaming)

        # only for rim to rim case
        ret0, signalValue = sim.simxGetFloatSignal(self.clientID, 'first_arm_done', sim.simx_opmode_streaming)

        # sensor under the container
        res, self.target = sim.simxGetObjectHandle(self.clientID, '/Floor/f2/Box/f1', sim.simx_opmode_blocking)
        returnCode, state, forceVector, toequeVector = sim.simxReadForceSensor(self.clientID, self.target,
                                                                               sim.simx_opmode_streaming)

        # sensor under the box
        res, self.box = sim.simxGetObjectHandle(self.clientID, '/Floor/f2', sim.simx_opmode_blocking)
        returnCode, state, forceVector, toequeVector = sim.simxReadForceSensor(self.clientID, self.box,
                                                                               sim.simx_opmode_streaming)
        # handles for joints
        res, self.joint6 = sim.simxGetObjectHandle(self.clientID, 'UR5_joint6', sim.simx_opmode_blocking)
        res, self.rim = sim.simxGetObjectHandle(self.clientID, 'rim', sim.simx_opmode_blocking)

        # handle for the pouring cup and target box
        res, self.cup = sim.simxGetObjectHandle(self.clientID, 'source', sim.simx_opmode_blocking)
        res, self.target_area = sim.simxGetObjectHandle(self.clientID, 'target', sim.simx_opmode_blocking)

        # position for rotating back
        ret, position = sim.simxGetJointPosition(self.clientID, self.joint6, sim.simx_opmode_streaming)
        ret, position = sim.simxGetObjectPosition(self.clientID, self.rim, -1, sim.simx_opmode_streaming)

        # positions for cup and target box
        ret, position = sim.simxGetJointPosition(self.clientID, self.cup, sim.simx_opmode_streaming)
        ret, position = sim.simxGetJointPosition(self.clientID, self.target_area, sim.simx_opmode_streaming)

        # depth info in 64*64 and avg pooling to 9*9
        res, self.camDepth = sim.simxGetObjectHandle(self.clientID, 'depth', sim.simx_opmode_blocking)
        returnCode, resolution, depthImage = sim.simxGetVisionSensorDepthBuffer(self.clientID, self.camDepth,
                                                                                sim.simx_opmode_streaming)
        self.triggerSim()
        self.setNumberOfBlock()
        ret0, signalValue1 = sim.simxGetFloatSignal(self.clientID, 'first_arm_done', sim.simx_opmode_buffer)
        while signalValue1 != 99:
            self.triggerSim()
            ret0, signalValue1 = sim.simxGetFloatSignal(self.clientID, 'first_arm_done', sim.simx_opmode_buffer)

        # move the arm to target height
        # and stay there for this episode
        self.py_get_pose()
        self.py_moveToPose([y_displacement, height], 0)
        loop = 20
        while loop > 0:
            self.triggerSim()
            loop -= 1

        sim.simxSetIntegerSignal(self.clientID, 'arm_done', 99, sim.simx_opmode_oneshot)

        # wait for objects to be created
        while True:
            self.triggerSim()
            # signal to mark the creation is finished
            ret, signalValue = sim.simxGetFloatSignal(self.clientID, 'init_done', sim.simx_opmode_blocking)
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

        # record the weights of the box to determine when is done
        ret, state, forceVector, torqueVector = sim.simxReadForceSensor(self.clientID, self.target,
                                                                        sim.simx_opmode_buffer)
        self.target_box_weight = -1 * forceVector[2]
        # print(self.target_box_weight)
        # big box weight = big box + small box(target)
        ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(self.clientID, self.box,
                                                                         sim.simx_opmode_buffer)
        self.big_box_weight = -1 * forceVector2[2]

        # prepare data for the first observation
        new_state = []

        if self.using_depth:
            # filter the depth image to be 9*9
            returnCode, resolution, depthImage = sim.simxGetVisionSensorDepthBuffer(self.clientID, self.camDepth,
                                                                                    sim.simx_opmode_buffer)

            depthImage = np.array(depthImage).reshape((resolution[0], resolution[1]))
            cropped = depthImage[10:100, 10:100]
            depthImage = np.array(cropped).reshape((1, 90, 90))
            depthImage_T = T.from_numpy(depthImage)
            # print(depthImage.shape)

            depth_filtered = self.out(depthImage_T)
            depth_filtered = np.squeeze(depth_filtered.cpu().detach().numpy())
            # print(depth_filtered.shape)

        ret, self.ori_position = sim.simxGetJointPosition(self.clientID, self.joint6, sim.simx_opmode_buffer)

        ret, rim_position = sim.simxGetObjectPosition(self.clientID, self.rim, -1, sim.simx_opmode_buffer)

        self.py_get_pose()
        self.ori_z = self.old_z
        new_state.append(0)  # joint6 angular displacement
        new_state.append(self.pouring_speed)  # joint6 angular velocity

        # y distance between the rim and the left of the container
        new_state.append(self.target_container_left_rim - rim_position[1])
        # y distance between the rim and the right of the container
        new_state.append(self.target_container_right_rim - rim_position[1])
        # height difference between the rim and the target container
        new_state.append(rim_position[2] - self.target_container_rim_hight)

        new_state.append(np.zeros(7))  # cross-section heat map

        if self.using_depth:
            new_state.append(depth_filtered)  # 9*9 array describe the depth info

        self.triggerSim()

        return new_state

    # call Lua script to spawn the objects the in scene
    def setNumberOfBlock(self):
        emptyBuff = bytearray()
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, 'target',
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'setNumberOfBlocks',
                                                                                    [self.num_object],
                                                                                    [self.mass, self.size,
                                                                                     self.width_scale,
                                                                                     self.height_scale],
                                                                                    [self.object], emptyBuff,
                                                                                    sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print("remote call results:", retStrings[0])
        else:
            print("remote init function call failed")
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
        self.old_y = retFloats[0]
        self.old_z = retFloats[1]

        if res != sim.simx_return_ok:
            print("something is wrong in getting pose")
            self.finish()
            exit(0)

    # call Lua script to control the end-effector
    def py_moveToPose(self, displacement, hold_z):

        penalty = 0
        # see ddpg_torch.py choose_action() for more info
        self.new_pose[0] = self.old_y + displacement[0]
        # print(self.old_y, self.new_pose, self.bound[0])
        self.new_pose[1] = self.old_z + displacement[1]

        # if they reach the boundary, apply penalty
        # because even on the boundary, it is not likely to pour into the target
        # boundary is calculated using start pose + rand + max_action
        # also we don't want it to go out of bound, so we clip

        if self.new_pose[0] >= self.bound[0]:
            # print(99, self.new_pose, self.bound[0])
            penalty += 1
            self.new_pose[0] = self.bound[0]

        if self.new_pose[0] <= self.bound[1]:
            # print(123)
            penalty += 1
            self.new_pose[0] = self.bound[1]

        # we won't change height for now
        # if self.new_pose[1] <= self.bound[2]:
        #     penalty += 1
        #     if self.new_pose[1] >= self.larger_bound[2] and self.model == 6:
        #         self.new_pose[1] = self.larger_bound[2]
        #     else:
        #         self.new_pose[1] = self.bound[2]
        #
        # if self.new_pose[1] >= self.bound[3]:
        #     penalty += 1
        #     if self.new_pose[1] >= self.larger_bound[3] and self.model == 6:
        #         self.new_pose[1] = self.larger_bound[3]
        #     else:
        #         self.new_pose[1] = self.bound[3]
        if penalty > 1:
            print("something is wrong in bound penalty")
            self.finish()
            exit(0)
        res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, 'UR5',
                                                                                    sim.sim_scripttype_childscript,
                                                                                    'py_moveToPose',
                                                                                    [hold_z],
                                                                                    [self.new_pose[0],
                                                                                     self.new_pose[1], self.ori_z],
                                                                                    [], self.emptyBuff,
                                                                                    sim.simx_opmode_blocking)

        if res != sim.simx_return_ok:
            print("remote move pose function call failed")
            self.finish()
            exit(0)

        return penalty

    def step(self, actions, episode=0):
        actions /= 100
        ret, error = sim.simxGetFloatSignal(self.clientID, 'exception', sim.simx_opmode_blocking)
        if error == 99:
            self.init_error += 1

        # determine termination
        ret, state, forceVector, torqueVector = sim.simxReadForceSensor(self.clientID, self.box,
                                                                        sim.simx_opmode_buffer)

        current_box_weight = -1 * forceVector[2]

        remaining_object_weight = self.total_block_weight - (current_box_weight - self.big_box_weight)
        self.weight_history.append(remaining_object_weight)
        if len(self.weight_history) > 30:
            remaining_object_weight = np.median(np.array(self.weight_history[-10:]))

        # when we poured everything out
        if remaining_object_weight <= self.single_block_weight * 1.1:
            print("pouring done")
            self.done = True

        # read state info
        new_state = []

        ret, position6 = sim.simxGetJointPosition(self.clientID, self.joint6, sim.simx_opmode_buffer)
        if position6 < -2.11:
            self.backward = True
        # exception handler
        if position6 < -3.3 and self.done is False or (self.backward and position6 > -1.1):
            print('done by now')
            self.done = True

        # cross-section view
        returnCode, resBlob = sim.simxGetStringSignal(self.clientID, 'pack_img', sim.simx_opmode_buffer)
        self.pdf = np.zeros(7)
        # the image x coordinate is scaled from 0 to 1
        # divided it into 7 slots
        blobs = sim.simxUnpackFloats(resBlob)
        num_blob = int(blobs[0])
        offset = int(blobs[1])
        if blobs[0] == 0 or position6 > -1:
            pass
        else:
            for i in range(num_blob):
                blobX = blobs[offset * i + 5]
                blobW = blobs[offset * i + 7]

                upper = blobX + blobW / 2
                lower = blobX - blobW / 2
                idx = math.floor(blobX / self.interval)
                try:
                    if idx == 0:
                        self.rw(0, upper, 1 / 7)
                        self.rw2(0, 1 / 7, lower)
                    else:
                        self.rw(idx, upper, self.img_bound[idx])
                        self.rw2(idx, self.img_bound[idx], lower)
                except:
                    upper_idx = math.floor(upper / self.interval)
                    lower_idx = math.floor(lower / self.interval)
                    while upper_idx > lower_idx:
                        self.pdf[upper_idx] = 1
                        upper_idx -= 1
                    self.init_error += 1
                    print('error in img')
                self.pdf = np.abs(self.pdf)
                img_sum = np.sum(self.pdf)
                if img_sum != 0:
                    self.pdf /= img_sum

        self.py_get_pose()
        old_y = self.old_y
        old_z = self.old_z

        ret, rim_position = sim.simxGetObjectPosition(self.clientID, self.rim, -1, sim.simx_opmode_buffer)
        rotated = position6 - self.ori_position

        new_state.append(rotated)
        new_state.append(self.pouring_speed)
        # y distance between the rim and the left of the container
        new_state.append(self.target_container_left_rim - rim_position[1])
        # y distance between the rim and the right of the container
        new_state.append(self.target_container_right_rim - rim_position[1])
        # height difference between the rim and the target container
        new_state.append(rim_position[2] - self.target_container_rim_hight)
        new_state.append(self.pdf)

        if self.using_depth:
            returnCode, resolution, depthImage = sim.simxGetVisionSensorDepthBuffer(self.clientID, self.camDepth,
                                                                                    sim.simx_opmode_buffer)

            depthImage = np.array(depthImage).reshape((resolution[0], resolution[1]))
            cropped = depthImage[10:100, 10:100]
            depthImage = np.array(cropped).reshape((1, 90, 90))
            depthImage_T = T.from_numpy(depthImage)

            depth_filtered = self.out(depthImage_T)
            depth_filtered = np.squeeze(depth_filtered.cpu().detach().numpy())
            new_state.append(depth_filtered)

        reward = 0
        penalty = 0
        D_speed = 0

        # use median filter to reduce the force reading noise
        outlier_reading = []
        total_reading = []
        if not self.done:
            # print(self.old_y, (self.old_y - self.target_container_left_rim))

            if episode < 7 and not self.eval:
                actions[0] = -0.005
                if not self.backward and position6 < -1.3:
                    self.py_moveToPose([actions, 0], 1)

            elif (10 > episode > 7 and not self.eval) or self.eval:
                actions[0] /= 10
                # move the end effector to target position,
                # for 1D models, action's shape (1,) = (displacement_y)
                # for 2D models, action's shape (2,) = (displacement_y, delta v)
                if len(actions) == 1:
                    penalty = self.py_moveToPose([actions, 0], 1)
                else:
                    penalty = self.py_moveToPose([actions[0], 0], 1)

                    if position6 < -1:
                        D_speed = actions[1]
                        self.pouring_speed += D_speed
                # wait for the arm to reach the target position
                force_out = 0  # exception handler
                while (abs(self.old_y - self.new_pose[0]) > 0.005
                       or abs(self.old_z - self.new_pose[1]) > 0.005) and force_out < 20:
                    self.triggerSim()
                    force_out += 1
                    self.py_get_pose()
            self.pouring_speed = self.amplitude[self.iteration]
            # print(self.pouring_speed, self.iteration)
            errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.joint6, self.pouring_speed,
                                                       sim.simx_opmode_oneshot)
            self.triggerSim()
            self.triggerSim()
            self.triggerSim()
            self.triggerSim()

            sim.simxSetJointTargetVelocity(self.clientID, self.joint6, 0, sim.simx_opmode_oneshot_wait)
            self.triggerSim()
        else:
            # we don't care things after pouring is done
            sim.simxSetJointTargetVelocity(self.clientID, self.joint6, 0, sim.simx_opmode_oneshot_wait)
            self.triggerSim()

            # make sure everything setting down before we read outliers
            temp = 0
            while temp < 50:
                if temp < 30:
                    ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(self.clientID, self.target,
                                                                                     sim.simx_opmode_buffer)
                    outlier_reading.append(forceVector2[2])
                    ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(self.clientID, self.box,
                                                                                     sim.simx_opmode_buffer)
                    total_reading.append(forceVector2[2])

                self.triggerSim()
                temp += 1
        # rewards
        '''
        -0.3 every move in the y axis
        encourage fewer movements    #
        '''
        displacement_y = abs(old_y - self.new_pose[0])
        r1 = 0.3 * displacement_y
        reward -= r1
        self.reward_history[0] -= r1  # histogram of each reward

        displacement_z = abs(old_z - self.new_pose[1])
        r2 = 0.03 * abs(displacement_z)
        # reward -= r2
        self.reward_history[1] -= r2

        # -5 if one joint move outside
        r3 = 2 * penalty
        reward -= r3
        self.reward_history[2] -= r3

        # continuity cost to smooth the movement
        # since the largest displacement is 0.08, largest penalty is 2*0.08**2=0.0128, sqrt = 0.1
        # since it's not the primary goal, maybe need to make it smaller
        #     encourage smooth movement  #
        r4 = 0.1 * np.sqrt(displacement_y ** 2 + displacement_z ** 2)
        # reward -= r4
        self.reward_history[3] -= r4

        # rotation speed penalty
        r9 = 0.01 * np.sqrt(D_speed ** 2)
        # reward -= r9
        self.reward_history[8] -= r9

        # punish ccw velocity and encourage cw velocity
        r10 = 0.01 * D_speed
        # reward -= r10
        self.reward_history[9] -= r10

        # time out penalty
        self.iteration += 1

        # if self.iteration > 250:
        #     self.done = True
        #     print('took too long')
        #     reward -= 200
        #     self.reward_history[7] -= 200

        # +10 for each object fall into the target area
        # +50 if the error is less than 5%
        # -20 for each that is out of the target box
        # only give these three rewards after pouring finished
        if self.done:
            # only give final_reward once
            if not self.final_reward:

                hit_reward = 0
                outlier_penalty = 0
                self.num_outlier = 0

                # read the force sensor to determine in/out: total-target= # outside
                # things in the target area, including the target box
                ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(self.clientID, self.target,
                                                                                 sim.simx_opmode_buffer)
                target_weight = np.median(outlier_reading)
                target_weight = -1 * target_weight - self.target_box_weight
                # outlier_weight = self.total_block_weight - target_weight
                ret, state, forceVector2, torqueVector = sim.simxReadForceSensor(self.clientID, self.box,
                                                                                 sim.simx_opmode_buffer)
                total_weight = np.median(total_reading)
                print(total_weight, 999)
                total_weight = -1 * total_weight - self.big_box_weight
                outlier_weight = total_weight - target_weight
                self.num_pour_out = round(total_weight / self.single_block_weight)
                # if we have any outlier
                if outlier_weight > 0:
                    self.num_outlier = round(outlier_weight / self.single_block_weight)
                    # at least we don't want impossible number
                    if self.num_outlier > self.num_pour_out:
                        self.num_outlier = self.num_pour_out
                    # here we use percentage as the penalty
                    outlier_penalty = outlier_weight / target_weight * 100
                    self.reward_history[4] -= outlier_penalty

                # if we have anything in the target area
                if self.num_pour_out > self.num_outlier:
                    hit_reward = (1 - outlier_weight / target_weight) * 50
                    self.reward_history[5] += hit_reward

                    if self.num_outlier / self.num_pour_out < 0.05:
                        hit_reward += 100
                        self.reward_history[6] += 100
                self.final_reward = True
                reward = reward + hit_reward - outlier_penalty
                print("done is", self.done)
        # we want to make the final reward before termination
        done = self.done * self.final_reward

        return new_state, reward, done

    def finish(self):
        x = sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxFinish(self.clientID)
        print('took', self.iteration)

        return self.num_outlier, self.num_pour_out

    def get_reward_history(self):
        return self.reward_history

    def get_init_history(self):
        return self.init_history, self.init_error

    # manually control the scene to go to next time step
    def triggerSim(self):
        e = sim.simxSynchronousTrigger(self.clientID)

    def rw(self, idx, upper, lower):
        if idx >= 6:
            self.pdf[6] += (upper - self.img_bound[idx]) * 7
            return self.img_bound[6]

        if upper > self.img_bound[idx + 1]:
            # middle value[idx] += g[idx]
            upper = self.rw(idx + 1, upper, self.img_bound[idx + 1])
        else:
            if lower <= self.img_bound[idx]:
                self.pdf[idx] += (upper - self.img_bound[idx]) * 7
                upper = self.img_bound[idx]
                return upper
            else:
                self.pdf[idx] += (upper - lower) * 7
                return upper

        if lower > self.img_bound[idx]:
            self.pdf[idx] += (upper - lower) * 7
        else:
            self.pdf[idx] += (upper - self.img_bound[idx]) * 7
            upper = self.img_bound[idx]

        return upper

    def rw2(self, idx, upper, lower):
        if idx <= 0:
            self.pdf[0] += (upper - lower) * 7
            return
        if lower < self.img_bound[idx]:
            self.rw2(idx - 1, self.img_bound[idx], lower)
            if upper > self.img_bound[idx + 1]:
                self.pdf[idx] += (self.img_bound[idx + 1] - self.img_bound[idx]) * 7
            else:
                self.pdf[idx] += (upper - self.img_bound[idx]) * 7
        else:
            if upper > self.img_bound[idx + 1]:

                self.pdf[idx] += (self.img_bound[idx] - lower) * 7
                return
            else:
                self.pdf[idx] += (upper - lower) * 7
                return
