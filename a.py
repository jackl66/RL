import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
 
 
# avg_out = np.zeros((240, 2))
# print(avg_out.shape)
# print(avg_out[:2])
# g = [1, 2, 3]
# t = [4, 5, 6]
# gg = np.concatenate((g,t))
# print(gg)
#
# exit()
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-f", help="provide test case filename as argument", required=True)
# args = parser.parse_args()
# file_name = args.f
# print(file_name)
# exit(0)
# trajectory = [1, 2, 3]
# g = np.ones(3)
# g -= trajectory
# print(g)
# exit()
# f'{states.size()},{actions.size()},{log_probs.size()}\n{returns.size()},{cross_sections.size()},{advantages.size()}',555)
# g=np.load('./npy/1649257321/actor.npy').flatten()
# print(g.shape)
# plt.plot(g,'.')
# plt.show()
# a = np.array([0, 3, 4])
# b = np.array([-2, 1,5])
# mean_a = np.mean(a)
# mean_b = np.mean(b)
# std_a = np.std(a)
# std_b = np.std(b)
# print(f'{std_a}  {mean_a}m {mean_b} {std_b}')
# st_a = (a - mean_a)/std_a
# st_b = (b - mean_b)/std_b

# print(st_a,st_b)
# exit()
# ori = np.array([-0.837988, 0.859548])
# path2 = 'speed.npy'
# data = np.load(path2)
# v = data[1][:50]
#
# # for it in range(len(v)):
# #     v[it] += np.random.uniform(-0.02
# #                                ,0.02)
# # v[25:]=0
# plt.plot(v,'.', label='rotation velocity')
# plt.legend()
# plt.show()

# path3 = 'position.npy'
# data2 = np.load(path3)
# new_y = data2[7, :40, 0]
# print(new_y.shape)
# new_y[1] = new_y[0]
# new_y[0] = ori[0]
# new_z = data2[7, :40, 1]
# print(new_z.shape)
# new_z[1] = new_z[0]
# new_z[0] = ori[1]
# plt.plot(new_y, '.r',label='y coordinate')
# plt.plot(new_z, '.g',label='z coordinate')
# plt.legend()
# plt.show()
# plt.savefig('speed.png', dpi=256)
# exit()
path2 = './data_numpy_arrays/image_arr_train.npy'
path3 = './data_numpy_arrays/num_arr_train.npy'
path2_t = './data_numpy_arrays/image_arr_test.npy'
path3_t = './data_numpy_arrays/num_arr_test.npy'
path4 = './data_numpy_arrays/40_mm_sphere_images_0.npy'
sim = np.load(path4)
print(f'{sim.shape} ')
idx = 34
x_box = [30, 110]
y_box = [20, 100]
fig = plt.figure()
fig.set_size_inches(9.5, 6.5)
# plt.subplot(2, 2, 1)
# plt.imshow(sim[idx,0,x_box[0]:x_box[1],y_box[0]:y_box[1]])
# plt.subplot(2, 2, 2)
# plt.imshow(sim[idx,1,x_box[0]:x_box[1],y_box[0]:y_box[1]])
# plt.subplot(2, 2, 3)
# plt.imshow(sim[idx,2,x_box[0]:x_box[1],y_box[0]:y_box[1]])
# plt.subplot(2, 2, 4)
# plt.imshow(sim[idx, 3, x_box[0]:x_box[1], y_box[0]:y_box[1]])
# plt.show()
# #
# # exit(0)
label = np.load(path3_t)

data = np.load(path2_t)
# print(label[-30:])
#
# print(f'{label.shape} and {data.shape}')
# exit()
plt.imshow(data[0])
print("gt", label[0])
plt.show()

plt.imshow(data[1])
print("gt", label[2])
plt.show()
plt.imshow(data[3])
print("gt", label[3])
plt.show()
exit()

t = np.array([-0.755330, 0.8527494])
bound = np.array(([-0.1, 0.1]))
rand = np.array([-5 * math.pi / 180, 5 * math.pi / 180])
print(rand)
# y position
# negative random, moving forward and backward
t1r1b1 = t[0] + rand[0] + bound[0]

# y position
# positive random, moving forward and backward

t1r2b2 = t[0] + rand[1] + bound[1]
print(t1r1b1, t1r2b2)
print('---------------------')

# z position
# negative random, moving forward and backward
t2r1b1 = t[1] + rand[0] + bound[0]

# z position
# positive random, moving forward and backward
t2r2b2 = t[1] + rand[1] + bound[1]

print(t2r1b1, t2r2b2)
exit(0)

# path1='./test/actor1639344259.npy'
# data1=np.load(path1)
path2 = './test/avg1639344259.npy'
data2 = np.load(path2)
path3 = './test/score1639344259.npy'
data3 = np.load(path3)
path4 = './test/out1639344259.npy'
data4 = np.load(path4)
# path5='./test/critic1639344259.npy'
# data5=np.load(path5)
path6 = './test/reward1639344259.npy'
data6 = np.load(path6)

fig = plt.figure()
fig.set_size_inches(9.5, 6.5)

plt.subplot(2, 3, 1)
plt.plot(data3, '.b')
plt.plot(data2, 'r')
plt.title("score")

plt.subplot(2, 3, 2)
plt.plot(data2, '.r')
plt.title("avg score")

reward = ['j1', 'j2', 'bound', 'out', 'hit', 'bonus']
total = np.sum(data6[:-1])
plt.subplot(2, 3, 5)
plt.bar(reward, data6[:-1] / total)
# plt.plot(data2, 'r')
plt.title("reward counts")

outlier = np.zeros(len(data4))
for i in range(len(data4)):
    outlier[i] = np.mean(data4[max(0, i - 100):(i + 1)])
# print(np.mean(data4))
# print(np.std(data4))
# print(len(data4))
plt.subplot(2, 3, 4)
plt.plot(data4, 'o')
plt.plot(outlier, 'r')
plt.title("outlier")

# plt.subplot(2,3,5)
# plt.plot(data1,'r')
# plt.title("actor")

# plt.subplot(2,3,6)
# plt.plot(data5)
# plt.title("critic")
# plt.subplots_adjust(left=0.05, right=1, bottom=0.0, top=1.0)
plt.savefig('./img/1639344259.png', dpi=256)
plt.show()
