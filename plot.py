import matplotlib.pyplot as plt
import numpy as np
import os
import torch as T
import torch.nn.functional as F
import ast
 
path1='./npy/1652051719/actor.npy'
data1=np.load(path1)
data1=data1[data1!=0]
path5='./npy/1652051719/critic.npy'
data5=np.load(path5)

path2='./npy/1652051719/avg.npy'
data2=np.load(path2)


path3='./npy/1652051719/score.npy'
data3=np.load(path3)   
# data2=data2[:350] 
# data2=data2[20:]
path4='./npy/1652051719/out.npy'
data4=np.load(path4)
data4=data4[10:]
path6='./npy/1652051719/reward.npy'
data6=np.load(path6)


fig = plt.figure()
fig.set_size_inches(10.5, 6.5)

plt.subplot(2,3, 1)
plt.plot(data3, '.b')
plt.plot(data2, 'r')
plt.title("score")

plt.subplot(2,3, 2)
plt.plot(data2, '.r')
plt.title("avg score")

reward=['y','z','b','sm','out','hit','bo','t_80','r','s','-','-']
print(data6.shape)
# exit(0)
norm=np.sum(np.abs(data6))
plt.subplot(2,3,3)
plt.bar(reward,data6/norm)
print(data6)
# plt.plot(data2, 'r')
plt.title("reward counts")

outlier = np.zeros(len(data4))
for i in range(len(data4)):
    outlier[i] = np.mean(data4[max(0, i - 100):(i + 1)])
print(f'0 outlier{data4[data4==0].shape}, less than 4{data4[data4<4].shape}, greater than 7{data4[data4>7].shape}')
print(data4.shape)
print(f'mean {np.mean(data4[-100:])} std {np.std(data4[-100:]):.4f}')
# print(len(data4))
plt.subplot(2,3,4)
plt.plot(data4,'o')
plt.plot(outlier,'r')
plt.title("outlier")

plt.subplot(2,3,5)
plt.plot(data1,'r.')
plt.title("actor")

plt.subplot(2,3,6)
plt.plot(data5)
plt.title("critic")

plt.savefig('./img/td3/1652051719.png',dpi=256)
plt.show()
