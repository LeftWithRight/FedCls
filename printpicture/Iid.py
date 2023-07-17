import matplotlib.pyplot as plt  #绘图模块
from scipy import interpolate  #插值模块
import numpy as np  #数值计算模块
from matplotlib.ticker import FuncFormatter


plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

#设置坐标轴
plt.xlim((0, 50))
plt.ylim((0, 1))

# plt.xlabel("通信轮数")
plt.xlabel("Communication Rounds")
plt.ylabel('Accuracy')
plt.title('IID', size=15, loc='center')
# plt.ylabel('loss')

#设置刻度
x_tick = np.arange(0, 50, 5)
y_tick = np.arange(0, 1, 0.1)

#读取数据
centralFile = 'C:/Users/zhu18/Desktop/CLS/printpicture/dp0.2IIDCentro.txt'
fedFile = 'C:/Users/zhu18/Desktop/CLS/printpicture/dp0.2IIDFedAvg.txt'
blockchainFile = 'C:/Users/zhu18/Desktop/CLS/printpicture/dp0.2IIDOurMethod.txt'
a = np.loadtxt(centralFile)
x = a[:, 0]
y1 = a[:, 1]
b = np.loadtxt(fedFile, usecols=4)
y2 = b
c = np.loadtxt(blockchainFile, usecols=4)
y3 = c

plt.plot(x, y1, label="Centralized Training", marker=">")
plt.plot(x, y2, label="FedAvg", marker=">")
plt.plot(x, y3, label="Our Method", color='red', marker=">")

# plt.plot(x, y1, label="集中式训练", color='black', linewidth=1.0, linestyle='--')  centralized training
# plt.plot(x, y1, label="Centralized Training", color='black', linewidth=1.0, linestyle='--')
# plt.plot(x, y2, label="FedAvg", color='black', linewidth=1.0, linestyle=':')
# plt.plot(x, y3, label="Fedk-ls", color='black', linewidth=1.0, linestyle='-')

# plt.plot(x, y1, color='blue', mark='^', linewidth=1.0, linestyle='-')
#
# plt.plot(x, y2, label="FedAvg")
# plt.plot(x, y1, color='red', linewidth=5.0, linestyle='-')
#
# plt.plot(x, y3, label="FedSm")
# plt.plot(x, y3, color='')

plt.grid(linestyle='-.')  # 生成网格
plt.legend(loc='best')
plt.show()
# tck = interpolate.splrep(x, y)
# xx = np.linspace(min(x), max(x), 200)
# yy = interpolate.splev(xx, tck, der=0)
# print(xx)


