import torchvision
import struct
from torchvision import transforms
from model import Mnist_2NN
import os
from model import *
from torch import optim
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
import torch
import torch
from Getdata import *
import argparse
from torch.utils.data import TensorDataset

# python main -iid 0

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="centrol")
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-epoch', '--epoch', type=int, default=150, help='train epoch')
parser.add_argument('-dp', '--dropout', type=float, default=0.1, help='set the dropout level in net')


args = parser.parse_args()
args = args.__dict__

mnistDataSet = GetDataSet("mnist", args['IID'])

# 加载测试数据
testDataLoader = mnistDataSet.test_data_loader


train_ds = mnistDataSet.train_dataset


test_txt = open("test_accuracy.txt", mode="a")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
net = Mnist_2NN(args['dropout']).to(dev)
localBatchSize = 100
Epoch = args['epoch']
# 定义全局损失函数
# 定义损失函数
# loss_func = nn.CrossEntropyLoss()
loss_func = F.cross_entropy
# 优化算法的，随机梯度下降法
# 使用下降法
opti = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
global_parameters = {}
# 得到每一层中全连接层中的名称fc1.weight
# 以及权重weights(tenor)
# 得到网络每一层上
for key, var in net.state_dict().items():
    # print("key:"+str(key)+",var:"+str(var))
    print("张量的维度:" + str(var.shape))
    print("张量的Size" + str(var.size()))
    global_parameters[key] = var.clone()
net.load_state_dict(global_parameters, strict=True)



# 加载本地数据
train_dl = DataLoader(train_ds, batch_size=localBatchSize, shuffle=True)
# 设置迭代次数
for epoch in range(1, Epoch+1):
    for data, label in train_dl:
        # 加载到GPU上
        data, label = data.to(dev), label.to(dev)
        # 模型上传入数据
        preds = net(data)
        # 计算损失函数

        loss = loss_func(preds, label)
        # 反向传播
        loss.backward()
        # 计算梯度，并更新梯度
        opti.step()
        # 将梯度归零，初始化梯度
        opti.zero_grad()

        global_parameters = net.state_dict()
    '''
        这里应该记录一下模型得损失值 写入到一个txt文件中
    '''

    # #  加载Server在最后得到的模型参数
    # if epoch % 3 == 0:
    #     epochnum = epoch / 3
    #     epochnum = int(epochnum)


    net.load_state_dict(global_parameters, strict=True)
    sum_accu = 0
    num = 0
    loss_num = 0
    # 载入测试集
    for data, label in testDataLoader:
        data, label = data.to(dev), label.to(dev)
        preds = net(data)
        loss1 = loss_func(preds, label)
        loss_num += loss1.item()
        preds = torch.argmax(preds, dim=1)
        sum_accu += (preds == label).float().mean()
        num += 1
    print("\n" + 'accuracy: {}'.format(sum_accu / num))
    loss_value = loss_num / len(testDataLoader)
    print("loss:" + str(loss_value) + "\n")
    test_txt.write(str(epoch) + "  ")
    test_txt.write(str(float(sum_accu / num)) + "  ")
    test_txt.write(str(float(loss_value))+"\n")
    # test_txt.close()

