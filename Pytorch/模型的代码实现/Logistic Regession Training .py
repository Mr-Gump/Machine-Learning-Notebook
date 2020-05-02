import  torch
import torch.nn as nn
import  matplotlib.pyplot as plt
import numpy as np
# 设置随机数种子
torch.manual_seed(10)

# ============================ step 1/5 生成数据 ============================
sample_nums = 100   # 定义样本个数
mean_value = 1.7    # 定义样本均值
bias = 1    # 偏移量
n_data = torch.ones(sample_nums , 2)      # 用来构造数据
x0 = torch.normal(mean_value*n_data , 1) + bias    # 左侧数据点,每一行为一个坐标点 shape=(100,2)
y0 = torch.zeros(sample_nums)               # 左侧的标签，分类为1 shape=(100,1)
x1 = torch.normal(-mean_value*n_data,1) + bias  #右侧数据点,每一行为一个坐标点  shape=(100,2)
y1 = torch.ones(sample_nums)               #右侧标签 分类为2 shape=(100,1)
train_x = torch.cat((x0,x1),0)             #拼接x数据
train_y = torch.cat((y0,y1),0)             #拼接y标签


# ============================ step 2/5 选择模型 ============================
class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()
        self.features = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()   # 实例化逻辑回归模型

# ============================ step 3/5 选择损失函数 ============================
loss_fn = nn.BCELoss()

# ============================ step 4/5 选择优化器   ============================
lr = 0.01  # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(),lr=lr,momentum=0.9)  #交叉熵

# ============================ step 5/5 模型训练 ============================
for iteration in range(1000):

    # 前向传播
    y_pred = lr_net(train_x)

    # 计算loss
    loss = loss_fn(y_pred.squeeze(),train_y)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 清空梯度
    optimizer.zero_grad()

    # 绘图
    if iteration%20 ==0 :    # 每训练20次作图

        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break

