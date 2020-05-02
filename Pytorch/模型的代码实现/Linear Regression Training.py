import torch
import matplotlib.pyplot as plt
torch.manual_seed(10)  # 生成随机数种子

lr = 0.05 # 学习率

# 创建训练数据
x = torch.rand(20,1) * 10   # shape=(20,1) 0,10的数
y = 3*x + (5 + torch.randn(20,1))    # shape=(20,1) y=3x+5+随机数

# 构建线性回归参数
# y = wx + b
w = torch.randn((1),requires_grad=True)
b = torch.zeros((1),requires_grad=True)

for iteration in range(1000):

    # 前向传播
    wx = torch.mul(w,x)
    y_pred = torch.add(wx,b)

    # 计算 MSE(均方根) Loss
    loss = (0.5 * (y-y_pred) ** 2) . mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr*b.grad)   # 自减
    w.data.sub_(lr*w.grad)

    # 清零张量梯度(否则会叠加)
    w.grad.zero_()
    b.grad.zero_()

    # 绘图
    if iteration % 20 == 0:

        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration: {}\nw: {} b: {}".format(iteration, w.data.numpy(), b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break