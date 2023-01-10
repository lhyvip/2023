"""
import numpy as np
import  matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
torch.manual_seed(1)#设置 CPU 生成随机数的 种子 ，方便下次复现实验结果。
#为 CPU 设置 种子 用于生成随机数，以使得结果是确定的。
np.random.seed(1)
#seed()中的参数被设置了之后，np.random.seed()可以按顺序产生一组固定的数组，如果使用相同的seed()值，则每次生成的随机数都相同。如果不设置这个值，
# 那么每次生成的随机数不同。但是，只在调用的时候seed()
# 一下并不能使生成的随机数相同，需要每次调用都seed()一下，表示种子相同，从而生成的随机数相同。



#新手画家 (Generator) 在作画的时候需要有一些灵感 (random noise),
# 我们这些灵感的个数定义为 N_IDEAS.
#而一幅画需要有一些规格, 我们将这幅画的画笔数定义一下, N_COMPONENTS
# 就是一条一元二次曲线(这幅画画)上的点个数.
#为了进行批训练, 我们将一整批画的点都规定一下(PAINT_POINTS).
#1.定义参数
batch_size=64# 每一批所处理的数据集
lr_G=0.0001#generator生成器的学习率
lr_D=0.0001#discrimination辨别器的学习率
n_ideas=5#新手画家的随机想法(Generator)
art_components=15 # generator创作的点数


#2. 一次处理中所产生的点数，二维矩阵
paint_points=np.vstack([np.linspace(-1,1,art_components) for _ in range(batch_size)])
# shape(64,15)




#4.生成一批艺术家的优美的画
def artist_works():
    # 从[1,2)的正态分布中随机选取64个数字，并在第二维度新加一个维度
    a=np.random.uniform(1,2,size=batch_size)[:,np.newaxis]
    #[64,1]*[64,15]
    # 生成这些点的纵坐标
    paintings=a*np.power(paint_points,2)+(a-1)
    #转换为tensor
    paintings=torch.from_numpy(paintings).float()
    #(64,15)
    return paintings

#5.构建模型
G=nn.Sequential(
    nn.Linear(n_ideas,128),
    nn.ReLU(),
    nn.Linear(128,art_components),
)
D=nn.Sequential(
    nn.Linear(art_components,128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid(),
)

#6.构建优化器
opt_D=torch.optim.Adam(D.parameters(),lr=lr_D)
opt_G=torch.optim.Adam(G.parameters(),lr=lr_G)


plt.ion()
#7.开始训练
for step in range(10000):
    artist_paintings=artist_works()#著名画家的画
    # 用于随机生成generator的灵感
    G_ideas=torch.randn(batch_size,n_ideas,requires_grad=True)
    # 根据生成的灵感来作画
    G_paintings=G(G_ideas)#新手画家的画



    # discriminator判断这些画作【G自己的灵感画作】
    # 来自著名画家的概率为多少
    pro_artistG=D(G_paintings)#锁住G的参数不求导


    G_loss=torch.mean(torch.log(1-pro_artistG))

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    # discriminator判断这些画作【著名画家画作】
    # 来自著名画家的概率为多少，希望越高越好
    pro_artista = D(artist_paintings)
    pro_artistG = D(G_paintings.detach())  # 锁住G的参数不求导

    D_loss = -torch.mean(torch.log(pro_artista) + torch.log(1 - pro_artistG))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)#这个参数是为了再次使用计算图纸
    opt_D.step()

    if step % 150 == 0:
        print(step)
        plt.cla()
        plt.plot(paint_points[0], G_paintings.data.numpy(
        )[0], c='r', lw=3, label='Generated painting')
        plt.plot(paint_points[0], 2 * np.power(paint_points[0],
                                               2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(paint_points[0], 1 * np.power(paint_points[0],
                                               2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy = %.2f(0.5 for D to converage)' %
                 pro_artista.data.numpy().mean(), fontdict={'size': 15, 'color': 'blue'})
        plt.text(-.5, 2, 'D score = %.2f(-1.38 for G to converage)' % -
        D_loss.data.numpy(), fontdict={'size': 15, 'color': 'blue'})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)
    plt.ioff()
    plt.savefig()
    plt.show()

"""

'''
Description: GAN--study
Autor: 365JHWZGo
Date: 2023-1-8 22:06:12
LastEditors: lhy
LastEditTime: 2023-1-8 22:06:12
'''

# 导包
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

# 超参数设置
BATCH_SIZE = 64  # 每一批所处理的数据集
LR_G = 0.0001  # generator的学习效率
LR_D = 0.0001  # discriminator的学习效率
N_IDEAS = 5  # generator的学习灵感
ART_COMPONENTS = 15  # generator创作的点数
# 一次处理中所产生的点数，二维矩阵
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS)
                          for _ in range(BATCH_SIZE)])  # shape(64,15)


# 著名画家的画
def artist_works():
    # 从[1,2)的正态分布中随机选取64个数字，并在第二维度新加一个维度
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    # 生成这些点的纵坐标
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)  # shape(64,15)
    # 将其转变为tensor数据
    paintings = torch.from_numpy(paintings).float()
    return paintings


# Generator network
# generator将自己的灵感变成15个点
G = torch.nn.Sequential(
    torch.nn.Linear(N_IDEAS, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, ART_COMPONENTS)
)

# Discriminator network
# 对画作进行鉴别，输出一个它判断该画作是否为著名画家画作的概率值，sigmoid()用于生成一个概率值
D = torch.nn.Sequential(
    torch.nn.Linear(ART_COMPONENTS, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid()
)

# 当在class中才需要新建实例
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

# training
if __name__ == '__main__':
    for step in range(10000):
        # 著名画家的画作
        artist_paintings = artist_works()

        # 用于随机生成generator的灵感
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)  # shape(64,5)

        # 根据生成的灵感来作画
        G_paintings = G(G_ideas)

        # discriminator判断这些画作【G自己的灵感画作】来自著名画家的概率为多少
        prob_G = D(G_paintings)

        # 这种概率要越低越好，因为它是永远是在模仿
        G_loss = torch.mean(torch.log(1. - prob_G))

        # 优化
        opt_G.zero_grad()  # 将模型中的梯度清零
        G_loss.backward()  # 求目标函数的梯度
        opt_G.step()  # 梯度下降，更新G的参数

        # discriminator判断这些画作【著名画家画作】来自著名画家的概率为多少，希望越高越好
        prob_a = D(artist_paintings)

        # G_paintings的梯度不会更新
        prob_G = D(G_paintings.detach())  # 锁住G的参数不求导

        # 我们是希望它越大越好，但是torch中只有减小误差才会提升
        D_loss = -torch.mean(torch.log(prob_a) + torch.log(1. - prob_G))
        opt_D.zero_grad()
        # retain_graph是为了再次使用计算图纸
        D_loss.backward(retain_graph=True)
        opt_D.step()

        if step % 150 == 0:
            print(step)
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_paintings.data.numpy(
            )[0], c='r', lw=3, label='Generated painting')
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0],
                                                   2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0],
                                                   2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy = %.2f(0.5 for D to converage)' %
                     prob_a.data.numpy().mean(), fontdict={'size': 15, 'color': 'blue'})
            plt.text(-.5, 2, 'D score = %.2f(-1.38 for G to converage)' % -
            D_loss.data.numpy(), fontdict={'size': 15, 'color': 'blue'})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=10)
            plt.draw()
            plt.pause(0.01)
    plt.ioff()
    plt.savefig()
    plt.show()






