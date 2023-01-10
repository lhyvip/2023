'''
Description描述: 基础gan  gan-study 手写数字集实现
Autor作者: lhy
Date日期: 2023-1-9 11:04:12
LastEditTime: 2023-1-10 10:37:05
'''

#1.导入包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
print(torch.__version__)

#2.数据准备
#对数据做归一化对gan而言推荐归一化到（-1,1）
transform=transforms.Compose([
    transforms.ToTensor(),#首先做0-1归一化，然后做channel high witch
    transforms.Normalize(0.5,0.5)#标准化为-1,1  (x-0.5)/0.5=-1/1
])
train_ds=datasets.MNIST(root="D:\\2023\project_data\gan学习强化1",
                                    train=True,
                                    transform=transform,
                                    download=True)
dataloader=DataLoader(dataset=train_ds,batch_size=64,shuffle=True)
imgs,_=next(iter(dataloader))
print(imgs.shape)
#图片大小1*28*28


#3.定义生成器
#输入是长度为100的噪声
#输出1*28*28的图片    可以利用linear全连接层
#linear1: 100---256
#linear2: 256-512
#linear3: 512-28*28
#reshape 28*28---1*28*28
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main=nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,28*28),
            nn.Tanh()#-1/1
        )
    def forward(self,x):#x表示长度为100的噪声输入
        img=self.main(x)
        img=img.view(-1,28,28,1)
        return img


#4.定义判别器
#输入为（1,28,28）的图片 输出为二分类的概率值，输出使用sigmod激活0-1
#BCELoss计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main=nn.Sequential(
            nn.Linear(28*28,512),
            nn.LeakyReLU(),# nn. LeakyReLU f(x) : x>0输出x，
            # 如果x<0，输出a*x a表示一个很小的斜率，比如0.1
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.main(x)
        return x

#5.初始化模型，优化器以及损失计算函数
device="cuda" if torch.cuda.is_available() else "cpu"
print("device=",device)
gen=Generator().to(device)
dis=Discriminator().to(device)
#优化器
d_optim=optim.Adam(dis.parameters(),lr=0.0001)
g_optim=optim.Adam(gen.parameters(),lr=0.0001)
#损失函数
loss_fn=nn.BCELoss()

#6.绘图函数 将每一个批次的图像生成为图像展示出来
def gen_img_plot(model,test_input):
    test_input=model(test_input).detach().cpu()
    prediction=np.squeeze(test_input.numpy())
    fig=plt.figure(figsize=(4,4))
    for i in range(np.size(prediction,0)):
        plt.subplot(4,4,i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis("off")
    plt.show()

test_input=torch.randn(16,100,device=device)

#7.GAN的训练
D_loss=[]
G_loss=[]
#训练循环
for epoch in range(200):
    d_epoch_loss=0
    g_epoch_loss=0
    count=len(dataloader)#len(datasets)返回样本数，len(dataloader)返回批次数
    for step,(img,_) in enumerate(dataloader):
        img=img.to(device)
        size=img.size(0)#获取批次大小，根据大小随机生成同样批次的随机数
        random_noise=torch.randn(size,100,device=device)#作为gengerate的输入

        #8.对判别器的优化
        d_optim.zero_grad()#将上面的梯度归零
        real_output=dis(img)#对判别器输入真实图片，对真实图片的预测结果希望判别器判定为1为true
        d_real_loss=loss_fn(real_output,
                            torch.ones_like(real_output))  #判别器在真实图片产生的损失
        d_real_loss.backward()

        #生成图片上调用判别器
        gen_img=gen(random_noise)
        fake_output=dis(gen_img.detach())#判别器输入生成图片，对生成图片的预测希望是0 false
        #对于生成图像产生的损失，我们的优化目标是判别器，对于生成器的参数不需要去做优化
        #希望将fake_output判别为0来优化判别器，所以在这里要使用detach来
        #截断梯度，detach（）可以得到一个没有梯度的tensor,因此这个梯度不会在传递到generator中了
        #优化目标判别器，对于生成器梯度截断非常重要
        d_fake_loss=loss_fn(fake_output,
                            torch.zeros_like(fake_output)
                           )#判别器在生成器上的损失希望越小越好
        d_fake_loss.backward()
        #总的判别器上的损失为
        d_loss=d_real_loss+d_fake_loss
        d_optim.step()

        #9生成器的损失以及优化
        #站在生成器的角度
        g_optim.zero_grad()
        fake_output = dis(gen_img)#此时对生成器优化，不需要detach（）梯度阶段
        #希望生成器生成的结果被判别器判别为1为真
        g_loss=loss_fn(fake_output
                       ,torch.ones_like(fake_output)
                       )#生成器希望生成的图片结果为1

        g_loss.backward()
        g_optim.step()
        with torch.no_grad():#下面不需要计算梯度
            d_epoch_loss+=d_loss
            g_epoch_loss+=g_loss
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print("epoch:",epoch)
        if epoch%50==0:
            gen_img_plot(gen,test_input)















