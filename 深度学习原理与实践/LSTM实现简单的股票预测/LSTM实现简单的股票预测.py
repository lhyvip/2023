'''
Description描述: LSTM实现简单的股票预测,有前几天的股票价格来预测明天的价格
Autor作者: lhy
Date日期: 2023-1-15 22:06:12
LastEditTime: 2023-1-17 18:24:05
'''
import numpy as np
import pandas as pd

#1.读入数据
root="D:\\2023\project_data\LSTM--master\\rlData.csv"
data=pd.read_csv(filepath_or_buffer=root)
data=data.sort_values("date",ascending=True,na_position="last")#按照日期升序排列
#date日期，open开盘价，high最高价，low最低价，close收盘价，volume成交量

#2.画图
import matplotlib.pyplot as plt
import seaborn as sns#基于matplotlib的二次封装画图库
plt.rcParams['font.family'] = 'SimHei'#绘图正常显示中文
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号#有中文出现的情况，
# 需要u'内容'
#sns.set_style("darkgrid")#会导致无法出现中文
plt.figure(figsize=(15,9))
plt.plot(data[["close"]])
plt.xticks(range(0,data.shape[0],20),data["date"].loc[::20],rotation=45)
plt.title(u'邮票价格stock price',fontsize=18)
plt.xlabel(u"date日期",fontsize=18)
plt.ylabel(u"close收盘价price",fontsize=18)
plt.show()

#3.选取我们需要的特征
print("------------3.选取我们需要的特征---------")
price=data[["close"]]
print(price.info())
print(price.head())
#4.数据归一化  缩放到（-1,1）
print("----------4.数据归一化  缩放到（-1,1）----------")
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(-1,1))#范围在-1---1注意：MinMaxScaler只能对ndim为2的向量进行操作，
# ndim为1的可以通过reshape(1,-1)改为2维的向量
price['close']=scaler.fit_transform(price['close'].values.reshape(-1,1))

#5.数据集的制作 划分数据集
def split_data(stock,lookback):
    #lookback,表示滑动窗口大小
    data_raw=stock.to_numpy()#dataframe转换为numpy
    data=[]

    for index in range(len(data_raw)-lookback):
        data.append(data_raw[index:index+lookback])
    data=np.array(data)
    print("data=",data.shape)#打印划分好的滑动窗口
    test_set_size=int(np.round(0.2*data.shape[0]))#round()四色五入取整
    print("test_set_size=",test_set_size)
    # 取全部数据的百分之20作为训练样本数
    train_set_size=data.shape[0]-test_set_size#剩下的作为训练集
    x_train=data[:train_set_size,:-1,:]#(样本数，滑动窗口，1) 训练特征
    y_train=data[:train_set_size,-1,:]#标签
    x_test=data[train_set_size:,:-1,:]
    y_test=data[train_set_size:,-1,:]

    return [x_train,y_train,x_test,y_test]

x_train,y_train,x_test,y_test=split_data(price,lookback=20)

#6构建模型LSTM
#注意pytorch的nn.LSTM input shape=(seq_length, batch_size, input_size)前两个变量的位置由batch_frist=false/true 控制
#输出 output.shape=(seq_length,batch_size,hidden_size*滑动窗口大小)
#seq_length=滑动窗口大小
import torch
import torch.nn as nn
import torch.optim as optim
x_train=torch.from_numpy(x_train).type(torch.Tensor)#加上type(torch.tensor)可以是数据Double转化为float，
# 就是把数据变得符合torch了
x_test=torch.from_numpy(x_test).type(torch.Tensor)
y_train=torch.from_numpy(y_train).type(torch.Tensor)
y_test=torch.from_numpy(y_test).type(torch.Tensor)
input_dim=1#输入的是一维数字预测第二天的股票所以是维度是一维
hidden_dim=32#隐藏层维度
num_layers=2#层数
out_put_dim=1#输出维度为1
num_epoch=400#训练次数

class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)#（batch,seq_length,inputsize）
        self.fc=nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        #x.shape= torch.Size([186, 19, 1])
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()#初始化,h0(num_layer,batch,hidden)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()#初始化c0(num_layer,batch,hidden)
        # h0.shape= torch.Size([2, 186, 32])
        # c0.shape= torch.Size([2, 186, 32])
        out,(hn,cn)=self.lstm(x,(h0.detach(),c0.detach()))#out(batch,seq_length,hidden_size)
        # out.shape= torch.Size([186, 19, 32])
        out=self.fc(out[:,-1,:])#取每一个批次的最后一个输出为最终输出（）
        # out1.shape= torch.Size([186, 1])
        return out
model=LSTM(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=out_put_dim,num_layers=num_layers)
criterion=torch.nn.MSELoss()#均方损失函数
optimizer=optim.Adam(model.parameters(),lr=0.01)


#7.模型的训练
import time
start_time=time.time()
lstmloss=[]
for t in range(num_epoch):
    y_train_pred=model(x_train)
    loss=criterion(y_train_pred,y_train)
    if t%20==0:
       print("epoch=",t,"损失值=",loss.item())
    lstmloss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
train_time=time.time()-start_time
print("train_time=",train_time)

#8.loss损失值可视化
plt.figure(figsize=(10,10))
plt.plot(lstmloss,color="red")
plt.ylabel("loss值")
plt.xlabel("epoch轮次值")
plt.title("损失值变化曲线")
plt.show()

#9.模型验证
print(x_test.shape)
print(y_test.shape)
y_test_pred=model(x_test)
print(y_test_pred.shape)
plt.figure(figsize=(10,10))
plt.plot(y_test_pred.detach().numpy(),color="red",label="预测值")
plt.plot(y_test.detach().numpy(),color="yellow",label="真实值")
plt.ylabel("股票值")
plt.xlabel("x")
plt.title("股票值")
plt.legend()
plt.show()

















