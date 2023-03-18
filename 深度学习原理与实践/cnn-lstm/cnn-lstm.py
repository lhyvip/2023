'''
Description描述: 学习cnn-lstm可以看到，该CNN-LSTM由一层一维卷积+LSTM组成。
Autor作者: lhy
Date日期: 2023-3-16 22:06:12
'''
import os

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
from tqdm import tqdm


# 参数设置
seq_length=6 # 时间步长
input_size=3
num_layers=6
hidden_size=64
batch_size=32
n_iters=2000
lr=0.01
out_channel=9
output_size=1
split_ratio=0.1
path='D:\\2023\software\pycharm\workspace\pythonProject\深度学习原理与实践学习\cnn-lstm\data\data.xls'



# 1.文件读取
def get_Data(data_path):
    data=pd.read_excel(data_path)
    data=data.iloc[:,:3]  # 以三个特征作为数据
    label=data.iloc[:,2:] # 取最后一个特征作为标签
    print(data.head())
    print(label.head())
    return data,label


# 2.数据预处理
def normalization(data,label):

    mm_x=MinMaxScaler() # 导入sklearn的预处理容器
    mm_y=MinMaxScaler()
    data=data.values    # 将pd的系列格式转换为np的数组格式
    label=label.values
    print("data.shape=",type(data))
    print("label.shape=",label.shape)
    data=mm_x.fit_transform(data) # 对数据和标签进行归一化等处理
    label=mm_y.fit_transform(label)
    return data,label,mm_y


#3. 时间向量转换
def split_windows(data,seq_length):
    x=[]
    y=[]
    for i in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _x=data[i:(i+seq_length),:]
        _y=data[i+seq_length,-1]
        x.append(_x)
        y.append(_y)
    x,y=np.array(x),np.array(y)
    print('x.shape,y.shape=\n',x.shape,y.shape)
    return x,y


#4. 数据集划分
def split_data(x,y,split_ratio):
    x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=split_ratio)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)  # 加上type(torch.tensor)可以是数据Double转化为float，
    # 就是把数据变得符合torch了
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    print('x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}'
          .format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    return x_train, y_train, x_test, y_test


#5.数据装入
def data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size):

    num_epochs=n_iters/(len(x_train)/batch_size) # n_iters代表一次迭代
    num_epochs=int(num_epochs)
    print("xtrain==",x_train.shape,y_train.shape)
    train_dataset=Data.TensorDataset(x_train,y_train)
    print("train_dataset_len",train_dataset.__len__())
    """
    我们需要将数据划分为许多组特征张量+对应标签的形式，因此最开始我们要将数据的特征张量与标签打包成一个对象。
    深度学习中的特征张量与标签几乎总是分开的，不像机器学习中标签常常出现在特征矩阵的最后一列或第一列。
    合并张量与标签，我们所使用的类是utils.data.TensorDataset，这个功能类似于utils.data.TensorDataset，
    这个功能类似于python中的zip,可以将最外面的维度一致的tensor进行打包，也就是将第一维度一致的tensor进行打包。
    当我们将数据打包成一个对象之后，我们需要使用划分小批量的功能DataLoader。DataLoader是处理训练前专用的功能，
    它可以接受任意形式的数组、张量作为输入，并把他们一次性转换为神经网络可以接入的tensor。
    """
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,drop_last=False) # 加载数据集,使数据集可迭代
    print("train_loader.len=",train_loader.__len__())
    return train_loader,x_test,y_test,num_epochs


# 6.定义模型
import torch.nn as nn
# 定义一个类
class convNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, seq_length):
        super(convNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_directions = 1  # 单向LSTM
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=64, seq_len=3, input_size=3) ---> permute(0, 2, 1)
        # (64, 3, 3)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=2),
            # shape(7,--)  ->(64,3,2)
            nn.ReLU())
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x.shape= torch.Size([32, 6, 3])
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        #x.shape= torch.Size([32, 9, 6])
        x = x.permute(0, 2, 1)
        batch_size, seq_len = x.size()[0], x.size()[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        #h_0= torch.Size([6, 32, 64])
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        # h_0=Variable(torch.zeros(self.num_layers,x.size(0),self.output_size))
        # c_0=Variable(torch.zeros(self.num_layers,x.size(0),self.output_size))# 初始化h_0和c_0

        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.fc(output[:,-1,:])
        return pred



moudle=convNet(input_size,out_channel,hidden_size,num_layers,output_size,batch_size,seq_length)
criterion=torch.nn.MSELoss(reduction="mean")
optimizer=torch.optim.Adam(moudle.parameters(),lr=lr)
print(moudle)



# 7.数据导入
data,label=get_Data(path)
data,label,mm_y=normalization(data,label)
x,y=split_windows(data,seq_length)
x_train,y_train,x_test,y_test=split_data(x,y,split_ratio)
train_loader,x_test,y_test,num_epochs=data_generator(x_train,y_train,x_test,y_test,n_iters,batch_size)


# 8.train
iter=0
for epochs in range(num_epochs):
    loop=tqdm(enumerate(train_loader),total=len(train_loader))
    for i,(batch_x, batch_y) in loop:
         #batch_x.shape torch.Size([32, 6, 3])
         outputs = moudle(batch_x)
         optimizer.zero_grad()   # 将每次传播时的梯度累积清除
         outputs=outputs.squeeze(dim=1)
         loss = criterion(outputs,batch_y) # 计算损失
         loss.backward() # 反向传播
         optimizer.step()
         loop.set_description(f'epoch:[{epochs}/{num_epochs}]')
         loop.set_postfix(loss=loss.item())
         iter+=1



#9.0验证一下
def result(x_test, y_test):
    print("x_test=",x_test.shape)
    print("y_test=",y_test.shape)
    train_predict = moudle(x_test)
    data_predict = mm_y.inverse_transform(train_predict.detach().numpy().reshape(-1,1))
    y_data_plot = mm_y.inverse_transform(y_test.detach().numpy().reshape(-1,1))

    plt.plot(y_data_plot)
    plt.plot(data_predict)
    plt.legend(('real', 'predict'), fontsize='15')
    plt.show()
    print("MAE=",mean_absolute_error(y_data_plot, data_predict))
    print("RMSE=",np.sqrt(mean_squared_error(y_data_plot, data_predict)))

result(x_test,y_test)




