from sklearn import svm
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as var
import torch.optim as optim
from torch.utils.data import TensorDataset # 텐서데이터셋
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader # 데이터로더
from sklearn.metrics import mean_squared_error

torch.manual_seed(1)

data=[]; x=[];y=[];

with open('test.csv','r',encoding='UTF-8') as f:
    rdr=csv.reader(f)
    for i, line in enumerate(rdr):
        data.append(line)
        
for i in range(1,301):
    x.append(float(data[i][0]))
    y.append(float(data[i][1]))

#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#x=input.astype(int)
#x=sc.fit_transform(x)

x=torch.Tensor(x)
x=x.reshape(300,1)
print('x.shape값 :',x.shape)
 
y=torch.Tensor(y)
y=y.reshape(300,1)
print('y.shape값 :',y.shape)

dataset = TensorDataset(x, y)
dataloader=DataLoader(dataset, batch_size=5, shuffle=True)

class linearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,9)
        self.fc2=nn.Linear(9,6)
        self.fc3=nn.Linear(6,1)
        
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


model=linearRegressionModel()
print(list(model.parameters()))

optimizer=torch.optim.SGD(model.parameters(),lr=1e-6)
nb_epochs=200
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=50)

for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
    # H(x) 계산
        prediction=model(x_train)

    # cost 계산
        cost = F.mse_loss(prediction, y_train)
    
    # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1,len(dataloader),
            cost.item()
        ))

new_var=torch.FloatTensor([[4.0]])
pre_y=model(new_var)
print("훈련 후 입력이 4일 때의 예측값 :", pre_y)

new_var=torch.FloatTensor([[36.0]])
pre_y=model(new_var)
print("훈련 후 입력이 36일 때의 예측값 :", pre_y)

new_var=torch.FloatTensor([[53.0]])
pre_y=model(new_var)
print("훈련 후 입력이 53일 때의 예측값 :", pre_y)

new_var=torch.FloatTensor([[72.0]])
pre_y=model(new_var)
print("훈련 후 입력이 72일 때의 예측값 :", pre_y) 

new_var=torch.FloatTensor([[92.0]])
pre_y=model(new_var)
print("훈련 후 입력이 92일 때의 예측값 :", pre_y)

y_pred=model(x_test)
y_pred_de =y_pred.detach().numpy().reshape(-1,1)
y_test_de =y_test.detach().numpy().reshape(-1,1)

mean_squared_error(y_pred_de,y_test_de)
print(mean_squared_error(y_pred_de,y_test_de))

from sklearn.metrics import r2_score
print(r2_score(y_pred_de,y_test_de))

plt.scatter(x_test,y_test_de)
plt.plot(x_test,y_pred_de,color="black")
plt.show()
