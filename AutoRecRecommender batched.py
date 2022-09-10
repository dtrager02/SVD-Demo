import os
import torch
from torch import Tensor
# from torch.nn import Linear
# from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import Adam
# from torch.nn import MSELoss
# from torch.nn.init import kaiming_uniform_,zeros_
import torch.nn as nn
import numpy as np
import pandas as pd

def load_data(path:str):
    a = pd.read_csv(path).astype("int32").to_numpy()[:,:3]
    return torch.from_numpy(a)

def shuffle(data:torch.Tensor):
    temp = data.numpy()
    np.random.shuffle(temp)
    return torch.from_numpy(temp)

def train_test_split(data:torch.tensor,ratio = .1):
    data = shuffle(data)
    return data[int(data.size()[0]*ratio):],data[:int(data.size()[0]*ratio)]

class AutoRec(Module):
    def __init__(self, num_users, num_items,epochs=30,hidden_size=256,beta1=.02):
        super(AutoRec,self).__init__()
        self.training = True
        self.device = torch.device("cpu")
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
        
        self.time_layer = torch.zeros(num_items,requires_grad=True,device=self.device)
        
        self.beta1 = beta1
        self.beta2 = self.beta1/2
        
        self.b = 3.2
        self.b_u = nn.Parameter(torch.empty((num_users)).normal_())
        self.b_i = nn.Parameter(torch.empty((num_items)).normal_())
        
        self.P = nn.Parameter(torch.empty((num_users,hidden_size)).normal_())
        self.Q = nn.Parameter(torch.empty((num_items,hidden_size)).normal_())
        # self.y = nn.Parameter(torch.empty((num_items,hidden_size)).normal_())
        
        self.epochs = epochs
        self.k = hidden_size
    def forward(self, rows:torch.Tensor):
        rows = rows.long()
        u,i = rows[:,0],rows[:,1]
        # print(u,i)
        out = torch.sum(self.P[u,:]*self.Q[i,:],dim=1)+self.b_u[u]+self.b_i[i]+self.b
        return out
        
        
    def loss(self,rows:torch.Tensor,yhat:torch.Tensor):
        rows = rows.long()
        u,i,r = rows[:,0],rows[:,1],rows[:,2]
        return (yhat-r)**2+self.beta1*(torch.sum(self.P[u]**2,1)+torch.sum(self.Q[i]**2,1)+\
            self.b_i[i]**2 + self.b_u[u]**2)
    
        
    # train the model
    def fit(self,data:torch.Tensor,test_data:torch.Tensor,lr=.01,batch_size=1024):
        # define the optimization
        optimizer = Adam(self.parameters(), lr=lr)
        save_loss = 1.0
        # enumerate epochs
        for epoch in range(self.epochs):
            #store errors
            predictions,actuals = torch.empty(0,device=self.device),torch.empty(0,device=self.device)
            # enumerate mini batches
            for i in range(0,data.size()[0]-batch_size,batch_size):
                optimizer.zero_grad(set_to_none=True)
                # compute the model output

                yhat = self.forward(data[i:i+batch_size])
                assert yhat.size() == (batch_size,)
                # print(yhat.size(),yhat)
                loss = self.loss(data[i:i+batch_size],yhat)
                # print(loss.size(),loss)
                with torch.no_grad():
                        predictions = torch.hstack((predictions,yhat))
                        actuals = torch.hstack((actuals,data[i:i+batch_size,2]))
                        assert actuals.size() == predictions.size()
                loss.sum().backward()

                optimizer.step()
            
            with torch.no_grad():
                data = shuffle(data)
                print(f"Predictions: N({torch.mean(predictions)},{torch.std(predictions)})")
                print(f"Labels: N({torch.mean(actuals)},{torch.std(actuals)})")
                train_rmse = torch.mean(torch.square(predictions-actuals))**.5
                
                test_predictions,test_actuals = torch.empty(0,device=self.device),torch.empty(0,device=self.device)
                for j in range(0,test_data.size()[0],batch_size):
                    yhat = self.forward(test_data[j:j+batch_size])
                    test_predictions = torch.hstack((test_predictions,yhat))
                    test_actuals = torch.hstack((test_actuals,test_data[j:j+batch_size,2]))
                    assert test_actuals.size() == test_predictions.size()
                test_rmse = torch.mean(torch.square(test_predictions-test_actuals))**.5
                save_loss = min((save_loss,test_rmse))
                print(f"Epoch {epoch}/{self.epochs}; Current Train RMSE: {train_rmse}; Current Test RMSE: {test_rmse} Best Test RMSE: {save_loss}")               
        return save_loss
      
#optional  
import heapq    
def grid_search(betas,sizes):
    heap = []
    for beta in betas:
        for size in sizes:
            model = AutoRec(data[:,0].max()+1,data[:,1].max()+1,epochs=12,hidden_size=size,beta1=beta)
            a = model.fit(train,test)
            heapq.heappush(heap,(a,beta,size))
    return heap
                         
if __name__ == "__main__":
    data = load_data("./ratings_1m.csv")
    print(f"loaded data users:{data[:,0].max()} items:{data[:,1].max()}")
    train,test = train_test_split(data)
    print("split data",train.size(),test.size())
    model = AutoRec(data[:,0].max()+1,data[:,1].max()+1,epochs=15,hidden_size=40,beta1=.06)
    print("Fitting model")
    # heap = grid_search([.01,.03,.05,.07],[30,50,70,100])
    # while len(heap):
    #     print(heapq.heappop(heap))
    model.fit(train,test)