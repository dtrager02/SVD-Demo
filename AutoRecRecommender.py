#sbatch -N1 -n1 --gpus=1 --mem-per-gpu=8192 --ntasks=1 --cpus-per-task=16  --constraint=g start.sub
#sacct  --format="JobID,Elapsed,CPUTime,MaxRSS,AveRSS"
#tail -f slurm-146258.out

"""
Notes:
The final model should incorporate a hybrid of MF output and content-user matching
The proportions of these two metrics is determined by how many items the user has rated
the content user matching system will include:
1. Genres of items vs. user genres
2. release dates of items vs typical "era" of user
3. popularity of user-rated items (how niche the user is)
"""
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
from AutoRecDataPrep import MovieLensTrainDataloader

def load_data(path:str):
    a = pd.read_csv(path).astype("int32").to_numpy()[:,:3]
    return torch.from_numpy(a)

def shuffle(data:torch.Tensor):
    temp = data.numpy()
    np.random.shuffle(temp)
    return torch.from_numpy(temp)

def train_test_split(data:torch.tensor,ratio = .1):
    data = shuffle(data)
    return data[:int(data.size()[0]*ratio)],data[int(data.size()[0]*ratio):]

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
    def forward(self, row):
        u,i = row[0],row[1]
        out = self.P[u].dot(self.Q[i])+self.b_u[u]+self.b_i[i]+self.b
        return out
        
        
    def loss(self,row:torch.Tensor,yhat:torch.Tensor):
        u,i,r = row[0],row[1],row[2]
        return (yhat-r)**2+self.beta1*(torch.linalg.norm(self.P[u])**4+torch.linalg.norm(self.Q[i])**4+\
            self.b_i[i]**2 + self.b_u[u]**2)
    
        
    # train the model
    def fit(self,data:torch.Tensor,test_data:torch.Tensor,lr=.005):
        # define the optimization
        optimizer = Adam(self.parameters(), lr=lr)
        save_loss = 1.0
        # enumerate epochs
        for epoch in range(self.epochs):
            #store errors
            predictions,actuals = torch.empty(0,device=self.device),torch.empty(0,device=self.device)
            # enumerate mini batches
            for i in range(data.size()[0]):
                optimizer.zero_grad(set_to_none=True)
                # compute the model output

                yhat = self.forward(data[i])
                # print(yhat)
                loss = self.loss(data[i],yhat)
                
                with torch.no_grad():
                        predictions = torch.hstack((predictions,torch.tensor([yhat])))
                        actuals = torch.hstack((actuals,torch.tensor([data[i,2]])))
                loss.backward()

                optimizer.step()
                
            with torch.no_grad():
                print(f"Predictions: N({torch.mean(predictions)},{torch.std(predictions)})")
                print(f"Labels: N({torch.mean(actuals)},{torch.std(actuals)})")
                train_rmse = torch.mean(torch.square(predictions-actuals))**.5
                
                test_predictions,test_actuals = torch.empty(0,device=self.device),torch.empty(0,device=self.device)
                for j in range(test_data.size()[0]):
                    yhat = self.forward(test_data[j])
                    test_predictions = torch.hstack((test_predictions,torch.tensor([yhat])))
                    test_actuals = torch.hstack((test_actuals,torch.tensor([test_data[j,2]])))
                test_rmse = torch.mean(torch.square(test_predictions-test_actuals))**.5
                save_loss = max((save_loss,test_rmse))
                print(f"Epoch {epoch}/{self.epochs}; Current Train RMSE: {train_rmse}; Current Test RMSE: {test_rmse} Best Test RMSE: {save_loss}")               
        return save_loss
                         
if __name__ == "__main__":
    data = load_data("./ratings.csv")
    print("loaded data")
    train,test = train_test_split(data)
    print("split data")
    model = AutoRec(data[:,0].max()+1,data[:,1].max()+1,epochs=30,hidden_size=60,beta1=.02)
    print("Fitting model")
    model.fit(train,test)