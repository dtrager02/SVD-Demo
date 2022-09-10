import torch
from torch.utils.data import Dataset
from time import perf_counter
import numpy as np
import pandas as pd
from time import perf_counter
class MovieLensTrainDataloader2(Dataset):

    def __init__(self,data,batch_size=4096,test_ratio = .15):
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.testing = False
        self.device = torch.device("cpu")
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
        self.sparse_ratings2,self.sparse_ratings3 = self.get_dataset(data)
    
        # self.sparse_times = sparse.coo_matrix((self.times, (self.users.astype(np.int32), self.items.astype(np.int32))),shape=(self.n_users,self.n_items),dtype=np.float32).tocsr()
        # self.row_order = torch.tensor(self.row_order,device=self.device)
        #Making test data
        # self.output,self.input = torch.zeros((self.batch_size,self.n_users),device=self.device),torch.zeros((self.batch_size,self.n_users),device=self.device)

    def __len__(self):
        return self.n_users
  
    def __getitem__(self, idx,test=False):
        indices = None
        if test:
            indices = self.test_row_order[idx:idx+self.batch_size]
        else:
            indices = self.row_order[idx:idx+self.batch_size]
            
        a = torch.index_select(self.sparse_ratings2,0,indices)
        a = a.to_dense()
        b = torch.index_select(self.sparse_ratings3,0,indices)
        b = b.to_dense()
        return a.to(self.device),b.to(self.device),torch.tensor(indices,device=self.device)
    
    def __iter__(self):
        self.current_index = 0
        self.current_test_index = 0
        return self

    def __next__(self): # Python 2: def next(self)
        if not self.testing:
            if self.current_index < self.__len__()-self.batch_size:
                output = self.__getitem__(self.current_index,test=False)
                self.current_index+= self.batch_size
                return output
            
            
            self.current_index = 0
            self.row_order = self.row_order.numpy()
            np.random.shuffle(self.row_order)
            self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
            raise StopIteration
        else:
            if self.current_test_index < len(self.test_row_order)-self.batch_size:
                output = self.__getitem__(self.current_test_index,test=True)
                self.current_test_index+= self.batch_size
                return output
            self.current_test_index = 0
            raise StopIteration

    def start_testing(self):
        self.testing = True
    
    def stop_testing(self):
        self.testing = False
    
    def get_dataset(self,data):   
        ##########################################
        #FROM VARIABLE
        ##########################################
        # drop_indices = np.random.choice(len(data),size=int(len(data)*ratio),replace=False)
        # test_data = data[drop_indices,:]
#         test_data = np.hstack((test,np.ones((test.shape[0],1))))
#         # data = np.delete(data,drop_indices,axis=0)
#         data = np.hstack((train,np.zeros((train.shape[0],1))))
#         # drop_rows = []
        
#         #remove test users and items not in train
#         # users_set = set(data[:,0])
#         # items_set = set(data[:,1])
#         # for i in range(test_data.shape[0]):
#         #     if test_data[i,0] not in users_set or test_data[i,1] not in items_set:
#         #         drop_rows.append(i)
#         # test_data = np.delete(test_data,drop_rows,axis=0)
#         # stack test data and data
#         data = np.vstack((data,test_data))
        # print("Merged train and test")
        #smuch items to  a 1-n scale
        times = data[:,3].copy() #keep this as a float while the rest are ints
        print("Times stats",times.max(),times.min())
        data = data.astype(np.int32)
        unique = np.unique(data[:,1])
        self.converter = dict(zip(unique,[*range(len(unique))]))
        for i in range(len(data)):
          data[i,1] = self.converter[data[i,1]]
        self.n_users = int(data[:,0].max())+1
        self.n_items = int(data[:,1].max())+1
        print("Rescaled item numbers")
        
        self.row_order = np.arange(self.n_users)
        np.random.shuffle(self.row_order)
        self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
        
        self.test_row_order = self.row_order[:int(len(self.row_order)*self.test_ratio)]
        self.row_order = self.row_order[int(len(self.row_order)*self.test_ratio):]
    
        
        self.current_index = 0
        self.current_test_index = 0
        #adjust ratings to 0-5 scale
        ratings = data[:,2]/2
        
        sparse_ratings2 = torch.sparse_coo_tensor(torch.from_numpy(data[:,[0,1]].T),torch.from_numpy(ratings),size=(self.n_users,self.n_items),device=self.device,dtype=torch.float32).coalesce()
        #this is timestamps
        sparse_ratings3 = torch.sparse_coo_tensor(torch.from_numpy(data[:,[0,1]].T),torch.from_numpy(times),size=(self.n_users,self.n_items),device=self.device,dtype=torch.float32).coalesce()
        
        print("Made sparse matrix")
        return sparse_ratings2,sparse_ratings3