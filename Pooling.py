import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch



class PoolingLayer() :
    '''
    Takes as input the features extracted on a per-patch level by a given foundation models and crated a  
    representation of the WSI by combining the patch features.
    -- 'Avg' performs average pooling across all patches.
    -- 'Max' performs max pooling across all patches.
    '''
    def __init__(self, type_of_pool='Avg'):
        super(PoolingLayer, self).__init__()
        self.type_of_pool = type_of_pool
        
    def forward(self, x):
        if self.type_of_pool == 'Avg':
            return torch.mean(x, dim=0)  
        elif self.type_of_pool == 'Max':
            return torch.max(x, dim=0)[0]
        else:
            raise ValueError("Invalid pooling type. Choose 'Avg' or 'Max'.")
        
        
if __name__ == '__main__':
    
    
    dummytensor = torch.randn(3,5)
    print(dummytensor)
    print(torch.mean(dummytensor, dim =0))
    print(torch.max(dummytensor, dim=0)[0])