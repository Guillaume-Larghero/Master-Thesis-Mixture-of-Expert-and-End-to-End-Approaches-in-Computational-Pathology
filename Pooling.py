import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

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
            return torch.mean(x, dim=0)  # Average pooling
        elif self.type_of_pool == 'Max':
            return torch.max(x, dim=0)[0]  # Max pooling
        else:
            raise ValueError("Invalid pooling type. Choose 'Avg' or 'Max'.")
        