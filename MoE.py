import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class Expert(nn.Module):
    '''
    Define The Expert network that sits on top of the pre-trained foundation model (FM). This Network will use
    the features extracted by the FM and predict target of interest based on those extracted features. We have
    to have 1 expert per FM.
    '''
    def __init__(self, input_dim : int, hidden_dim_fc1 : int, hidden_dim_fc2 : int, output_dim : int, dropout_prob : float):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_fc1)
        self.fc2 = nn.Linear(hidden_dim_fc1, hidden_dim_fc2)
        self.fc3 = nn.Linear(hidden_dim_fc2,output_dim)
        #self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x
        
    
class GatingNetwork(nn.Module):
    '''
    The GatingNetwork (Router) computes for a given input, how much each Expert should contribute to the final 
    prediction.
    '''
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
    

class MixtureOfExperts(nn.Module):
    '''
    Implementation of the Mixture of Expert model
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        gate_output = self.gate(x)
        expert_outputs = [expert(x) for expert in self.experts]
        weighted_expert_outputs = [gate_output[:, i:i+1] * expert_output for i, expert_output in enumerate(expert_outputs)]
        final_output = torch.sum(torch.stack(weighted_expert_outputs, dim=0), dim=0)
        return final_output
    
    
    
def train_MoEmodel(model, train_loader, optimizer, final_loss_fn, num_epochs):
    '''
    Function to train the mixture of experts model.
    
    Input:
    - model: The MixtureOfExperts model instance
    - train_loader: DataLoader for training data
    - optimizer: Optimizer for training the model
    - criterion: Loss function
    - num_epochs: Number of epochs for training
    '''
    model.train() 
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            expert_weights, expert_logits = model(inputs)
            loss = final_loss_fn(expert_weights, expert_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    
if __name__ == '__main__' :
    
    Dummy_tensor = torch.randn((3,3,100))
    print(Dummy_tensor)