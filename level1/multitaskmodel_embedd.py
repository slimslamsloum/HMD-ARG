import enum

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepchain.models.torch_model import TorchModel
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class MultiTaskModelEmbedd(TorchModel):

    # Model architecture
    def __init__(self, input_shape: int = 1280, **kwargs):
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(0.9)

        self.fc1 = nn.Linear(1280, 1024)
        self.fc2 = nn.Linear(1024,1024)
    
        self.output_atb_class = nn.Sequential(nn.Linear(1024, 25), nn.Softmax())
        self.output_mechanism = nn.Sequential(nn.Linear(1024,5), nn.Softmax())
        
        self.hidden = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

        self.loss = F.cross_entropy
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    # Forward pass
    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.dropout(x)
        hidd = self.hidden(x)
        atb_class = self.output_atb_class(hidd)
        mechanism = self.output_mechanism(hidd)

        return atb_class, mechanism #, mobility

    # Train model
    def training_step(self, trainloader, nb_epochs, atb_coeff, mech_coeff):
        for epoch in range(nb_epochs):

            for (idx, batch) in enumerate(trainloader):
                x = batch[:,:-2]
                y = batch[:,-2:]

                self.optimizer.zero_grad()

                #atb_class_pred, mechanism_pred, mobility_pred = self.forward(inputs)
                atb_class_pred, mechanism_pred = self.forward(x.unsqueeze(1))
                
                #print(np.argmax(atb_class_pred.detach().numpy(), axis = 1))
                #print(y[:,0])
                
                loss_atb_class = self.loss(atb_class_pred.squeeze(1), y[:,0].type(torch.LongTensor))
                loss_mechanisms = self.loss(mechanism_pred.squeeze(1), y[:,1].type(torch.LongTensor))

                loss = loss_atb_class*atb_coeff + loss_mechanisms*mech_coeff

                loss.backward()
                
                self.optimizer.step()

    # Save model
    def save_model(self, path: str):
        torch.save({'model': self.state_dict()}, path)   