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


class MultiTaskModel(TorchModel):
    
    # Model architecture
    def __init__(self, input_shape: int = 1576*24, **kwargs):
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(0.9)

        self.conv1 = nn.Conv1d(24,32,kernel_size=40*4)
        self.max_pooling1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32,64,30*4)
        self.conv3 = nn.Conv1d(64,128,30*4)
        self.max_pooling2 = nn.MaxPool1d(1)
        self.conv4 = nn.Conv1d(128,256,20*3)
        self.conv5 = nn.Conv1d(256,256,20*3)
        self.max_pooling3 = nn.MaxPool1d(1)
        self.conv6 = nn.Conv1d(256,256,20*3)
        self.max_pooling4 = nn.MaxPool1d(2*1) 

        self.conv7 = nn.Conv1d(256,128,20)
        self.max_pooling5 = nn.MaxPool1d(2)
        self.conv8 = nn.Conv1d(128,128,20*2)
        self.max_pooling6 = nn.MaxPool1d(2)
        


        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024,1024)
    
        self.output_atb_class = nn.Sequential(nn.Linear(1024, 25), nn.Softmax())
        self.output_mechanism = nn.Sequential(nn.Linear(1024,5), nn.Softmax())
        self.output_mobility = nn.Sequential(nn.Linear(1024,2), nn.Softmax())

        self.cnn = nn.Sequential(
            self.conv1, nn.ReLU(), self.max_pooling1, nn.ReLU(), self.conv2, nn.ReLU(),
            self.conv3, nn.ReLU(), self.max_pooling2, nn.ReLU(), self.conv4, nn.ReLU(), 
            self.conv5,  nn.ReLU(), self.max_pooling3, nn.ReLU(), self.conv6, nn.ReLU(), self.max_pooling4, nn.ReLU(),
            self.conv7,  nn.ReLU(), self.max_pooling5, nn.ReLU(),self.conv8,  nn.ReLU(), self.max_pooling6, nn.ReLU(),
            nn.Flatten())
            
        
        self.hidden = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

        self.loss = F.cross_entropy
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    # Forward pass
    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.dropout(x)
        conv = self.cnn(x)
        hidd = self.hidden(conv)
        atb_class = self.output_atb_class(hidd)
        mechanism = self.output_mechanism(hidd)
        #mobility = self.output_mobility(hidd)

        return atb_class, mechanism #, mobility

    # Train model
    def training_step(self, trainloader, nb_epochs, atb_coeff, mech_coeff):
        for epoch in range(nb_epochs):

            for (idx, batch) in enumerate(trainloader):
                x = batch["sequence"]
                atb_class_true = batch['atb_class']
                mech_true = batch['mech']


                self.optimizer.zero_grad()

                #atb_class_pred, mechanism_pred, mobility_pred = self.forward(inputs)
                
                atb_class_pred, mechanism_pred = self.forward(x)
                
                #print(np.argmax(atb_class_pred.detach().numpy(), axis = 1))
                #print(y[:,0])
                
                loss_atb_class = self.loss(atb_class_pred, atb_class_true.squeeze(1))
                loss_mechanisms = self.loss(mechanism_pred, mech_true.squeeze(1))

                loss = loss_atb_class*atb_coeff + loss_mechanisms*mech_coeff

                loss.backward()
                
                self.optimizer.step()

    # Save model
    def save_model(self, path: str):
        torch.save({'model': self.state_dict()}, path)   