import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepchain.models.torch_model import TorchModel
from scaling_weights import weights_atb, weights_mech
from torch import nn


class MultiTaskModelEmbedd(TorchModel):

    # Model architecture
    def __init__(self, input_shape: int = 1280, **kwargs):
        super().__init__(**kwargs)

        # Dropout rate
        self.dropout = nn.Dropout(0.9)

        # Hidden layers
        self.fc1 = nn.Linear(1280, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Output layers for multi task
        self.output_atb_class = nn.Sequential(nn.Linear(1024, 22), nn.Softmax())
        self.output_mechanism = nn.Sequential(nn.Linear(1024, 5), nn.Softmax())

        # Hidden layer
        self.hidden = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

        # Loss and optimizer
        self.loss_atb = torch.nn.CrossEntropyLoss(weight=weights_atb)
        self.loss_mech = torch.nn.CrossEntropyLoss(weights_mech)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    # Forward pass
    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.dropout(x)
        hidd = self.hidden(x)
        atb_class = self.output_atb_class(hidd)
        mechanism = self.output_mechanism(hidd)

        return atb_class, mechanism

    # Train model
    def training_step(self, trainloader, nb_epochs, atb_coeff, mech_coeff):
        for epoch in range(nb_epochs):

            for (idx, batch) in enumerate(trainloader):
                # Separate samples and labels
                x = batch[:, :-2]
                y = batch[:, -2:]

                # Zero out gradients
                self.optimizer.zero_grad()

                # Forward pass
                atb_class_pred, mechanism_pred = self.forward(x.unsqueeze(1))

                # Compute separate losses
                loss_atb_class = self.loss_atb(
                    atb_class_pred.squeeze(1), y[:, 0].type(torch.LongTensor)
                )
                loss_mechanisms = self.loss_mech(
                    mechanism_pred.squeeze(1), y[:, 1].type(torch.LongTensor)
                )

                # Compute overall loss
                loss = loss_atb_class * atb_coeff + loss_mechanisms * mech_coeff

                # Backpropagation
                loss.backward()

                # Adjust weights
                self.optimizer.step()

    # Save model
    def save_model(self, path: str):
        torch.save({"model": self.state_dict()}, path)
