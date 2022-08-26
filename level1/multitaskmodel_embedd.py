import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepchain.models.torch_model import TorchModel
from scaling_weights import weights_atb, weights_mech
from torch import nn
from torchmetrics import Accuracy


class MultiTaskModelEmbedd(pl.LightningModule):

    # Model architecture
    def __init__(self, input_shape: int = 1280, **kwargs):
        super().__init__(**kwargs)

        # Dropout rate
        self.dropout = nn.Dropout(0.9)

        # Hidden layers
        self.fc1 = nn.Linear(1280, 1024)
        self.fc2 = nn.Linear(1024, 128)

        # Output layers for multi task
        self.output_atb_class = nn.Sequential(nn.Linear(128, 22), nn.Softmax())
        self.output_mechanism = nn.Sequential(nn.Linear(128, 5), nn.Softmax())

        # Hidden layer
        self.hidden = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU())

        # Loss and optimizer
        self.loss_atb = torch.nn.CrossEntropyLoss(weight=weights_atb)
        self.loss_mech = torch.nn.CrossEntropyLoss(weight=weights_mech)

    # Forward pass
    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.dropout(x)
        hidd = self.hidden(x)
        atb_class = self.output_atb_class(hidd)
        mechanism = self.output_mechanism(hidd)

        return atb_class, mechanism

    # Train model
    def training_step(self, batch, batch_idx):

                x = batch[:, :-2]
                y = batch[:, -2:]

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
                loss = loss_atb_class * 1 + loss_mechanisms * 0.2

                self.log('training_loss', loss, on_epoch=True)

                return loss

    # Validate model
    def validation_step(self, batch, batch_idx):

                x = batch[:, :-2]
                y = batch[:, -2:]

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
                loss = loss_atb_class * 1 + loss_mechanisms * 0.2

                probas = self.forward(x=x)
                atb_proba = probas[0]
                mech_proba = probas[1]

                pred = []


                for i in range(len(batch)):
                    pred.append(
                    (
                        np.argmax(atb_proba[i].detach().numpy()),
                        np.argmax(mech_proba[i].detach().numpy()),
                    )
                )

                accuracy = Accuracy()
                acc = accuracy(torch.Tensor(pred).int(), y.int())

                self.log('val_loss', loss, on_epoch=True)
                self.log_dict({"val_acc": acc, "val_loss": loss}, on_epoch=True)

                return loss

                
    # Test model
    def test_step(self, batch, batch_idx):

                x = batch[:, :-2]
                y = batch[:, -2:]

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
                loss = loss_atb_class * 1 + loss_mechanisms * 0.2

                probas = self.forward(x=x)
                atb_proba = probas[0]
                mech_proba = probas[1]

                pred = []


                for i in range(len(batch)):
                    pred.append(
                    (
                        np.argmax(atb_proba[i].detach().numpy()),
                        np.argmax(mech_proba[i].detach().numpy()),
                    )
                )

                accuracy = Accuracy()
                acc = accuracy(torch.Tensor(pred).int(), y.int())

                self.log_dict({"test_acc": acc, "test_loss": loss}, on_epoch=True)

                return acc


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    # Save model
    def save_model(self, path: str):
        torch.save({"model": self.state_dict()}, path)
