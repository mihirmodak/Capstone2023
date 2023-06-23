import utils
import pathlib
import torch
import torch.nn as nn

# ---------- CLASSES ----------

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename:pathlib.Path = None, optimizer = None, loss = None, epoch = None):
        utils.saveModelToFile(self, filename, optimizer, loss, epoch)

    @classmethod
    def load(cls,  optim:str=None, learning_rate:float=0.001, device:torch.device=torch.device("cpu")):
        return utils.loadModelFromFile(optim, learning_rate, device=device)


class H4O1(Model):
    def __init__(self, dropout_frac:float=0.5):
        super(H4O1, self).__init__()
        self.fc1 = nn.Linear(125, 125)
        self.fc2 = nn.Linear(125, 125)
        self.fc3 = nn.Linear(125, 125)
        self.fc4 = nn.Linear(125, 125)
        self.fc5 = nn.Linear(125, 1)

        # Define proportion or neurons to dropout
        assert dropout_frac < 1, r"You cannot drop 100% of the nodes in the model. Enter a dropout fraction less than 1."
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class FullyConnectedRegressor(H4O1):
    def __init__(self, dropout_frac:float=0.5):
        super().__init__(dropout_frac)  