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


class FullyConnectedRegressor(Model):
    def __init__(self, dropout_frac:float=0.5):
        super(FullyConnectedRegressor, self).__init__()
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


class LargeRegressor(Model):
    def __init__(self):
        super(LargeRegressor, self).__init__()
        self.fc1 = nn.Linear(125, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class DualOutputRegressor(Model):
    def __init__(self):
        super(DualOutputRegressor, self).__init__()
        self.fc1 = nn.Linear(125, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
   

class DualOutDropoutRegressor(DualOutputRegressor):
    def __init__(self, dropout_frac:float=0.5):
        super().__init__()

        # Define proportion or neurons to dropout
        assert dropout_frac < 1, r"You cannot drop 100% of the nodes in the model. Enter a dropout fraction less than 1."
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class LargeDualOutRegressor(LargeRegressor):
    def __init__(self):
        super().__init__()
        self.fc6 = nn.Linear(16, 2)

    # def forward(self, x): Not defined, using the one from LargeRegressor


class LargeDualOutDropoutRegressor(LargeRegressor):
    def __init__(self, dropout_frac:float=0.5):
        super().__init__()

        # Define proportion or neurons to dropout
        assert dropout_frac < 1, r"You cannot drop 100% of the nodes in the model. Enter a dropout fraction less than 1."
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x


class VeryLargeRegressor(Model):
    def __init__(self, num_hidden_nodes:int = 256):
        super(VeryLargeRegressor, self).__init__()
        self.fc1 = nn.Linear(125, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.fc3 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.fc4 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.fc5 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.fc6 = nn.Linear(num_hidden_nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class VeryLargeDropoutRegressor(VeryLargeRegressor):
    def __init__(self, dropout_frac:float=0.5):
        super().__init__()

        # Define proportion or neurons to dropout
        assert dropout_frac < 1, r"You cannot drop 100% of the nodes in the model. Enter a dropout fraction less than 1."
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x


class VeryLargeDualOutRegressor(VeryLargeRegressor):
    def __init__(self, num_hidden_nodes: int = 256):
        super().__init__(num_hidden_nodes)
        self.fc6 = nn.Linear(num_hidden_nodes, 2)
    
    # def forward(self, x): # Use the same forward function as the parent


class VeryLargeDualOutDropoutRegressor(VeryLargeDualOutRegressor):
    def __init__(self, dropout_frac:float=0.5):
        super().__init__()

        # Define proportion or neurons to dropout
        assert dropout_frac < 1, r"You cannot drop 100% of the nodes in the model. Enter a dropout fraction less than 1."
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x