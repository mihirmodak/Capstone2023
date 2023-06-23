
import argparse
import numpy as np
from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import utils
from models import Model

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, train_loader:DataLoader, val_loader:DataLoader, epochs:int, criterion, optimizer):

    # Initialize the TensorBoard writer
    writer = SummaryWriter()

    # # Initialize the EarlyStopping function
    # early_stopper = EarlyStopper(patience=3, min_delta=100)

    # Iterate ovver a number of epochs
    for epoch in range(epochs):

        # Reset the loss at each epoch
        running_loss = 0.0

        # For every epoch, go through multiple batches
        for i, (inputs_batch, labels_batch) in enumerate(train_loader):
         
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs_batch = model(inputs_batch)

            # In the case of a single output node, we want to reshape the labels array
            # from a row vector to a column vector
            if labels_batch.shape[-1] != outputs_batch.shape[-1]:
                labels_batch = labels_batch.unsqueeze(1)

            # # Debugging Code
            # print(labels_batch.shape, labels_batch, sep='\n')
            # print(outputs_batch.shape, outputs_batch, sep='\n')
            # print('\n')

            # Compute the training loss
            loss = criterion(outputs_batch, labels_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Remove the gradients, and convert to numpy array
            labels_batch_np = labels_batch.detach().cpu().numpy()
            outputs_batch_np = outputs_batch.detach().cpu().numpy()

            mae = mean_absolute_error(y_true=labels_batch_np, y_pred=outputs_batch_np)
            rmse = np.sqrt( mean_squared_error(y_true=labels_batch_np, y_pred=outputs_batch_np) )
            r2 = r2_score(y_true=labels_batch_np, y_pred=outputs_batch_np)

            # # Log the loss value to plot later
            # writer.add_scalar("Loss/train", loss, epoch)
            # writer.add_scalar("Mean Absolute Error/train", mae, epoch)
            # writer.add_scalar("Root Mean Squared Error/train", rmse, epoch)
            # writer.add_scalar("R2 Value/train", r2, epoch)

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/100} mae:{mae} r2: {r2}")
                running_loss = 0.0 # Recalculate loss for every mini-batch

        # Log the loss value to plot later
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Mean Absolute Error/train", mae, epoch)
        writer.add_scalar("Root Mean Squared Error/train", rmse, epoch)
        writer.add_scalar("R2 Value/train", r2, epoch)

        # Calculate the validation metrics
        with torch.no_grad():
            for val_inputs_batch, val_labels_batch in val_loader:
                val_outputs_batch = model(val_inputs_batch)
                if val_labels_batch.shape[-1] != val_outputs_batch.shape[-1]:
                    val_labels_batch = val_labels_batch.unsqueeze(1)
                val_loss = criterion(val_outputs_batch, val_labels_batch)
                val_labels_batch_np = val_labels_batch.detach().cpu().numpy()
                val_outputs_batch_np = val_outputs_batch.detach().cpu().numpy()

                val_mae = mean_absolute_error(y_true=val_labels_batch_np, y_pred=val_outputs_batch_np)
                val_rmse = np.sqrt( mean_squared_error(y_true=val_labels_batch_np, y_pred=val_outputs_batch_np) )
                val_r2 = r2_score(y_true=val_labels_batch_np, y_pred=val_outputs_batch_np)
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Mean Absolute Error/val", val_mae, epoch)
                writer.add_scalar("Root Mean Squared Error/val", val_rmse, epoch)
                writer.add_scalar("R2 Value/val", val_r2, epoch)

        # if early_stopper.early_stop(running_loss):
        #     print(f"EarlyStopping: Stopped training at epoch {epoch}.")
        #     break

    writer.close()
    print('Finished training')
    return model, optimizer, loss, epoch

def evaluate(model, dataset:TensorDataset, loader:DataLoader, loss_fn:Callable):
    # Evaluate the neural network on the validation set
    total_loss = 0.0
    with torch.no_grad():
        for inputs_batch, labels_batch in loader:

            # Forward pass
            outputs_batch = model(inputs_batch)

            # In the case of a single output node, we want to reshape the labels array
            # from a row vector to a column vector
            if labels_batch.shape[-1] != outputs_batch.shape[-1]:
                labels_batch = labels_batch.unsqueeze(1)

            # Compute the loss
            criterion = loss_fn()
            loss = criterion(outputs_batch, labels_batch)

            # Accumulate the total loss
            total_loss += loss.item() * inputs_batch.shape[0]

    return f"Validation loss: {total_loss / len(dataset)}"

def start(args:dict, loss_fn:Callable = nn.MSELoss, optim:str = "adam"):

    if not isinstance(args, dict):
        args = vars(args)

    # Use the GPU if available, else use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device `{device}`")

    if args['mode'].lower() == "new":
        # Initialize the neural network
        net = utils.name2model(model_name=args['model_name'], device=device)

        # initialize the optimizer
        optimizer = utils.assignOptimizer(net, optim, learning_rate=float(args['learning_rate']))
    
    elif args['mode'].lower() == "continue":
        net, optimizer, loss, epoch = Model.load(optim="adam", device=device)
        net.train()
        optimizer.param_groups[0]['lr'] = float(args['learning_rate'])
    
    # Define the loss function
    criterion = loss_fn()

    num_output_nodes = net(torch.randn(1, 1, 125).to(device)).shape[-1]

    # Ask the user for data
    data_df = utils.getData()

    # Create the training/validation/testing split
    splits = utils.splitData(data_df, is_dual_out= num_output_nodes == 2) # seed = 1, train_frac = 0.75, val_frac = 0.2, shuffle = True,
    (inputs_train, labels_train), (inputs_val, labels_val), (inputs_test, labels_test) = splits

    # Create the PyTorch Datasets and Data Loaders
    train_data, train_loader = utils.buildDataSet(
        inputs = inputs_train, 
        labels = labels_train, 
        shuffle = True, 
        device = device # place the newly created data objects on the appropriate device
    )

    val_data, val_loader = utils.buildDataSet(
        inputs = inputs_val, 
        labels = labels_val, 
        device = device # place the newly created data objects on the appropriate device
    )

    # Train the model
    net, optimizer, loss, epoch = train(
        model = net, 
        train_loader = train_loader,
        val_loader = val_loader,
        epochs = int(args['epochs']), 
        criterion=criterion,
        optimizer=optimizer
    )

    # Calculate the validation loss
    print( evaluate(model = net, dataset = val_data, loader = val_loader, loss_fn = nn.MSELoss) )

    # Save the model to a file
    net.save(filename=None, optimizer=optimizer, loss=loss, epoch=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["new", "continue"], help="Whether to train a new model or continue training a previously saved model")
    parser.add_argument("model_name", help="The name of the class in models.py to initialize and train", default="H4O1")
    parser.add_argument("--epochs", help="The number of epochs to run training", default=50)
    parser.add_argument("-lr", "--learning_rate", help="The learning rate to use in training", default=0.001)
    parser.add_argument("--dropout", help="The proportion of neurons to drop temporarily during every epoch", required=False, default=0.5)
    args = parser.parse_args()
    args = vars(args)

    if args['mode'].lower() in ["new", "continue"]:
        start(args)
    else:
        print('Invalid mode selected, please try again with either "new" or "continue"')