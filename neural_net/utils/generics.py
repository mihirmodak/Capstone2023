import os
from pathlib import Path
import re
from datetime import datetime
import torch
import torch.optim as optim
import models

try:
    from tkinter.filedialog import askdirectory, askopenfilename
    has_tkinter = True
except:
    has_tkinter = False

def assignOptimizer(model, optimizer_code:str, learning_rate:float=0.001):
    match optimizer_code.lower():
        case "adam":
            return optim.Adam(model.parameters(), lr=learning_rate)
        case "sgd":
            return optim.SGD(model.parameters(), lr = learning_rate)
        case "asgd":
            return optim.ASGD(model.parameters(), lr = learning_rate)

def saveModelToFile(self, filename:Path = None, optimizer = None, loss = None, epoch = None):
    if filename is None:
        filename = _generate_model_save_name(self.__class__.__name__)

    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.absolute()

    # Save the entire model
    torch.save(self, filename)

    # If the necessary arguments are provided, then save the checkpoint for further training
    if all([optimizer, loss, epoch]): # None evaluates to False, so this is a check that all 3 exist
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 
            filename.parent / f"{filename.stem}_checkpoint.pt"
        )
    else:
        print(optimizer)
        print(loss)
        print(epoch)
        print("All 3 optional arguments (optimizer, loss, and epoch) must be provided to save a model checkpoint")

def loadModelFromFile(optim:str=None, learning_rate:float=0.001, device=torch.device("cpu")):
    
    # Get the filename
    filename = askopenfilename(
        title="Select PyTorch model file", 
        filetypes=[("All Files", "*"), ("PyTorch", "*.pt, *.pth")],
        initialdir=os.getcwd()
    )

    # Convert to pathlib for consistent file path formatting
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.absolute()
    assert filename.exists(), f"The given filepath does not point to a file.\n{filename}"

    # # Extract the model type/name from the filename
    # model_name = filename.stem.split('_')[1]

    model_name = path2name(filename)

    # Make sure the model name exists and get the class
    assert model_name in dir(models), "The chosen model is not valid. Please choose a model class that exists in models.py"
    model_class = getattr(models, model_name)

    # Initialize a new untrained model
    model = model_class().to(device)

    # Different loading processes depending on whether the file is a checkpoint or not
    if "checkpoint" in filename.stem and optim is not None:
        optimizer = assignOptimizer(model, optim, learning_rate)

        # Load the checkpoint information into a dict
        checkpoint = torch.load(filename)

        # Assign the model, optimizer, and parameters their saved values
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    else:
        
        # Load the model
        model = torch.load(filename, map_location=device)

        # Set the other values to None, since they are not necessary
        optimizer, epoch, loss = None, None, None
    
    # Print out the model structure so that the user can check
    print(model.eval())
    
    # Return all
    return model, optimizer, epoch, loss

def name2model(model_name, device):
    # Initialize the neural network
    assert model_name in dir(models), "The chosen model is not valid. Please choose a model class that exists in models.py"
    model_class = getattr(models, model_name)
    net = model_class().to(device)
    return net

def path2name(model_path):
    # Convert to pathlib for consistent file path formatting
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    model_path = model_path.absolute()
    assert model_path.exists(), f"The given filepath does not point to a file.\n{model_path}"

    # Extract the model type/name from the filename
    model_name = model_path.stem.split('_')[1]

    return model_name

def _generate_model_save_name(model_name):
    # Save the model to a file

    # Get the directory to save the model
    if has_tkinter:
        model_save_path = askdirectory(
            title="Save PyTorch model as", 
        )
    else:
        model_save_path = input("\nEnter an absolute directory path: ")

        # Reset the value of p so that the error message in the assert statement doesn't arbitrarily point to the last file inputted
        assert model_save_path.exists(), f"That directory path is invalid, please create a new directory at the location or provide a valid directory"

    model_save_path = Path(model_save_path).absolute()

    # Pattern for any models trained in the past
    filename_pattern = r"\d{8}_\w+_(\d+).pt[h]?"
    r = re.compile(filename_pattern)

    # Search through all files and subdirectories in model_save_path
    files = list(model_save_path.rglob("*"))

    # Convert all the pathlib.Paths to strings
    files_str = list(map(str, files))

    # Get only the file paths that match the pattern
    prev_model_files = list(filter(r.search, files_str))

    # Extract the numbers at the end of the filename
    matched_files = []
    for fname in prev_model_files:
        try:
            matched_files = [int(fname.split('.')[0].split('_')[-1])]
        except ValueError: # except if you cannot convert to int
            continue
    
    # If at least 1 of the files has a number at the end of the filename
    if len(matched_files) > 0:
        prev_model_count = max(matched_files)
    else:
        prev_model_count = 0

    model_save_path = model_save_path / \
        f"{datetime.today().strftime(r'%Y%m%d')}_{model_name}_{prev_model_count+1}.pt"
    
    return model_save_path