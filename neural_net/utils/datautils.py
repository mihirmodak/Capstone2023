import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from pathlib import Path
try:
    from tkinter.filedialog import askopenfilenames
    has_tkinter = True
except:
    has_tkinter = False

def loadCSV(filepath:os.PathLike):

    assert os.path.exists(filepath), "This file does not exist. Please enter a valid data file path."

    # Read in the CSV file
    df = pd.read_csv(filepath) # ./data/2023-03-09-fructose-glucose-water-signals.csv

    # Rename the first column to index, then set that as the index
    df.rename(columns={"Unnamed: 0":"index"}, inplace=True)
    df.set_index("index", inplace=True)

    return df

def getData():

    # Get the filenames
    if has_tkinter:
        data_paths = askopenfilenames(title = "Select data file", defaultextension = ".csv", initialdir=os.getcwd())
    else:
        keep_asking = True
        data_paths = []
        while keep_asking:
            p = Path( input("\nEnter an absolute file path: ") ).absolute()
            data_paths.append(p)
            keep_asking = input("Add another path? [y/n]".lower()[0]) == 'y'

        # Reset the value of p so that the error message in the assert statement doesn't arbitrarily point to the last file inputted
        p = None
        assert all( [p.exists() for p in data_paths] ), f"At least one of the paths\n{p}\nwas invalid, please make sure all the paths point to files that exist"

    # Load the Data
    data_dfs = []
    for f in data_paths:
        d = loadCSV(f)
        data_dfs.append(d)

    metadata = None
    data = pd.concat(data_dfs)

    # return data, metadata
    return data

def cleanData(df:pd.DataFrame):

    # Regular expression pattern to match decimal column names
    pattern = re.compile(r"^\d+\.?\d*$")

    # Filter DataFrame columns by matching pattern
    decimal_cols = [col for col in df.columns if pattern.match(col)]

    # Add the fructose and glucose to the selected cols
    selected_cols = ["fructose_mgdl", "glucose_mgdl"]
    selected_cols.extend(decimal_cols)

    # select columns in the list
    data = df.loc[:, selected_cols]
    data = data.astype('float64')

    # select columns not in the list
    metadata = df.loc[:, ~df.columns.isin(selected_cols)]

    return metadata, data

def splitData(data:pd.DataFrame, train_frac:float = 0.8, val_frac:float = 0.15, shuffle:bool = True, is_dual_out:bool=False, seed_val:int=1):

    # Set the seed for numpy.random
    np.random.seed(seed_val)

    # Get unique images
    unique_images = pd.unique(data['path'])
    if shuffle:
        np.random.shuffle(unique_images) # does it inplace by default

    # Get split indices based on inputted portions
    split_params = [int( train_frac*len(unique_images) ), int( (train_frac+val_frac)*len(unique_images) )]

    # split/partition the array of file paths
    train_imgs, val_imgs, test_imgs = np.split(unique_images, split_params)

    # loop through the different partitions and use the file paths as indices
    # to split the actual data
    outs = []
    for images_part in [train_imgs, val_imgs, test_imgs]:

        # Filter out the rows that are not in the current partition (train/val/test)
        split_data = data.loc[ data['path'].isin(images_part) ]

        # Remove all the metadata columns that we don't need
        split_metadata, split_data = cleanData(split_data)

        # Define the labels, i.e. true values that we will try to predict
        if is_dual_out:
            labels = split_data[["fructose_mgdl", "glucose_mgdl"]]
        else:
            labels = split_data["fructose_mgdl"]
        
        # Drop the fructose_mgdl and glucose_mgdl columns from the data
        split_data.drop(["fructose_mgdl", "glucose_mgdl"], axis=1, inplace=True)

        outs.append( (split_data.to_numpy(), labels.to_numpy()) )

    # outs = [
    #   (train_labels, train_data),
    #   (val_labels, val_data),
    #   (test_labels, test_data)
    # ] 
    return outs

def buildDataSet(inputs:np.array, labels:np.array, 
                 batchsize:int = 32, shuffle:bool = False, 
                 data_type = torch.float32, device:torch.device=torch.device("cpu")):
    
    try:
        # Move the inputs and the labels to the specific device (GPU)
        inputs = torch.from_numpy(inputs).type(data_type).to(device)
        labels = torch.from_numpy(labels).type(data_type).to(device)

        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
        return dataset, loader
    except:
        print(inputs.shape, '\n', inputs)