import re
import pandas as pd
import numpy as np
from tkinter.filedialog import askopenfilenames
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

def loadCSV(filepath:os.PathLike):

    assert os.path.exists(filepath), "This file does not exist. Please enter a valid data file path."

    # Read in the CSV file
    df = pd.read_csv(filepath) # ./data/2023-03-09-fructose-glucose-water-signals.csv

    # Rename the first column to index, then set that as the index
    df.rename(columns={"Unnamed: 0":"index"}, inplace=True)
    df.set_index("index", inplace=True)

    # # Find the first column that has reflectance data
    # # i.e. a column name that can be converted to a float
    # for idx, colname in enumerate(df.columns):
    #     try:
    #         float(colname)
    #         break
    #     except ValueError:
    #         continue
    # # print("The variable `idx-1` points to the column '", df.columns[idx-1], "'", sep='')

    # # Split the dataframe at `idx` into metadata and data tables
    # metadata = df.iloc[:, :idx]
    # data = df.iloc[:, [1,2]+list(range( idx-1, len(df.columns) ))] # include the path column in data for train/test split
    
    metadata, data = cleanData(df)

    return metadata, data

def getData():

    # Get the filenames
    data_paths = askopenfilenames(title = "Select data file", defaultextension = ".csv", initialdir=os.getcwd())

    # Load the Data
    meta_dfs, data_dfs = [], []
    for f in data_paths:
        m, d = loadCSV(f)
        meta_dfs.append(m)
        data_dfs.append(d)

    metadata = pd.concat(meta_dfs)
    data = pd.concat(data_dfs)

    return metadata, data

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

    # # Find the first column that has reflectance data
    # # i.e. a column name that can be converted to a float
    # for idx, colname in enumerate(df.columns):
    #     try:
    #         float(colname)
    #         break # When you find the first column name that doesn't error out, break the loop
    #     except ValueError:
    #         continue # if you cannot convert to float, keep the loop going with the next column name
        
    # # print("The variable `idx-1` points to the column '", df.columns[idx-1], "'", sep='')

    # # Split the dataframe at `idx` into metadata and data tables
    # metadata = df.iloc[:, :idx]
    # data = df.iloc[:, [1,2]+list(range( idx, len(df.columns) ))] # include the path column in data for train/test split

    return metadata, data

def splitData(data:pd.DataFrame, train_frac:float = 0.75, val_frac:float = 0.2, shuffle:bool = True, seed:int=None, is_dual_out:bool=True):

    # Shuffle the dataset using pd.DataFrame.sample
    if shuffle:
        if seed is not None:
            data = data.sample(frac=1, random_state=seed)
        else:
            data = data.sample(frac=1)
        # frac = 1 means return the entire dataset instead of a smaller fraction of it
    
    # Define the labels, i.e. true values that we will try to predict
    if is_dual_out:
        labels = data[["fructose_mgdl", "glucose_mgdl"]]
    else:
        labels = data["fructose_mgdl"]

    # Drop the fructose_mgdl and glucose_mgdl columns from the data
    data.drop(["fructose_mgdl", "glucose_mgdl"], axis=1, inplace=True)
        
    # Define the split parameters
    split_params = [int( train_frac*len(data) ), int( (train_frac + val_frac)*len(data) )]

    # Split the dataframe according to the params
    inputs_train, inputs_val, inputs_test = np.split(data, split_params)
    # Split the labels along the same params
    labels_train, labels_val, labels_test = np.split(labels, split_params)

    return (inputs_train.to_numpy(), labels_train.to_numpy()), \
        (inputs_val.to_numpy(), labels_val.to_numpy()), \
            (inputs_test.to_numpy(), labels_test.to_numpy())

def splitData2(data:pd.DataFrame, train_frac:float = 0.85, val_frac:float = 0.14, shuffle:bool = True, is_dual_out:bool=True):

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
        split_data = cleanData(split_data)[-1]

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