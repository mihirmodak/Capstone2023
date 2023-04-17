import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader
import utils
from models import Model
import argparse

def predict(model, loader:DataLoader):
    predictions = [] # An array that holds the predicted values
    results = [] # An array that holds the true values and the predicted values

    # Make predictions on the test data
    with torch.no_grad():
        for (inputs_batch, labels_batch) in loader:

            # In the case of a single output node, we want to reshape the labels array
            # from a row vector to a column vector
            if len(labels_batch.shape) == 1:
                labels_batch = labels_batch.unsqueeze(1)

            # Forward pass
            outputs_batch = model(inputs_batch)

            # Add Output to Predictions
            predictions.extend(outputs_batch.tolist())
            
            # Build the structure of the results and add it to the large array
            next_result = list(zip(labels_batch.tolist(), outputs_batch.tolist()))
            results.extend(next_result)

    # Convert both to numpy arrays to enable complex indexing
    predictions = np.array(predictions)
    results = np.array(results)

    return predictions, results


def plot(results, args=None):


    # Generate the subplots with shared X and Y axes
    fig, ax = plt.subplots(1, 1)
        
    labels = results[:, 0, 0] # First row is the labels
    preds = results[:, 1, 0]

    # Plot the points as a scatter plot, then plot the y=x line from the labels
    ax.scatter(labels, preds)
    ax.plot(labels, labels, 'r')
    ax.set_xlabel("True Concentration (mg/dL)")
    ax.set_ylabel("Predicted Concentration (mg/dL)")
    ax.set_title(f"Fructose Concentration with R^2={round(r2_score(y_true=labels, y_pred=preds), 5)}")

    plt.show()

def plot_agg(results, args:dict=None):


    # Generate the subplots with shared X and Y axes
    fig, ax = plt.subplots(1, 1)
        
    labels = results[:, 0, 0] # First row is the labels
    # Find the unique labels
    unique_labels = np.unique(labels)

    # Calculate the mean of predictions for each label
    pred_means = np.zeros((unique_labels.size,))
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        if args['aggregation'] == "median":
            pred_means[i] = np.median(results[mask, 1, 0])
        else:
            pred_means[i] = np.mean(results[mask, 1, 0])

    # Plot the points as a scatter plot, then plot the y=x line from the labels
    ax.scatter(unique_labels, pred_means)
    ax.plot(unique_labels, unique_labels, 'r')
    ax.set_xlabel("True Concentration (mg/dL)")
    ax.set_ylabel("Predicted Concentration (mg/dL)")
    ax.set_title(f"Fructose Concentration with R^2={round(r2_score(y_true=unique_labels, y_pred=pred_means), 5)}")

    plt.show()


def plot_multiple(results, args=None):

    if results.shape[-1] ==2:
        glucose_arr = results[:, 0, 1]
    else:
        glucose_arr = np.array([0])
    # One subplot for each glucose concentration
    unique_glucose_concs = np.unique(glucose_arr)
    num_subplots = len(unique_glucose_concs)

    # Try to get as square of a grid as possible, using the sqrt to find the number of rows
    num_rows = round(np.sqrt(num_subplots))
    num_cols = round(num_subplots / num_rows)

    # Generate the subplots with shared X and Y axes
    fig, axs = plt.subplots(num_rows, num_cols)

    axs_loop = axs.ravel() if num_subplots > 1 else [axs]

    for idx, ax in enumerate(axs_loop):

        # Each axis corresponds to one glucose conc, so the indexes are the same
        curr_glucose_conc = unique_glucose_concs[idx]

        # Separate out the labels and the predictions for each glucose conc.
        if results.shape[-1] == 2:
            curr_results = results[ glucose_arr == curr_glucose_conc ]
        else:
            curr_results = results
            
        curr_labels = curr_results[:, 0, 0] # First row is the labels
        curr_preds = curr_results[:, 1, 0] # Seconds row is the predictions

        # Plot the points as a scatter plot, then plot the y=x line from the labels
        ax.scatter(curr_labels, curr_preds)
        ax.plot(curr_labels, curr_labels, 'r')
        ax.set_xlabel("True Concentration (mg/dL)")
        ax.set_ylabel("Predicted Concentration (mg/dL)")
        if results.shape[-1] == 2:
            ax.set_title(f"Fructose Concentration for Glucose = {curr_glucose_conc} mg/dL")
        else:
            ax.set_title(f"Fructose Concentration with R^2={round(r2_score(y_true=curr_labels, y_pred=curr_preds), 5)}")

    plt.show()

def main(args:dict=None):

    # Use the GPU if available, else use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device `{device}`")

    # Load the model from a file
    net = Model.load(device=device)[0]
    num_output_nodes = net(torch.randn(1, 1, 125).to(device)).shape[-1]

    # Get a filename -> "./neural_net/data/2023-03-09-fructose-glucose-water-signals.csv"
    data_df = utils.getData()

    splits = utils.splitData(data_df, train_frac = 0, val_frac = 0, shuffle = False, is_dual_out= num_output_nodes == 2) # train_frac = 0.75, val_frac = 0.2, seed = 1,
    _, _, (inputs_test, labels_test) = splits

    test_data, test_loader = utils.buildDataSet(
        inputs = inputs_test,
        labels = labels_test,
        device = device # place the newly created data objects on the appropriate device
    )

    # Predict on the test dataset and plot the predictions
    predictions, results = predict(net, test_loader)
    # print(results.shape, results[0])
    if args['aggregation'] in ["mean", "median"]:
        plot_agg(results, args)
    elif args['aggregation'] == "none":
        plot(results)
    # print( r2_score(y_true=labels_test, y_pred=predictions) )

    return predictions, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-agg", "--aggregation", choices=["mean", "median", "none"], default="none")
    args = parser.parse_args()
    args = vars(args)
    main(args)