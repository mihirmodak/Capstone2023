# Establishing Fructose as a Biomarker for Liver Function Using Hyperspectral Imaging

This code is currently under development by Team 5 for the Bioengineering Senior Capstone Project at the University of Maryland, College Park.

## Abstract

Liver cancer is one of the most prevalent and lethal types of cancer in the United States. It has a low survival rate and the imaging technologies used to detect liver cancer such as MRIs, CT scans, and angiograms are expensive, slow, invasive, and not available as point-of-care devices. However, hepatocellular carcinomas have been correlated with decreased levels of fructose metabolism and higher concentrations of fructose in the blood. Changes in fructose concentrations can be used as a marker for hepatic cancer. Our group’s project therefore involved the use of novel hyper-spectral imaging (HSI) techniques to detect fructose concentrations in various samples. Specifically, a HSI camera was used to image solutions consisting of either water or horse serum containing different ratios of fructose and glucose. The data was then used to train a neural network to model the correlation between the transmittance data from the HSI system and the fructose concentration that was imaged. We found that the neural network was able to effectively differentiate fructose from glucose. It also had an R2 value greater than 0.9 on an unseen validation dataset, indicating that it can successfully use HSI data to predict fructose concentrations in serum samples. In the future, this technology could be minimized in order to be used as a point-of-care device. An ethical impact of this research would be the ability to inexpensively screen for hepatic cancers in their early stages. This would avoid expensive treatments associated with late stages of liver cancer and be available to more people than current diagnostics.

## Code Structure

This repository contains two iterations of the data processing algorithm. The first version is a Partial Least-Squares Regression model, trained on mixutres of sugar in water. The working code for this data processing pipeline is located in `plsr/current.ipynb`. There is currently no way of saving this model's parameters to a file. However, the model does train extremely quickly, so it is possible to re-train the model multiple times without a signficant delay.

The second data processing pipeline is a neural network, trained on mixtures of fructose and glucose in both water and serum. All the necessary code is broken down into the following directory structure:

```bash
.
│   FullyConnectedRegressorv1.pt
│   models.py
│   predict.py
│   training.py
|
├───data
|
├───images
│   ├───20230409
│   ├───20230424
│   └───animation
|
├───models
│   ├───20230409
│   └───20230424
|
└───utils
```

`models.py` contains the various model class definitions. Currently there are two &mdash; `H4O1`, and it's child class (identical to its parent) `FullyConnectedRegressor`.

`predict.py` contains all the code necessary to use the model to make predictions. This will open dialog boxes to select the model file to use, and the test dataset CSV files. If the tkinter library is not installed, then it falls back to asking for the model and data paths in the command line.

`training.py` contains all the code to train the model on new data. Similar to the prediction code, it also uses tkinter library to generate dialog boxes with the command line as a fallback to get a data source directory path.

`./models` contains a few fully trained `FullyConnectedRegressor` models. These models are divided up by the date they were trained. The best-performing model of these is saved as the `{date}_best.pt`

`./data` contains the CSV files used for training the model. These files come from the HSI camera after being processed through the data preprocessing pipeline developed by Anjana. Each data file needs to be in the following format.

| No. | sample_key | fructose_mgdl | glucose_mgdl | Date Imaged | row | col | path | ... |
| --- | ---------- | ------------- | ------------ | ----------- | --- | --- | ---- | --- |
| 0 | 0 | 0 | 112 | 04/06/2023 | 1 | 1 | ~/hsi/... | ... |

`./utils` contains some helper functions and classes that the training and prediction code files use for their core functionality. Operations like saving and loading models from a file are encoded in this directory.

`./images` contains graphs and images that were generated on various datasets either through the training loop or during the prediction process. Some code to generate an animation of continuous monitoring using this model is also stored in the `animation` sub-directory.

## Prerequisites & Dependencies

For prediction on a novel dataset, the following dependencies are required:

- torch (installed as per your system configuration from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
- matplotlib==3.7.1

To train the model locally, the following additional dependencies are required:

- numpy==1.24.1
- pandas==2.0.0
- scikit_learn==1.2.2

All of the above dependencies &mdash; except torch and the associated PyTorch modules &mdash; are in the requirements.txt file included with this repo.

To install all required packages (except PyTorch), use the following command:

```bash
pip install -r requirements.txt
```

## Mentors

- Sui-Seng Tee
- Yang Tao
- Maurizio Cattaneo
- Anjana Hevaganinge
