## Predicting seizures from EEG data

This repo contains scripts for a Kaggle competition ([1]) aimed at predicting 
seizures from 10-minute clips of EEG voltage recordings. The data came from 
2 human subjects and 5 dogs.

The base directory includes modules with functions used for generating 
features from the raw time series data, exploratory analysis, training 
and optimizing generic models, and applying models to test data.

The computed features (but not the raw data) are stored in the `data`
directory.
Scripts to train and optimize specific classification models, using 
the functions in `train_model.py` and `optimize_model.py`, can be 
found in `scripts`, and the `submissions` directory contains CSV files
and log files for all submitted predictions.

One of the main challenges was normalizing the predicted probabilities
across the 7 different subjects to maximize the area under the ROC 
curve (AUC) for all subjects. Despite trying isotonic regression and other
methods to combine the predictions, the combined AUC was limited to around
0.6 while the estimated AUC for individual subjects was in the range
0.75 to 0.95.

[1]: https://www.kaggle.com/c/seizure-prediction
