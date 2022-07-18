# Classification model developement

This folder holds the notebooks used to develop the classification model used in `pipeline/model_workflow.` The notebooks in this directory aren't part of the codebase to run, but are here for for anyone interested in how the models were developed.

### Notebooks
- `classification_model_development.py`: This notebook contains the code to train and different classification models and parameters using grid-search.
- `classification_model_test.py`: This notebook contains the code to test the results of the models on the held out test set.
- `model_test_drc_dataset.py`: This notebook contains the code to test the model on the DRC test dataset.
- `model_threshold_setting.py`: This notebook contains the code to test the results of model on different thresholds as well as to perform the bias audit.
