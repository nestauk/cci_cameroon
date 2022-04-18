# Running the models

<br>
<br>
The x2 scripts in this folder can be used to run the classification and clustering models on new rumours (defaulted to the test set). The scripts use the held out test dataset to first classfiy the rumours and then cluster the rumours that cannot be classified. Both scripts can be run seperately and can use any new rumour datasets that are in the correct format (contain a comment field for predicting and clustering).

### Steps to take before running

To run the models you will first need to setup the project. Follow the below two steps to do this:

1. Clone the project and cd into the cci_cameroon directory
2. Run the command make install to create the virtual environment and install dependencies

To note the project is setup using the Nesta Cookiecutter (guidelines on the Nesta Cookiecutter [can be found here](https://nestauk.github.io/ds-cookiecutter/structure/)).

#### Input needed

After you setup the project you will need your test dataset. To build our models we used the dataset provided to us by the IFRC and Cameroon Red Cross containing rumours, beliefs and observations from community members (more information on the data source can be found on the [IFRC.Go website](https://go.ifrc.org/emergencies/4583#community-data)). To run the scripts the data needs to be in the following format:

- Filename: X_test.xlsx
- Saved in inputs/data/data_for_modelling

| id      | comment     |
| ------- | ----------- |
| numeric | rumour text |

### Run the models

Perform the following steps to run the models:

- run python3 classification_model_run.py
- run python3 clustering_model_run.py

### Outputs

There are three files created from running the models and saved to outputs:

1. all_predictions.xlsx
2. not_classified.xlsx
3. clusters.xlsx

The first two are created from the classification_model_run file and the third is created from clustering_model_run file. If a rumour is classified by the model to one or more of the eight codes it is saved in the 'all_predictions' file. If the rumour cannot be classfied it is saved into the the 'not_classified' file. Both files also save the rumours 'ID' assigned so it can be referenced back to the test set for reporting.

The 'not_classified' file is used as input to run the clustering algorthm in the 'clustering_model_run' file in the same folder.

The 'clusters' file contains the unclassified comments broken into the clusters chosen by the clustering model. Each cluster in saved as a seperate sheet in an excel file.
