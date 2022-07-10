# Tuning of classification models

## Introduction

In this document, an overview of the tuning done to improve the performance of the classification models is done. Using two different transformer models, different encodings are produced and fed into the classification models. Grid search is used to tune the parameters of the models and the best performed model is chosen. The cleaned data is used to train and test models' performance.

## Model tuning

This is done in `model_tuning.py` with helper functions found in the files `process_workshop_data.py` and `model_tuning_report.py`. Using the train/test datasets, four different models are trained and tested. The models used are Random Forest Classifier, Decision Tree Classifier, Support Vector Classifier, K-Neighbors Classifier

## Final output

The final output of the tuning process is the best model which is KNN classifier. It is later used to classify other data points.
