# COMP 551
Applied Machine Learning coursework at McGill University.

## Project 1 - Introduction
As an introduction to practical machine learning, these are implementations (from scratch) of logistic regression with gradient descent and linear discriminant analysis. Other introductory steps include loading and cleaning the data set, extracting some basic additional features, creating a training-validation-test split, and using K-fold validation to evaluate the models.

## Project 2 - Reddit Comment Classification
This project classifies uses the text of Reddit comments to classify them by the subreddit in which they were posted. The data is pre-processed and feature vectors are extracted, then fed into the models: Bernoulli Naive Bayes (from scratch), Multinomial Naive Bayes (from SciKit-Learn), and a LSTM neural network (using PyTorch). Results were submitted to [this Kaggle leaderboard](https://www.kaggle.com/c/reddit-comment-classification-comp-551) of McGill and University of Montreal graduate-level students.

## Project 3 - MNIST Image Classification
This classifier uses a convolutional neural network to perform classification on a modified version of the MNIST dataset. Pre-processing techniques and CNN structure are varied for optimal performance. The datasets are too large to be included directly in the repository, but can be [found on Kaggle](https://www.kaggle.com/c/modified-mnist), along with the competition leaderboard.

## Project 4 - Paper Reproduction of CNN for Sentence Classification
This is a reproduction of the results in Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence). [writeup.pdf](Project_4/reports/writeup.pdf) contains the full report of this project.

The CNN repdroduction and hyperparameter tuning is performed in [ccn-rand.ipynb](Project_4/models/cnn-rand.ipynb). This file is configured to run locally, but executing it on a GPU will greatly reduce runtime.

Data pre-processing code from Kim's original paper can be found in [OriginalCode.ipynb](Project_4/models/OriginalCode.ipynb). The LSTM and Naive Bayes model implementations are located in [BasicModels.ipynb](Project_4/models/BasicModels.ipynb). These files were written to be run on Google Colab, and the "baseFilepath" variable must match the location of the [data](Project_4/data/) in your Google Drive.
