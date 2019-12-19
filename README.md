# COMP 551 - Assignment 4 - Group 2
This is a reproduction of the results in Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://github.com/yoonkim/CNN_sentence). [writeup.pdf](reports/writeup.pdf) contains the full report of this project.

The CNN repdroduction and hyperparameter tuning is performed in [ccn-rand.ipynb](models/cnn-rand.ipynb). This file is configured to run locally, but executing it on a GPU will greatly reduce runtime.

Data pre-processing code from Kim's original paper can be found in [OriginalCode.ipynb](models/OriginalCode.ipynb). The LSTM and Naive Bayes model implementations are located in [BasicModels.ipynb](models/BasicModels.ipynb). These files were written to be run on Google Colab, and the "baseFilepath" variable must match the location of the [data](data/) in your Google Drive.