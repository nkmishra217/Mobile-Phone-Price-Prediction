#source_code file 1 (experimentation)
The 'mobile_prediction.py' file contains the code wherein we have trained and tested the model using only the 'train.csv'. We splitted the training dataset into 15% test set and 85% training set. We achieved an accuracy score of 95% + using the Logistic Regression classification model.

The feature set has been feature-scaled using StandardScaler class.

#source code file 2 (actual result)
The 'prediction_model_to_test.py' file contains the code wherein we trained the model using the 'train.csv' and predicted the outcomes for the dataset in the 'test.csv'. The outcome/result has been complied into 'result.csv' file. The algorithm used is the same: Logistic Regression.