# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:50:05 2018

@author: Administrator
"""

# All Imports to the files
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import datetime
import time
# This is our function file being imported
import utils2 as myUtil



training_dataset = 'https://s3.ap-south-1.amazonaws.com/ci-ml-credit-risk/Credit+Risk/dataset/train/02-train-engg.csv'
test_dataset_url = 'https://s3.ap-south-1.amazonaws.com/ci-ml-credit-risk/Credit+Risk/dataset/test/02-test-engg.csv'

Train1 =  pd.read_csv(training_dataset,sep = ',',encoding='latin-1')
Train1 = Train1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)
Test1 =  pd.read_csv(test_dataset_url,sep = ',',encoding='latin-1')
Test1 = Test1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)
#Validate1 =  pd.read_csv('02-validate-engg.csv',sep = ',',encoding='latin-1')
#Validate1 = Validate1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)


# Creating Training and Test Features & Training and Test Target variables for modeling purpose
Train_Features = Train1.loc[:, Train1.columns != 'default']
Train_Target = Train1['default']

Test_Features = Test1.loc[:, Test1.columns != 'default']
Test_Target = Test1['default']

#Validate_Features = Validate1.loc[:, Validate1.columns != 'default']
#Validate_Target = Validate1['default']


# Initialize the logistic regression model
lg = LogisticRegression(random_state=1)

lg.fit(Train_Features,Train_Target)


# Timestamp
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

# Prepare the pickle file.

filenameLogReg = 'Logistic_model_'+ st +'.pkl'
pickle.dump(lg, open(filenameLogReg, 'wb'))
# loading the pickled model from disk 
loaded_model = pickle.load(open(filenameLogReg, 'rb'))
#Predictions
predictionPD = loaded_model.predict_proba(Test_Features)[:,1]
predict_fn_eclf = lambda x: loaded_model.predict_proba(x).astype(float)
Test_Score = loaded_model.fit(Train_Features, Train_Target).decision_function(Test_Features)
predictionClass = loaded_model.predict(Test_Features)
#calculate ROC curve
fpr, tpr, thresholds = myUtil.calculate_roc_curve(Train1,Test_Target, Test_Score,2) 
#calculate Confusion Matrix
myUtil.calculate_confusion_matrix(Test_Target, predictionClass)
print(accuracy_score(Test_Target, predictionClass))


