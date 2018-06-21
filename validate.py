# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:18:29 2018

@author: Administrator
"""




# All Imports to the files

from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly
import plotly.graph_objs as go
from plotly import tools

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy import interp
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
# This is our function file being imported
import utils2 as myUtil


Test1 =  pd.read_csv('02-test-engg.csv',sep = ',',encoding='latin-1')
Test1 = Test1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)
Validate1 =  pd.read_csv('02-validate-engg.csv',sep = ',',encoding='latin-1')
Validate1 = Validate1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)



Test_Features = Test1.loc[:, Test1.columns != 'default']
Test_Target = Test1['default']
#Test_Features = Test_Features.drop(['Unnamed: 0'],axis=1)
Validate_Features = Validate1.loc[:, Validate1.columns != 'default']
Validate_Target = Validate1['default']
#Validate_Features = Validate_Features.drop(['Unnamed: 0'],axis=1)


# loading the pickled model from disk 
filenameGBM = 'GBM_model.pkl'
loaded_model2 = pickle.load(open(filenameGBM, 'rb'))
#Predictions
preds2 = loaded_model2.predict_proba(Validate_Features)[:,1]
Test_pred2=loaded_model2.predict(Validate_Features)
y_score1 = loaded_model2.decision_function(Validate_Features)
predict_fn_gbm = lambda x: loaded_model2.predict_proba(x).astype(float)
#calculate Confusion Matrix
myUtil.calculate_confusion_matrix(Validate_Target, Test_pred2)
accuracy_score_calc = accuracy_score(Validate_Target, Test_pred2)*100
print('Accuracy : ',accuracy_score(Validate_Target, Test_pred2)*100,'%')
import sys
if accuracy_score_calc > 95:
    print('Overfitting! Accuracy more than 95%')
    sys.exit(-1)
elif accuracy_score_calc < 85:
    print('Underfitting! Accuracy less than 85%')
    sys.exit(-1)


#calculate ROC curve
#fpr2,tpr2,thresholds2 = myUtil.calculate_roc_curve(Validate1 , Validate_Target, y_score1,2) 
