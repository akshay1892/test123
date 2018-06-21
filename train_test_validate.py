# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:49:22 2018

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


class Train:
    
    training_dataset = 'https://s3.ap-south-1.amazonaws.com/ci-ml-credit-risk/Credit+Risk/dataset/train/02-train-engg.csv'
    test_dataset_url = 'https://s3.ap-south-1.amazonaws.com/ci-ml-credit-risk/Credit+Risk/dataset/test/02-test-engg.csv'
    
    def train_model(self):
        
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
        
        
        # Initialize the logistic regression model
        lg = LogisticRegression(random_state=1)
        
        lg.fit(Train_Features,Train_Target)
        
        
        # Timestamp
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        
        # Prepare the pickle file.
        
        filenameLogReg = 'model/Logistic_model.pkl'
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

        print('MOdel trained')
        
        
class Test:
    
    validate_dataset_url = 'https://s3.ap-south-1.amazonaws.com/ci-ml-credit-risk/Credit+Risk/dataset/validate/02-validate-engg.csv'
    test_dataset_url = 'https://s3.ap-south-1.amazonaws.com/ci-ml-credit-risk/Credit+Risk/dataset/test/02-test-engg.csv'

    
    def test_model(self):
        
        Test1 =  pd.read_csv(test_dataset_url,sep = ',',encoding='latin-1')
        Test1 = Test1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)
        Validate1 =  pd.read_csv(validate_dataset_url, sep = ',',encoding='latin-1')
        Validate1 = Validate1.drop(['data_uid','States','Unnamed: 0','Date','Zipcode'],axis=1)
        
        
        
        Test_Features = Test1.loc[:, Test1.columns != 'default']
        Test_Target = Test1['default']
        #Test_Features = Test_Features.drop(['Unnamed: 0'],axis=1)
        Validate_Features = Validate1.loc[:, Validate1.columns != 'default']
        Validate_Target = Validate1['default']
        #Validate_Features = Validate_Features.drop(['Unnamed: 0'],axis=1)
        
        
        
        
        
        # loading the pickled model from disk 
        filenameGBM = 'model/Logistic_model.pkl'
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
        


training_model = Train()
training_model.train_model()       
        
        
validate_model = Test()
validate_model.test_model()
        
