
# coding: utf-8

# In[ ]:


# Functions for the model accuracy tests
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import label_binarize

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly import tools

import pandas as pd


# In[2]:


def profileIntCols(oneDf):
    metricsInt = pd.DataFrame(columns=oneDf.select_dtypes(include=['int64']).columns,index=["No of observations", "No of Missing Values", "Maximum", "Minimum", "Mean", "Std Dev", "Percentage of Missing Values","25th Pctl", "50th Pctl", "75th Pctl"])

    for col in oneDf:
        if oneDf[col].dtype=='int64':
            a=np.array(oneDf[col])
            metricsInt.loc["No of observations",col]=oneDf[col].count()
            metricsInt.loc["No of Missing Values",col]=np.isnan(a).sum()
            metricsInt.loc["Percentage of Missing Values",col]=(metricsInt.loc["No of Missing Values",col])/(metricsInt.loc["No of observations",col]+metricsInt.loc["No of Missing Values",col])*100
            metricsInt.loc["Maximum",col]=np.nanpercentile(a, 100)
            metricsInt.loc["Minimum",col]=np.nanpercentile(a, 0)
            metricsInt.loc["Mean",col]=np.nanmean(a)
            metricsInt.loc["Std Dev",col]=np.nanstd(a)
            metricsInt.loc["25th Pctl",col]=np.nanpercentile(a, 25)
            metricsInt.loc["50th Pctl",col]=np.nanpercentile(a, 50)
            metricsInt.loc["75th Pctl",col]=np.nanpercentile(a, 75)
    metricsInt = metricsInt.transpose()
    
    # TODO: we need to print it properly
    print('Continuous variable distribution done.')
    print_full(metricsInt)
    return metricsInt

def profileFloatCols(oneDf):
    metricsFloat=pd.DataFrame(columns=oneDf.select_dtypes(include=['float64']).columns,index=["No of observations", "No of Missing Values", "Maximum", "Minimum", "Mean", "Std Dev", "Percentage of Missing Values", "25th Pctl", "50th Pctl", "75th Pctl"])

    for col in oneDf:
        if oneDf[col].dtype=='float64':
            a=np.array(oneDf[col])
            metricsFloat.loc["No of observations",col]=oneDf[col].count()
            metricsFloat.loc["No of Missing Values",col]=np.isnan(a).sum()
            metricsFloat.loc["Percentage of Missing Values",col]=(metricsFloat.loc["No of Missing Values",col])/(metricsFloat.loc["No of observations",col]+metricsFloat.loc["No of Missing Values",col])*100
            metricsFloat.loc["Maximum",col]=np.nanpercentile(a, 100)
            metricsFloat.loc["Minimum",col]=np.nanpercentile(a, 0)
            metricsFloat.loc["Mean",col]=np.nanmean(a)
            metricsFloat.loc["Std Dev",col]=np.nanstd(a)
            metricsFloat.loc["25th Pctl",col]=np.nanpercentile(a, 25)
            metricsFloat.loc["50th Pctl",col]=np.nanpercentile(a, 50)
            metricsFloat.loc["75th Pctl",col]=np.nanpercentile(a, 75)
    metricsFloat = metricsFloat.transpose()
    print('Continuous variable float distribution done.')
    print_full(metricsFloat)

def profileCategoricalCols(oneDf):
    metricsCategorical=pd.DataFrame(columns=oneDf.select_dtypes(include=['object']).columns,index=["No of observations", "No of Missing Values", "%age of Missing Values"])

    for col in oneDf:
        if oneDf[col].dtype=='object':

            metricsCategorical.loc["No of observations",col]=oneDf[col].count()
            metricsCategorical.loc["No of Missing Values",col]=oneDf[col].isnull().sum()
            metricsCategorical.loc["No of Unique Values",col]=len(oneDf[col].unique())
            metricsCategorical.loc["%age of Missing Values",col]=(metricsCategorical.loc["No of Missing Values",col])/(metricsCategorical.loc["No of observations",col]+metricsCategorical.loc["No of Missing Values",col])
    
    metricsCategorical = metricsCategorical.transpose()
    print('Profiling complete for categorical distribution done.')
    print_full(metricsCategorical)
    
# Creation of the target variable in the data frame

def addTargetVariable(df):
    Loan_Default=[]
    for row in df['loan_status']:
        if row in ["Fully Paid", "Late (16-30 days)", "In Grace Period","Current"] :
            Loan_Default.append(0)
        else: 
            Loan_Default.append(1)  

    df['Loan_Default']=Loan_Default

import seaborn as sns
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

#plotly function adddedd
def viewColumnBreakup(oneDf): 
    labels =  ['int_type','float_type','object_type']
    a = (oneDf.dtypes=='int64').value_counts()[1]
    b = (oneDf.dtypes=='float64').value_counts()[1]
    c = (oneDf.dtypes=='object').value_counts()[1]
    values=[a,b,c]

    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)
    colors = ['#42A5B3','#D15A86','#5C8100']
    data = [go.Bar(
            y=values,
            x=labels,  marker = dict( color = colors)
    )]
    layout = go.Layout(
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    )
    )
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(height=400, width=600, title='Column counts  based on Datatype')
    plotly.offline.iplot(fig, filename='horizontal-bar')


 #plotly function added   
def viewDefaulterBreakup(oneDf): 
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go  
    plotly.offline.init_notebook_mode(connected=True)
    labels =  ['Non-Defaulter','Defaulter']
    yes = len(oneDf[oneDf['default']==1])
    no = len(oneDf[oneDf['default']==0])
    values=[no,yes]

    trace1 = go.Pie(labels=labels, values=values)
    trace2 =  go.Bar(x=labels,y=values, marker=dict(color=['rgba(205,205,204,1)', 'rgba(222,45,38,0.8)']), xaxis='x2',
    yaxis='y2',name='Frequency counts')
   
    data = [trace1, trace2]
    layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.7]
    ),
    xaxis2=dict(
        domain=[0.9, 10]
    ),
    yaxis2=dict(
        anchor='x2'
    )
    )
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(height=400, width=700, title='Default / Non default % in the data set')
    plotly.offline.iplot(fig)     
    
    
    
    
# Remove the columns which have missing data more than the threshold
# 3ai
def Data_Cleanse_Missing_Value_Deletion_Threshold(df,threshold):
    column_names=[]
    for col in df:
        if df[col].isnull().mean()>=threshold/100:
            column_names.append(col)
    df=df.drop(column_names, axis=1)
    return df

#3aii
def Data_Cleanse_percentage_months(df):
    
    df['term'].astype(str)
    df['term']=df['term'].str.strip(' months')
    df['term']=pd.to_numeric(df['term'])
    
    df['int_rate'].astype(str)
    df['int_rate']=df['int_rate'].str.strip('%')
    df['int_rate']=pd.to_numeric(df['int_rate'])
    
    df['revol_util'].astype(str)
    df['revol_util']=df['revol_util'].str.strip('%')
    df['revol_util']=pd.to_numeric(df['revol_util'])
    
    return df

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
    


# In[3]:


#3aiii
def Data_Cleanse_missingvalue(df):
    
    for col in df:
        if df[col].dtype=='int64':
            print('Imputation with Mean: %s' % (col))
            df[col].fillna((df[col].mean()), inplace=True)
        elif df[col].dtype=='float64':
            print('Imputation with Mean: %s' % (col))
            df[col].fillna((df[col].mean()), inplace=True)
        elif df[col].dtype=='object':
            print('Imputation with Mode: %s' % (col))
            df[col].fillna((df[col].mode()), inplace=True)
    return df


# In[5]:


#3c: Dimensionality reduction:
def Data_Cleanse_DimensionalityReduction_Corr(df,threshold):
    # Extract values and row, column names
    df_corr = df.corr()
    arr = df_corr.values
    index_names = df_corr.index
    col_names = df_corr.columns

    #  Get indices where such threshold is crossed; avoid diagonal elems
    R,C = np.where(np.triu(arr,1)>threshold)

    # Arrange those in columns and put out as a dataframe
    out_arr = np.column_stack((index_names[R],col_names[C],arr[R,C]))
    df_out = pd.DataFrame(out_arr,columns=[['row_name','col_name','value']])
    #result = df_out.sort(['value'], ascending=[1, 0])

    
    return df_out


# In[6]:


#3b Feature Encoding of all the Categorical Variables on Train, Test and Validate CSVs 

def Data_Cleanse_Feature_Encoding(df):
    #purpose_domestic appliances+ radio/television
    purpose_da=[]
    for row in df['purpose']:
        if row in ["domestic appliances","radio/television"] :
            purpose_da.append(1)
        else: 
            purpose_da.append(0)    
    df['purpose_da']=purpose_da
    
    #purpose car_new+car_used
    purpose_car=[]
    for row in df['purpose']:
        if row in ["car (new)","car (used)"] :
            purpose_car.append(1)
        else: 
            purpose_car.append(0)    
    df['purpose_car']=purpose_car
    
    #purpose repairs+furniture/equipment+retraining
    purpose_rep=[]
    for row in df['purpose']:
        if row in ['furniture/equipment', 'repairs', 'retraining'] :
            purpose_rep.append(1)
        else: 
            purpose_rep.append(0)    
    df['purpose_rep']=purpose_rep
    
    #purpose business
    purpose_bns=[]
    for row in df['purpose']:
        if row in ['business'] :
            purpose_bns.append(1)
        else: 
            purpose_bns.append(0)    
    df['purpose_bns']=purpose_bns
    
    
    #purpose vacation
    purpose_vac=[]
    for row in df['purpose']:
        if row in ['(vacation - does not exist?)'] :
            purpose_vac.append(1)
        else: 
            purpose_vac.append(0)    
    df['purpose_vac']=purpose_vac
    
    
    #purpose education
    purpose_edu=[]
    for row in df['purpose']:
        if row in ['education'] :
            purpose_edu.append(1)
        else: 
            purpose_edu.append(0)    
    df['purpose_edu']=purpose_edu
    
    
    #personal_status_sex(male:single+divorced+married)
    personal_status_sex_male=[]
    for row in df['personal_status_sex']:
        if row in ['male : single', 'male : divorced/separated', 'male : married/widowed'] :
            personal_status_sex_male.append(1)
        else: 
            personal_status_sex_male.append(0)    
    df['personal_status_sex_male']=personal_status_sex_male
    
    
    #personal_status_sex(male:single+divorced+married)
    personal_status_sex_female=[]
    for row in df['personal_status_sex']:
        if row in ['female : divorced/separated/married'] :
            personal_status_sex_female.append(1)
        else: 
            personal_status_sex_female.append(0)    
    df['personal_status_sex_female']=personal_status_sex_female
    
    
    #other debtors
    other_debtors_gnt=[]
    for row in df['other_debtors']:
        if row in ['guarantor', 'co-applicant'] :
            other_debtors_gnt.append(1)
        elif row in ["none"] : 
            other_debtors_gnt.append(0)    
        else: 
            other_debtors_gnt.append(0)  
    df['other_debtors_gnt']=other_debtors_gnt
    
    return df


# In[7]:



##Computing KS stat with help of function    
def KS_Stats_Calculation(df):
    #Ks Statistics:
    data_good=[]  #Non Defaulters

    for row in df['Observed']:
        if row in [0] :
            data_good.append(1)
        else: 
            data_good.append(0)    
    df['data_good']=data_good
    data_bad=[]  #Defaulters

    for row in df['Observed']:
        if row in [1] :
            data_bad.append(1)
        else: 
            data_bad.append(0)    
    df['data_bad']=data_bad
    
    # Define buckets
    df['bucket'] = pd.qcut(df.PD,8,duplicates='drop')
    #data['bucket'] = pd.qcut(data.PD, 10,duplicates='drop')
    # Group dataframe with buckets
    grouped = df.groupby('bucket', as_index = False)
    
    # Create a summary data frame 
    agg1 = grouped.min().PD 
    agg1 = pd.DataFrame(grouped.min().PD, columns = ['min_scr'])
    agg1['min_scr'] = grouped.min().PD
    agg1['max_scr'] = grouped.max().PD 
    agg1['bads'] = grouped.sum().data_bad 
    agg1['goods'] = grouped.sum().data_good 
    agg1['total'] = agg1.bads + agg1.goods
    
    #Sorting table by minimum score
    agg2 = (agg1.sort_index(by = 'min_scr')).reset_index(drop = True)
    agg2['bad_rate'] = (agg2.bads / agg2.total).apply('{0:.2%}'.format)
    agg2['good_rate'] = (agg2.goods / agg2.total).apply('{0:.2%}'.format)
    #calculating ks for each bin
    agg2['ks'] = np.round(((agg2.goods / df.data_good.sum()).cumsum() - (agg2.bads / df.data_bad.sum()).cumsum()), 4) * 100
    
    # define a function to flag max KS
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    # Flag out max KS
    agg2['max_ks'] = agg2.ks.apply(flag)
    #print (agg2) 
    
    #Storing cumulative good and BAd
    df['cum_good']=np.round((agg2.goods / df.data_good.sum()).cumsum(), 4)*100
    df['cum_bad']=np.round((agg2.bads / df.data_bad.sum()).cumsum(), 4)*100
    
    
    cum_bad=list(df['cum_bad'].dropna(how='any'))
    cum_good=list(df['cum_good'].dropna(how='any'))
    
    Cum_Bad=[0.0]
    for i in range(len(cum_bad)):
        Cum_Bad.append(cum_bad[i])
    Cum_good=[0.0]
    for i in range(len(cum_good)):
        Cum_good.append(cum_good[i])
        
    ks=pd.DataFrame(data=dict(Cum_good=Cum_good, Cum_Bad=Cum_Bad))
    return (ks,agg2)
    
    
    
    
#Creating the Function for creating the Confusion Matrix
def calculate_confusion_matrix(y_true, y_pred):
    
    cm=confusion_matrix(y_true, y_pred)
    print(cm)


# In[8]:


#3b Feature Encoding of all the Categorical Variables on Train, Test and Validate CSVs 

def Data_get_dummies(df):

    dff = pd.get_dummies(df, columns=['savings', 'present_emp_since','property',  'other_installment_plans','housing', 'job', 'foreign_worker'])
    
    return dff


# In[9]:


def calculate_roc_curve(Traindf,Y_test, y_pred, pos_label):    
# Compute ROC curve and ROC area for each class
    y = Traindf['default']
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
        roc_auc = auc(fpr, tpr)
    fpr1=pd.DataFrame(data=dict(fpr=fpr)) 
    tpr1=pd.DataFrame(data=dict(tpr=tpr))
    thresholds1=pd.DataFrame(data=dict(thresholds=thresholds))
    return (fpr1,tpr1,thresholds1)


# In[10]:


def calculate_Precision_Recall(Traindf,Y_test, y_pred, pos_label):    
# ComputePrecision n recall values for each class
    from sklearn.metrics import precision_recall_curve      
    y = Traindf['default']
    y = label_binarize(y, classes=[0, 1])
    n_classes = y.shape[1]
    Precision = dict()
    Recall = dict()
    Threshold = dict()
    for i in range(n_classes):
        Precision, Recall, Threshold = precision_recall_curve(Y_test, y_pred) 
    Precision1=pd.DataFrame(data=dict(Precision=Precision)) 
    Recall1=pd.DataFrame(data=dict(Recall=Recall))
    Threshold1=pd.DataFrame(data=dict(Threshold=Threshold))
    return (Precision1,Recall1,Threshold1)   


# In[11]:


#plotly function for creating KS table
def KSTable(oneDf): 
    
    plotly.offline.init_notebook_mode(connected=True)
    trace = go.Table (
    header=dict(values=oneDf.columns,
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[oneDf.min_scr,oneDf.max_scr,oneDf.bads,oneDf.goods,oneDf.total,oneDf.bad_rate,oneDf.good_rate,oneDf.ks,oneDf.max_ks],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

    data = [trace] 
    
    plotly.offline.iplot(data, filename = 'pandas_table')


# In[12]:


##plotly function for evaluation metrics
def EvaluationMetrics(FP,TP,TH,Prec,Rec,Default,NDefault): 

    plotly.offline.init_notebook_mode(connected=True)

    TH = np.append(TH, 1)
    id1=[0,1,2,3,4,5,6,7,8]
    trace1 = go.Scatter(x=FP, y=TP,name='ROC Curve',mode='ROC Curve',line=dict(color='blue'))
    trace2=go.Scatter(x=[0,1],y=[0,1],name='',mode='dash',line=dict(color='navy',  dash='dash'),
                    showlegend=False)

    trace3 = go.Scatter(x=TH, y=Prec,name='Precision',mode='Precision',xaxis='x2',yaxis='y2',line=dict(color='red'))
    trace4=go.Scatter(x=TH,y=Rec,name='Recall',mode='Recall',xaxis='x2',yaxis='y2',line=dict(color='green'))

    trace5 = go.Scatter(x=id1, y=Default,name='Cum_Defaulters',mode='Cum_Defaulters',xaxis='x3',yaxis='y3')
    trace6=go.Scatter(x=id1,y=NDefault,name='Cum_NonDefaulters',mode='Cum_NonDefaulters',xaxis='x3',yaxis='y3')


    fig = tools.make_subplots(rows=1, cols=3, subplot_titles=('ROC Curve', 'Precision vs Recall',
                                                          'KS Stat'))
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 1, 2)
    fig.append_trace(trace5, 1, 3)
    fig.append_trace(trace6, 1, 3)


    fig['layout']['xaxis1'].update(title='1-Specificty ')
    fig['layout']['xaxis2'].update(title='Recall')
    fig['layout']['xaxis3'].update(title='Cum defaulters', showgrid=False)


    fig['layout']['yaxis1'].update(title='Sensitivity')
    fig['layout']['yaxis2'].update(title='Precision')
    fig['layout']['yaxis3'].update(title='Cum Non defaulters', showgrid=False)

    fig['layout'].update(height=400, width=1200, title='Model evaluation Metrics')
    plotly.offline.iplot(fig)


# In[13]:


def FeatureImportance(labels,values): 
    plotly.offline.init_notebook_mode(connected=True)

    trace1 = go.Bar(
    x=labels,
    y=values,
    #orientation = 'h',
    marker = dict(
        color = 'rgba(248, 248, 255)',
        line = dict(
            color = 'rgba(248, 248, 255)',
            width = 1)
    ))
    layout = go.Layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=True,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=True,
    ),
      showlegend=False,
    )
    data = [trace1]
    fig = go.Figure(data=data)
    fig['layout'].update(title='Feature Importance',height=400, width=800)
    plotly.offline.iplot(fig)

