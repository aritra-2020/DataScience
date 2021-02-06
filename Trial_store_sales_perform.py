# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:09:11 2020

@author: ARITRA
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import datetime
Data_stores=pd.read_csv('D:/IISWBM_Business_Analytics/Quantium Virtual Internship/Final_dat.csv')
Data_stores.head()
Data_stores.tail()

'''Objective is to select control store for each trial store and analyse the difference in sales and the drivers behind
this difference'''

#Changed Date format to 'DD Month YYYY' in excel and added month and year metrics for further calculations

Data_stores['Month']=Data_stores['DATE'].apply(lambda date: date.split(' ')[1])
Data_stores['Year']=Data_stores['DATE'].apply(lambda date : date.split(' ')[2])
Data_stores.head()

'''Algorithm
#1.consider only those stores with transaction in every month (July 2018 to June 2019) : if condn
#2.forming a dataframe with store numbers representing rows/records,rev,customer_visits,avg customer transactions as columns
   or metrics/attributes
#3.apply KNN algorithm to compare the three trial stores with this dataframe to obtain the respective control store
   for each of the trial store.  
#4.compare the sales revenue of the trial stores with respect to these control stores to check the difference in revenue generated
#5.find the driver of this sales revenue for the trial stores(if trial stores have higher sales during this period)'''

#Find all store informations on monthly basis and return metric information as dataframes.
#This dataframes contains store numbers as rows/records and 12 months(month number(0-11)) as columns
revenue_per_mn_per_str=[];custs_per_mn_per_str=[];avgtranspercust_per_mn_per_str=[]
dict_stores={};dict_custs={};dict_avgtrans={}
for store in set(Data_stores['STORE_NBR']):
    #print(store)
    if len(set(Data_stores[Data_stores['STORE_NBR']==store]['Month'].sort_values()))
    ==len(set(Data_stores['Month'])):
        #print(store)
        for month in set(Data_stores['Month']):
            #print(month)
            revenue_per_mn_per_str.append(Data_stores[Data_stores['STORE_NBR']==store][Data_stores['Month']==month]['TOT_SALES'].sum())
            custs_per_mn_per_str.append(Data_stores[Data_stores['STORE_NBR']==store][Data_stores['Month']==month]['LYLTY_CARD_NBR'].count())
            avgtranspercust_per_mn_per_str.append(Data_stores[Data_stores['STORE_NBR']==store][Data_stores['Month']==month].groupby('LYLTY_CARD_NBR')['TXN_ID'].count().sum()/Data_stores[Data_stores['STORE_NBR']==store][Data_stores['Month']==month]['LYLTY_CARD_NBR'].count())
        dict_stores[store]=revenue_per_mn_per_str
        dict_custs[store]=custs_per_mn_per_str
        dict_avgtrans[store]=avgtranspercust_per_mn_per_str
        revenue_per_mn_per_str=[]
        custs_per_mn_per_str=[]
        avgtranspercust_per_mn_per_str=[]
rev_perstr_allmnths=pd.DataFrame(dict_stores).T
custs_perstr_allmnths=pd.DataFrame(dict_custs).T
avgtranspercust_perstr_allmnths=pd.DataFrame(dict_avgtrans).T


##adding store number as target variable while merging to form a grand dataframe with rows as store numbers,columns as (metrics :total revenue,total customers,avg transaction) for 12 months respectively.(12*3 =36 attributes total + 1 store attribute(target))
Str_nbr=pd.DataFrame(rev_perstr_allmnths.index.values)
Stores_data=pd.concat([rev_perstr_allmnths,custs_perstr_allmnths,avgtranspercust_perstr_allmnths],axis=1)
Stores_data.index=range(len(Stores_data)) # contains store nbr in rows,(metrics :total revenue,total customers,avg transaction) for 12 months respectively.
Stores_data=pd.concat([Stores_data,Str_nbr],axis=1)
col_name=[]
for col in ['Rev','Cust','Avg_Transac_per_cust']:
    #print (col)
    for i in range(1,13):
        #print(i)
        col_name.append(col+f'{i}')
col_name=col_name+['Target']
Stores_data.columns=col_name  #final dataset with all metrics for each store number(in row) for finding control stores for each trial store.
Stores_data.head()


#Categorizing/labelling the Stores_data['Target'] attribute in a new variable 'Target_Cat' below.
Target_cat=[]
for cat in Stores_data['Target']:
        #print(i)
        Target_cat.append('Stor_'+f'{cat}')
Target_cat=pd.DataFrame(Target_cat)
Target_cat.columns=['Target_store']


##Splitting train and test data from Stores_data and Target_cat
x_train=Stores_data.drop(columns='Target',axis=1)
y_train=Target_cat


#function to measure similarity between each trial store and control stores based on the 3 attributes\metrics
#function returns metric values for a trial store for all the 3 metrics*12 months=36 values
#contains (3 attributes * 12 months =36 variables/attributes/metrics) for each trial store number(in each row) respectively,(like Rev_July2018,..,Rev_June2019,Cust_July2018,...CustJune2019,..)
def cal_rev_custs_trialstr_allmnths(store_nbr):
    rev_per_mn_per_trialstr=[]
     custs_per_mn_trialstr=[]
    avgcusttrans_per_mn_trialstr=[]
    for month in set(Data_stores['Month']):
            #print(month)
             rev_per_mn_per_trialstr.append(Data_stores[Data_stores['STORE_NBR']==store_nbr][Data_stores['Month']==month]['TOT_SALES'].sum())
             custs_per_mn_trialstr.append(Data_stores[Data_stores['STORE_NBR']==store_nbr][Data_stores['Month']==month]['LYLTY_CARD_NBR'].count())
             avgcusttrans_per_mn_trialstr.append(Data_stores[Data_stores['STORE_NBR']==store_nbr][Data_stores['Month']==month].groupby('LYLTY_CARD_NBR')['TXN_ID'].count().sum()/Data_stores[Data_stores['STORE_NBR']==store_nbr][Data_stores['Month']==month]['LYLTY_CARD_NBR'].count())
    df=pd.concat([pd.Series(rev_per_mn_per_trialstr),pd.Series(custs_per_mn_trialstr),pd.Series(avgcusttrans_per_mn_trialstr)],axis=0)
    #df.columns=['Totalrev_allmnths','Total custs_allmnths','Avg_custtransc_allmnths']
    df.index=range(len(df))
    return df

## function returns the control store for each trial store : Used KNN algorithm to find nearest neighbor/control store to each trial store
def find_control_store(store_number):
    from sklearn.neighbors import KNeighborsClassifier
    trial_store=cal_rev_custs_trialstr_allmnths(store_number)       #calling the function to cal trial store metrics.
    x_test=[trial_store]                                            # test data is taken from trial store data each time for each trial store.
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(x_train,y_train)
    control_store=neigh.predict(x_test)
    control_store=control_store[0]
    return control_store   


#running a loop that will return the respective control stores for the 3 trial stores :
trial_list=[77,86,88]
control_for_trial={}
for stor in trial_list:
    #print(stor)
    control_for_trial[stor]=find_control_store(stor)                # calling the function to find control store for a trial store.
#print('The key is the trial store and the value is the control store below\n\n:',control_for_trial)   



#Compare each control store to each trial store
Trl_perd_data=pd.read_csv('D:/IISWBM_Business_Analytics/Quantium Virtual Internship/Trial_data.csv') #selected the trial period data from 'Data_stores' dataset from Feb2019 to Apr2019 from excel and saved in a new file : 'Trial data'
Trl_perd_data.tail()
Trial_str_rev=[];Control_str_rev=[];Diff=[]
for i in range(0,len(control_for_trial)):
    #print(i)
    Trial_str_rev.append(round(Trl_perd_data[Trl_perd_data['STORE_NBR']==list(control_for_trial.keys())[i]]['TOT_SALES'].sum(),2))
    Control_str_rev.append(round(Trl_perd_data[Trl_perd_data['STORE_NBR']==int(list(control_for_trial.values())[i].split('_')[1])]['TOT_SALES'].sum(),2))
    #print(Trial_str_rev,Control_str_rev)
    Diff.append((round(Trl_perd_data[Trl_perd_data['STORE_NBR']==list(control_for_trial.keys())[i]]['TOT_SALES'].sum(),2))-(round(Trl_perd_data[Trl_perd_data['STORE_NBR']==int(list(control_for_trial.values())[i].split('_')[1])]['TOT_SALES'].sum(),2)))
Comprsn_on_sales=pd.concat([pd.Series(np.array(Trial_str_rev)),pd.Series(np.array(Control_str_rev)),pd.Series(np.array(Diff))],axis=1)
Comprsn_on_sales.columns=['Trial Store Revenue','Control Store Revenue','Difference in Revenue']
Comprsn_on_sales=Comprsn_on_sales.rename(index={0:list(control_for_trial.items())[0],1:list(control_for_trial.items())[1],2:list(control_for_trial.items())[2]})
print('\nThe key is the trial store and the value is the control store below\n\n:',control_for_trial,'\n\nThe total sales revenue for each trial store:control store pair along with their difference:\n\n',Comprsn_on_sales)

Comprsn_on_sales
pd.DataFrame(control_for_trial,)

#Also changed the date format in trial data to 'DD Month YYYY'  in excel before proceeding to check sales drivers.
#Check for sales drivers for store number 77,86,88 since they have higher revenues.
Trl_perd_data['Month']=Trl_perd_data['DATE'].apply(lambda date: date.split(' ')[1])
Trl_perd_data['Year']=Trl_perd_data['DATE'].apply(lambda date : date.split(' ')[2])
Trl_perd_data.tail()

#find customer visit during trial period in each trial store.
Custs_trial_stores={}
for i in range(0,len(control_for_trial)):
    #print(i)
    Cust_MAF=[]
    for month in set(Trl_perd_data['Month']):
        #print(month)
        Cust_MAF.append(Trl_perd_data[Trl_perd_data['STORE_NBR']==int(list(control_for_trial.keys())[i])][Trl_perd_data['Month']==month]['LYLTY_CARD_NBR'].count())
    Custs_trial_stores[int(list(control_for_trial.keys())[i])]=Cust_MAF    
Cust_visit_trl=pd.DataFrame(Custs_trial_stores,index=['March','April','February']).T


#find customer visit during trial period in each control store.
Custs_ctrl_stores={}
for i in range(0,len(control_for_trial)):
    #print(i)
    Cus_MAF=[]
    for month in set(Trl_perd_data['Month']):
        #print(month)
        Cus_MAF.append(Trl_perd_data[Trl_perd_data['STORE_NBR']==int(list(control_for_trial.values())[i].split('_')[1])][Trl_perd_data['Month']==month]['LYLTY_CARD_NBR'].count())
    Custs_ctrl_stores[int(list(control_for_trial.values())[i].split('_')[1])]=Cus_MAF    
Cust_visit_ctrl=pd.DataFrame(Custs_ctrl_stores,index=['March','April','February']).T
print('\nCustomer counts visiting each control store during the trial period:\n\n',Cust_visit_ctrl,'\n\nCustomer counts visiting each trial store during the trial period:\n\n',Cust_visit_trl)


    
'''CONCLUSIONS BASED ON OBSERVATIONS/INSIGHTS:
   1.Trial stores does perform better in terms of total sales with respect to the control stores.
   2.Trial store no. 86 has the maximum difference in terms of sales revenue when compared to its control store no. 110
   3.On checking further,we can derive an insight that trial stores 77 and 86 have more customer visits/purchasing customers
     per month when compare to the control stores,except for trial store no. 88 which though had 2 customers less in March,
     5 less in April,but had 1 customer visit more in February 2019.
     Therefore on checking overall we can infer that customer visits per month/more purchasing customers for the trial stores, is definitely a potential
     sales driver.'''
     
     