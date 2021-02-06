# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 08:24:53 2020

@author: ARITRA
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import re
pd.set_option('display.max_columns',None)

Trans=pd.read_excel('D:/IISWBM_Business_Analytics/Quantium Virtual Internship/QVI_transaction_data.xlsx')
Trans
Trans_chip=Trans[Trans['PROD_NAME'].str.contains("Chip")]

Purchs=pd.read_csv('D:/IISWBM_Business_Analytics/Quantium Virtual Internship/QVI_purchase_behaviour.csv')

'''Data preprocessing : Checking missing values, duplicate values and outliers for Transactions dataset'''

Trans_chip.isnull()
Trans_chip.isnull().values.any() # returns False for entire dataframe.
Trans_chip.isnull().any() # All columns return False,meaning no column with null values.
Trans_chip.isnull().any().any()

Trans.isnull().all

####checking for null value row wise and column wise in Trans dataframe.

#for row in range(0,len(Trans)):
#    if any(Trans.iloc[row,:].isnull()):
#        print(Trans.iloc[row,:])
        
sum(Trans_chip.isnull().any(axis=1))
sum(Trans_chip.isnull().any(axis=0))
sum([True for idx,row in Trans_chip.iterrows() if any(row.isnull())])

####checking for null values column wise.

Trans_chip.isnull().sum()

#Count the null columns/columns with null value counts.
null_cols=Trans_chip.columns[Trans.isnull().any()]
Trans_chip[null_cols].isnull().sum() #returns no value.

# checking for 'all null' columns and rows
Trans_chip[Trans_chip.isnull().any(axis=1)][null_cols]

Trans_chip.isnull().all(axis=0)
sum(Trans_chip.isnull().all(axis=1))

Trans_chip.shape
Trans_chip.head()

#colnames=Trans.columns
set(Trans_chip['PROD_NAME'])
#colnames=Trans_1.columns

#checking for outliers using boxplot

sns.boxplot(x=Trans_chip['PROD_QTY']) #outliers present.
sns.boxplot(x=Trans_chip['DATE'])
sns.boxplot(x=Trans_chip['STORE_NBR'])
sns.boxplot(x=Trans_chip['LYLTY_CARD_NBR']) # contains outliers
sns.boxplot(x=Trans_chip['TXN_ID'])
sns.boxplot(x=Trans_chip['PROD_NBR']) 
sns.boxplot(x=Trans_chip['TOT_SALES'],width=0.8) # contains outliers

#removing outliers using IQR

Trans_1=Trans_chip[['DATE','STORE_NBR','LYLTY_CARD_NBR','TXN_ID','PROD_NBR','PROD_QTY','TOT_SALES']]
Trans_1
Q1=Trans_1.quantile(0.25)
Q3=Trans_1.quantile(0.75)
IQR=Q3-Q1
print(IQR)

Trans_wo_outl=Trans_1[~((Trans_1 < (Q1-1.5*IQR)) | (Trans_1 > (Q3 + 1.5 *IQR))).any(axis=1)]
Trans_wo_outl
Trans
'''Checking missing values for Purchase dataset'''

Purchs.isnull()
Purchs.isnull().values.any() # returns False for entire dataframe.
Purchs.isnull().any() # All columns return False,meaning no column with null values.
Purchs.isnull().any().any()

Purchs.isnull().all

####checking for null value row wise  and column wise in Trans dataframe.

#for row in range(0,len(Trans)):
#    if any(Trans.iloc[row,:].isnull()):
#        print(Trans.iloc[row,:])


sum(Purchs.isnull().any(axis=1)) #returns 0
sum(Purchs.isnull().any(axis=0)) # returns 0

sum([True for idx,row in Purchs.iterrows() if any(row.isnull())])

####checking for null values column wise.
Purchs.isnull().sum()

#Count the null columns/columns with null value counts.
null_cols=Purchs.columns[Purchs.isnull().any()]
null_cols
Purchs[null_cols].isnull().sum() #returns no value.

# checking for 'all null' columns and rows.
Purchs[Purchs.isnull().any(axis=1)][null_cols]

Purchs.isnull().all(axis=0)
sum(Purchs.isnull().all(axis=1))


Trans_wo_outl.duplicated().any()
Purchs.duplicated().any()


'''Merging of the two datasets to get final data for analysis'''

Trans_wo_outl
Purchs

Final_data=pd.merge(Trans_wo_outl,Purchs,on="LYLTY_CARD_NBR",how="inner")
Final_data.columns

Final_data.duplicated().any()


'''Customer Analytics'''


Final_data.PROD_NBR

#list(set(Final_data['LIFESTAGE']))
#Final_data[Final_data['LIFESTAGE']=='RETIREES']['TOT_SALES'].sum().round(2)

'''total sales contributed over categorical column wise in the Final_data.Function build for same :'''

#total_sales_lifst={cat:Final_data[Final_data['LIFESTAGE']==cat]['TOT_SALES'].sum().round(2) for cat in list(set(Final_data['LIFESTAGE']))}   
#total_sales_lifst   


def cal_totalsales_catwise_forcolumn(column,response):
    total_sales_catwise={cat:Final_data[Final_data[column]==cat][response].sum().round(2) for cat in list(set(Final_data[column]))}
    return total_sales_catwise

sales_dict={}
for col in Final_data.columns:
    #if Final_data[col].dtype.name == 'category': 
    if Final_data[col].dtype != 'int64' and Final_data[col].dtype != 'float64':
        #print(col)
        l=cal_totalsales_catwise_forcolumn(col,'TOT_SALES')
        #print(col+' '+'wise total sales contribution')
        #print (l)
        sales_dict[col+' '+'wise total sales contribution']=l

sales_dict    #sales contribution across categorical attributes
sales_by_affluence=list(sales_dict.items())[1][1]
sales_by_affluence=pd.DataFrame.from_dict(sales_by_affluence).T
sales_by_affluence

sns.barplot(x="Customers by Affluence", y="Total Sales contribution", data=sales_by_affluence,palette="Blues_d")
#NEW FAMILIES in LIFESTAGE attribute contributes minimum,OLDER SINGLES/COUPLES contributes maximum
#Premium in PREMIUM_CUSTOMER attribute contributes minimum,Mainstream contributes maximum
#When total sales compared,PREMIUM_CUSTOMER wise checking returns more total sales amount
#in comparison to LIFESTAGE attribute wise checking


Final_data.TOT_SALES.plot(kind='hist',bins=int(180/10))

sns.distplot(Final_data['TOT_SALES'], hist=True, kde=True, 
             bins=int(180/10), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.hist(Final_data['TOT_SALES'], bins = int(180/10),
             color = 'blue', edgecolor = 'black')

'''Customer counts each categorical attribute wise, visulaized'''
Final_data['PREMIUM_CUSTOMER'].value_counts().plot(kind='bar')
Final_data['LIFESTAGE'].value_counts().plot(kind='bar')
Final_data.head()

#From viz it is evident that, the sales frequency (number of people catering to a particular sales amount)
#occassionally rises and falls. It falls flat between 4-5.5, whereas it takes the highest at around 7.5-8.0,
#also between 6-6.5,6.5-7.0,8.5-9.0 the sales frequency is high enough.'''

#Since PREMIUM_CUSTOMER wise total sales contribution is higher, hence we shall segment customers
#by this category to find the purchasing behavour/pattern
#Mainstr=Final_data[Final_data['PREMIUM_CUSTOMER']=="Mainstream"]

'''Customer segmentation based on sales value frequency(value where more sales/customer transactions
occured) as stated above, as well as based on overall data sales frequency check'''
    
Sales_1=Final_data[(Final_data['TOT_SALES'].between(7.5,8.0,inclusive=True))]
Sales_1.groupby("PREMIUM_CUSTOMER")['TOT_SALES'].sum() #mainstream, then budget
Sales_1['PREMIUM_CUSTOMER'].value_counts() #mainstream,then budget
Sales_1.groupby("LIFESTAGE")['TOT_SALES'].sum() #older singles/couples,then retirees followed by older families,then young families
Sales_1['LIFESTAGE'].value_counts() #older singles/couples, retirees, older families,then young families
Sales_1.groupby(['PREMIUM_CUSTOMER','LIFESTAGE'])['TOT_SALES'].sum() #mainstream young single/couples,
#Budget older families, mainstream retirees, budget older singles/couples,mainstream older singles/couples,
#premium older singles/couples, then budget young families

Sales_2=Final_data[(Final_data['TOT_SALES'].between(6.0,7.0,inclusive=True))]
Sales_2.groupby("PREMIUM_CUSTOMER")['TOT_SALES'].sum() #mainstream > budget
Sales_2['PREMIUM_CUSTOMER'].value_counts() #mainstream > budget
Sales_2.groupby("LIFESTAGE")['TOT_SALES'].sum() #older singles/couples > older families > retirees > young families
Sales_2['LIFESTAGE'].value_counts() #older singles/couples
Sales_2.groupby(['PREMIUM_CUSTOMER','LIFESTAGE'])['TOT_SALES'].sum()
#budget older families > mainstream retirees > budget young families > premium older singles/couples
# > mainstream older singles/couples > mainstream young singles/couples > mainstream older families.

Sales_3=Final_data[(Final_data['TOT_SALES'].between(8.5,9.0,inclusive=True))]
Sales_3.groupby("PREMIUM_CUSTOMER")['TOT_SALES'].sum() #mainstream > budget
Sales_3['PREMIUM_CUSTOMER'].value_counts() #mainstream > budget
Sales_3.groupby("LIFESTAGE")['TOT_SALES'].sum() #older singles/couples > retirees > older families > young families
Sales_3['LIFESTAGE'].value_counts() #older singles/couples > retirees > older families > young families
Sales_3.groupby(['PREMIUM_CUSTOMER','LIFESTAGE'])['TOT_SALES'].sum()
#mainstream young singles/couples > mainstream retirees > budget older families > mainstream older singles/couples,
# > budget older singles/couples > budget young families

Final_data.groupby("PREMIUM_CUSTOMER")['TOT_SALES'].sum() #mainstream > budget
Final_data['PREMIUM_CUSTOMER'].value_counts() #mainstream > budget
Final_data.groupby("LIFESTAGE")['TOT_SALES'].sum() #older singles/couples > older families > retirees > young families
Final_data['LIFESTAGE'].value_counts() #older singles/couples,retirees > older families > young families
Final_data.groupby(['PREMIUM_CUSTOMER','LIFESTAGE'])['TOT_SALES'].sum()

a=(Final_data.groupby(['PREMIUM_CUSTOMER','LIFESTAGE'])['LYLTY_CARD_NBR'].count()/Final_data.groupby(['PREMIUM_CUSTOMER','LIFESTAGE'])['LYLTY_CARD_NBR'].count().sum()) *100
a=pd.DataFrame(a)
#a.T.to_csv('D:\plot.csv')
a.plot.bar(stacked=True)
#Budget older families > mainstream retirees > budget young families > mainstream young singles/couples
#> mainstream older singles/couples > budget older singles/couples > premium oler singles/couples


'''Final segmentation for which purchasing behaviour will be checked'''
#mainstream retirees
#mainstream young singles/couples
#mainstream older singles/couples
#budget older families
#budget young families
#budget older singles/couples
#premium older singles/couples


'''Find purchasing behaviour/pattern for each segment'''

##susetting the datasets into Mnstr,Bugt,Prmum based on 'PREMIUM_CUSTOMERS' with most sales contributing 'LIFESTAGE' values.

Final_data.head()
Mnstr_1=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Mainstream') & (Final_data['LIFESTAGE']=='YOUNG SINGLES/COUPLES')]
Mnstr_2=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Mainstream') & (Final_data['LIFESTAGE']=='RETIREES')]
Mnstr_3=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Mainstream') & (Final_data['LIFESTAGE']=='OLDER SINGLES/COUPLES')]

Mnstr=pd.concat([Mnstr_1,Mnstr_2,Mnstr_3],axis=0)

Bugt_1=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Budget') & (Final_data['LIFESTAGE']=='OLDER SINGLES/COUPLES')]
Bugt_2=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Budget') & (Final_data['LIFESTAGE']=='OLDER FAMILIES')]
Bugt_3=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Budget') & (Final_data['LIFESTAGE']=='YOUNG FAMILIES')]

Bugt=pd.concat([Bugt_1,Bugt_2,Bugt_3],axis=0)

Prmum=Final_data[(Final_data['PREMIUM_CUSTOMER']=='Premium') & (Final_data['LIFESTAGE']=='OLDER SINGLES/COUPLES')]


###checking for Mnstr### :
    
    
 #checking by prod qty :   
Mnstr.groupby('PROD_QTY').sum() #prod quantity is 2

#checking by prod nbr :
    
Mnstr.PROD_NBR.value_counts()# prod number 30 contributes 5.5 % of the total number of transactions on any unique product.
Perc_contrib=(Mnstr.PROD_NBR.value_counts().max()/Mnstr.PROD_NBR.value_counts().sum()) * 100
print(f'{round((Perc_contrib),2)}'+"%") # 5.5 %

Mnstr.groupby("PROD_NBR").sum().max() #also prod nbr 30 contributes the maximum sales in this category(Mainstream)
Mnstr[Mnstr["PROD_NBR"]==30]["TOT_SALES"].sum()

#checking by store nbr :

Mnstr.STORE_NBR.value_counts() # store number 4 gets repeated max times,so most transactions occur in this store.
Mnstr.groupby("STORE_NBR").sum().max() #store number 4 gives the max sales in this category.
Mnstr[Mnstr["STORE_NBR"]==4]["TOT_SALES"].sum() 
Mnstr[Mnstr["STORE_NBR"]==4].groupby('PROD_NBR').sum() #prod nbr giving max sales in the seq 33>30>77 in store number 4

###checking for Bugt###

#checking by prod qty :   
Bugt.groupby('PROD_QTY').sum() #prod quantity is 2
set(Bugt.PROD_QTY)

#checking by prod nbr :
    
Bugt.PROD_NBR.value_counts() # prod number 30 contributes 4.7 % of the total number of transactions on any unique product.
Perc_contrib_2=(Bugt.PROD_NBR.value_counts().max()/Bugt.PROD_NBR.value_counts().sum()) * 100
print(f'{round((Perc_contrib_2),2)}'+"%")

Bugt.groupby("PROD_NBR").sum().max() #also prod nbr 30 contributes the maximum sales in this category(Budget)
Bugt[Bugt["PROD_NBR"]==30]["TOT_SALES"].sum()

#checking by store nbr :

Bugt.STORE_NBR.value_counts() # store number 43,128 gets repeated max times,so most transactions occur in these 2 stores under this category
Bugt.groupby("STORE_NBR").sum().max()
Bugt[Bugt["STORE_NBR"]==43]["TOT_SALES"].sum()
Bugt[Bugt["STORE_NBR"]==128]["TOT_SALES"].sum() #also store 128 contributes max sales under this category(Budget)

Bugt[Bugt["STORE_NBR"]==128].groupby('PROD_NBR').sum() # prod nbr 47>100>30 contributes to max sales in store 128. 
Bugt[Bugt["STORE_NBR"]==43].groupby('PROD_NBR').sum() #prod nbr 77>12>2>28,44>61 gives max sales in store 43


###checking for prmum###

#checking by prod qty :   
Prmum.groupby('PROD_QTY').sum() #prod quantity is 2
set(Prmum.PROD_QTY)

#checking by prod nbr :
    
Prmum.PROD_NBR.value_counts() # prod number 93 contributes 5.9 % of the total number of transactions on any unique product.
Perc_contrib_3=(Prmum.PROD_NBR.value_counts().max()/Prmum.PROD_NBR.value_counts().sum()) * 100
print(f'{round((Perc_contrib_3),2)}'+"%")

Prmum.groupby("PROD_NBR").sum().max() #also prod nbr 30 contributes the maximum sales,followed by 93 in this category(PREMIUM)
Prmum[Prmum["PROD_NBR"]==30]["TOT_SALES"].sum()

#checking by store nbr :

Prmum.STORE_NBR.value_counts() # store number 67 gets repeated max times,so most transactions occurs in this store under this category(PREMIUM)
Prmum.groupby("STORE_NBR").sum()
Prmum[Prmum["STORE_NBR"]==67]["TOT_SALES"].sum() #also store 128 contributes max sales under this category(Budget)
Prmum[Prmum["STORE_NBR"]==67].groupby('PROD_NBR').sum() # prod nbr 40,>111>42 contributes to max sales in store 67.

Final_data.to_csv('D:/IISWBM_Business_Analytics/Quantium Virtual Internship/Final_dat.csv')


'''Our Initial findings at a glance ::
NOTE : We have checked the TOT_SALES distribution to understand the sales values with more frequency,hence most transactions,
and from these regions as already shown above, we have found out the most contributing customer segments.
1.When checked on overall data 'Final_data', 'PREMIUM_CUSTOMER' :'Mainstream' contributes to most of the total sales
  followed by 'Budget'
  And when checked on 'LIFESTAGE' : 'OLDER SINGLES/COUPLES','RETIREES','OLDER FAMILIES','YOUNG FAMILIES' contribute
  significantly on the sales.
  Hence we have tried extracting the possible exhaustive combinations of 'PREMIUM CUSTOMER' and 'LIFESTAGE' to 
  segment the customers into categories contributing to maximum possible total sales.

2.Gathered initial insights based on the following customer segmentations on 'PREMIUM_CUSTOMER' & 'LIFESTAGE' attributes
combined :
Please note  : that 'LIFESTAGE' categorical values are selected based on their contribution on total sales,
as already checked and displayed in the code above,
    
#mainstream retirees
#mainstream young singles/couples
#mainstream older singles/couples
#budget older families
#budget young families
#budget older singles/couples
#premium older singles/couples

2.Grouped the 'LIFESTAGE' values selected into the 'PREMIUM_CUSTOMER' attribute value wise data.
 Hence we get three datasets based on this final segmentation : 'Mnstr'(for all 'Mainstream'),
 'Bugt'(for all BUDGET),'Prmum'(for all PREMIUM_CUSTOMER)
 Therefore we check the purchasing behaviour on these 3 segmented customer transaction data to draw some pattern
 on transaction.
 
 A.)When 'PREMIUM_CUSTOMER':Mainstream,'LIFESTAGE':[Young singles/couples,Retirees,Older singles/couples], then
 > product quantity bought is always 2

 > on checking by product number, we get that
   product number 30 contributes the maximum( 5.5 % )of the total number of transactions on any unique product.
   Also product number 30 contributes the maximum sales here.

 > on checking by store number, we get that
   store number 4 gets repeated maximum times,so maximum transactions occur in this store.
   store number 4 gives the max sales in this category.
   product numbers contributing to maximum sales in the seq 33>30>77(increasing to decreasing) in store number 4


 B.)When 'PREMIUM_CUSTOMER':Budget,'LIFESTAGE':[Older Families,Young Families,Older singles/couples], then
 > product quantity bought is always 2

 > on checking by product number, we get that
   product number 30 contributes the maximum( 4.7 % )of the total number of transactions on any unique product.
   Also product number 30 contributes the maximum sales here as well.

 > on checking by store number, we get that
   store number 43 and 128 gets repeated maximum times,so maximum transactions occur in these stores.
   store number 128 contributes to the maximum sales in this category.
   product numbers contributing to maximum sales in the seq 47>100>30(increasing to decreasing) in store number 128
   product numbers contributing to maximum sales in the seq 77>12>2>28,44>61(increasing to decreasing) in store number 43



 C.)When 'PREMIUM_CUSTOMER':Premium,'LIFESTAGE':[Older singles/couples], then
 > product quantity bought is always 2

 > on checking by product number, we get that
   product number 93 contributes the maximum( 5.9 % )of the total number of transactions on any unique product.
   Also product number 30 contributes the maximum sales here as well,followed by product number 93(the maximum repeated product in the transactions)

 > on checking by store number, we get that
   store number 67 gets repeated maximum times,so maximum transactions occur in this store.
   store number 128 contributes to the maximum sales in this category.
   product numbers contributing to maximum sales in the seq 40,>111>42 (increasing to decreasing) in store number 67


3.Therefore one common insight derived from all the above 3 segments of data :
  product number :30(product name : Doritos Corn Chips  Cheese Supreme 170g ),lifestage :older singles/couples, product quantity : 2, should be more focussed on.

4.Some of the product names that comes up in terms of the maximum total sales contribution or maximum repetitions(transactions or sold)
Doritos Corn Chip Southern Chicken 150g : 93
Doritos Corn Chips  Original 170g : 47
Doritos Corn Chips  Nacho Cheese 170g : 77
Doritos Corn Chip Mexican Jalapeno 150g : 42

Smiths Crinkle Cut  Chips Chicken 170g : 61
Smiths Crinkle Cut  Chips Chs&Onion170g : 100
Smiths Chip Thinly  Cut Original 175g : 111

Thins Potato Chips  Hot & Spicy 175g : 28
Thins Chips Light&  Tangy 175g : 44
Thins Chips Seasonedchicken 175g : 40

Natural Chip Co  Tmato Hrb&Spce 175g : 12
Cobs Popd Swt/Chlli &Sr/Cream Chips 110g : 33



5. From this 3 brands which tends to be more likely sold : Doritos,Smiths,Thins. So special focus
should be given to products from these brands.
'''

