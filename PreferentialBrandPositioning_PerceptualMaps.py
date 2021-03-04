# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:32:37 2021

@author: ARITRA
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('Display.max_columns',None)

df=pd.read_excel('D:/IISWBM_Business_Analytics/PGDM_PROJECT/Survey1.xlsx')
df


####Data Preprocessing####
#checking data summary
df.head()
df.describe()
df.shape
df.info() #since no data type category is observed, hence we can directly impute.No need to encode.

#Checking for NULL/Missing values shows missing values are present.
df.isnull()
df.isnull().values.any() # returns True if df contains any null value.
df.isnull().any() # Column wise check for null values.
df.isnull().all
df.isnull().sum() #column wise sum of missing values


#Count the null columns/columns with null values.
null_cols=df.columns[df.isnull().any()]
null_cols
df[null_cols].isnull().sum() #returns each column with sum of the missing values present in them.

# extracting records/rows for which it returns True. 
df[df.isnull().any(axis=1)]
#df[df.isnull().any(axis=1)][null_cols]

#check proportion of records with missing values
#We won't be able to remove/impute an entire record due to nature of questions.
prop_missing=(np.count_nonzero(df.isnull().any(axis=1).values)/len(df)) *100
print(f'Records with missing values are {prop_missing} % of total records')

#sns.boxplot(x=df.iloc[:,15], data=df,palette="Set3")

#Need to check for duplicate values
df.duplicated().any() # No duplicate records obtained.

#Descriptive Statistics to understand the structure of the data.
#MAKE THE HISTOGRAM,HEATMAPS,AR PLOTS HERE...

#################################################################################################################################################

# Omitting missing values from Age, Gender, Income and Years of Mobile phone usage
df.iloc[:,0] = df.iloc[:,0].fillna(value = "25 - 34 Years", axis = 0)
df.iloc[:,1] = df.iloc[:,1].fillna(value = "Male", axis = 0)
df.iloc[:,2] = df.iloc[:,2].fillna(value = "Up to Rs. 2,50,000", axis = 0)
df.iloc[:,15] = df.iloc[:,15].fillna(value = df.iloc[:,15].median(), axis = 0)

#Checking distribution of Age Groups

a = df.groupby(df.iloc[:,0]).count().iloc[:,0]
a

#Visualizing Age Group distribution
a.plot.barh(label="", title="DISTRIBUTION OF AGE GROUP")
plt.xlabel("Age Group")
plt.ylabel("No. of Respondents")
plt.grid()
plt.show()

# this can also be done by
#labels =  ['25 - 34 Years', '16 - 24 Years', 'Above 65 Years',
#       '35 - 44 Years', '45 - 54 Years', 'Less than 16 Years']
#frequency = [128, 27, 2, 5, 3, 1]
#def plot_bar_x():
#    plt.barh(labels, frequency)
#    plt.xlabel('No of Respondents', fontsize=10)
#    plt.ylabel('Age Groups', fontsize=10)
#    plt.xticks(labels, fontsize=10)
#    plt.title('Distribution of Age Groups')
#    plt.show()
#plot_bar_x()

#Since age group 25-34 Years and 16-24 Years are predominant, hence focus our analysis for age group 16-34 Years and 25-34 Years

#subsetting the data basis age group 16-34 Years
df1 = df[(df["Under which age group do you fall?"] == '16 - 24 Years') | (df["Under which age group do you fall?"] == '25 - 34 Years')]
df1

#Checking Distribution of Gender in the selected Age Group
g = df1.groupby(df1.iloc[:,1]).count().iloc[:,1]
g

#Visualizing Gender distribution
g.plot.pie(label="", title="DISTRIBUTION OF GENDER",
           startangle=90, autopct='%1.1f%%',
           colors = ["#E13F29", "#D69A80"] )
                     
g.plot.barh(label="", title="DISTRIBUTION OF GENDER")
plt.xlabel("Gender")
plt.ylabel("No. of Respondents")
plt.grid()

#Checking Distribution of Income Group in the selected Age Group
i = df1.groupby(df1.iloc[:,2]).count().iloc[:,2]
i

#Visualizing Income Group distribution
i.plot.barh(label="", title="DISTRIBUTION OF INCOME GROUP")
plt.xlabel("Income Group")
plt.ylabel("No. of Respondents")
plt.grid()
i.plot.pie(label="", title="DISTRIBUTION OF INCOME GROUP",
           autopct='%1.1f%%')

#Checking Distribution of Years of Mobile Phone Usage
mb = df1.groupby(df1.iloc[:,15]).count().iloc[:,15]
mb

#Visualizing Years of Mobile Phone Usage distribution
mb.plot.barh(label="",title="DISTRIBUTION OF YEARS OF MOBILE PHONE USAGE")
plt.xlabel("No. of Respondents")
plt.ylabel("Years of Mobile Phone Usage")
plt.grid()
mb.plot.pie(label="", title="DISTRIBUTION OF YEARS OF MOBILE PHONE USAGE",
           autopct='%1.1f%%')

#Checking Distribution of Previously Used Phone Brands
Apple = df1.groupby(df1.iloc[:,3]).count().iloc[:,3].values
Samsung = df1.groupby(df1.iloc[:,4]).count().iloc[:,4].values
OnePlus = df1.groupby(df1.iloc[:,5]).count().iloc[:,5].values
Motorola = df1.groupby(df1.iloc[:,6]).count().iloc[:,6].values
LG = df1.groupby(df1.iloc[:,7]).count().iloc[:,7].values
Nokia = df1.groupby(df1.iloc[:,8]).count().iloc[:,8].values
Huawei = df1.groupby(df1.iloc[:,9]).count().iloc[:,9].values
Xiaomi = df1.groupby(df1.iloc[:,10]).count().iloc[:,10].values
Oppo = df1.groupby(df1.iloc[:,11]).count().iloc[:,11].values
Vivo = df1.groupby(df1.iloc[:,12]).count().iloc[:,12].values
RealMe = df1.groupby(df1.iloc[:,13]).count().iloc[:,13].values
RealMe
Index = ["Apple", "Samsung", "OnePlus", "Motorola", "LG", "Nokia", "Huawei", 
            "Xiaomi", "Oppo", "Vivo", "RealMe"]
values = [Apple, Samsung, OnePlus, Motorola, LG, Nokia, Huawei, 
            Xiaomi, Oppo, Vivo, RealMe]
values
pub = pd.DataFrame(values, Index)
pub.columns=['Frequency of Each Brand Used']

#Visualizing Distribution of Previously Used Phone Brands
pub.plot.barh(label="", title="DISTRIBUTION OF PREVIOUSLY USED PHONE BRANDS")
plt.xlabel("No of Responses")
plt.ylabel("Mobile Phone Brands")
plt.grid()

pub.plot.pie(label="", title="DISTRIBUTION OF PREVIOUSLY USED PHONE BRANDS",
           startangle=90, autopct='%1.1f%%', figsize = (10,10),
           colors = ["#E13F29", "#D69A80", "#D63B59", "#AE5552", "#CB5C3B", "#EB8076", "#96624E"], subplots = True)
plt.tight_layout()

#Checking Distribution of Shifting Reasons
Fast_processing = df1.groupby(df1.iloc[:,22]).count().iloc[:,22].values
Operating_system = df1.groupby(df1.iloc[:,23]).count().iloc[:,23].values
Storage = df1.groupby(df1.iloc[:,24]).count().iloc[:,24].values
Camera_Quality = df1.groupby(df1.iloc[:,25]).count().iloc[:,25].values
Presence_of_multiple_apps = df1.groupby(df1.iloc[:,26]).count().iloc[:,26].values
Increase_in_your_Budget = df1.groupby(df1.iloc[:,27]).count().iloc[:,27].values
Index1 = ["Fast processing", "Operating system", "Storage", "Camera Quality", "Presence of multiple apps", "Increase in your Budget"]
values1 = [Fast_processing, Operating_system, Storage, Camera_Quality, Presence_of_multiple_apps, Increase_in_your_Budget]
shift = pd.DataFrame(values1, Index1)
shift

#Visualizing Distribution of Shifting Reasons
shift.plot.barh(label="", title="DISTRIBUTION OF SHIFTING REASONS")
plt.xlabel("No of Responses")
plt.ylabel("Reasons of Shift")
plt.grid()

shift.plot.pie(label="", title="DISTRIBUTION OF SHIFTING REASONS",
           startangle=90, autopct='%1.1f%%', figsize = (10,10),
           colors = ["#E13F29", "#D69A80", "#D63B59", "#AE5552", "#CB5C3B", "#EB8076", "#96624E"], subplots = True)
plt.tight_layout()


######################################################################################################################################################################################


#Reasons of Shift
#df3=df.iloc[:,22:30]
#set(df3.iloc[:,-1])
#df3.Reasons_Merged.value_counts() #counts of combination of reasons responsible.


#Selecting attributes with similarity ratings for Perceptual Maps
df1=df1.iloc[:,30:155]
len(df1.columns)
#checking for null values in subset rating data df1
df1.isnull().any(axis=1)
df1.isnull().sum()
#df1.apply(lambda x : x.astype(np.object))
prop_missing=(np.count_nonzero(df1.isnull().any(axis=1).values)/len(df1)) *100
print(f'Records with missing values are {prop_missing} % of total records')
#df1.iloc[:,-1]

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
df_enc=df1
cat_cols=df_enc.columns.values
encoder = OrdinalEncoder()

def encode_each_column(colum):     #function to encode a categorical column before imputing it.
    #removes na values from column
    nonul=np.array(colum.dropna())
    #reshapes colum for encoding
    impute_reshpe=nonul.reshape(-1,1)
    encode_catcol=encoder.fit_transform(impute_reshpe)
    #Assign back encoded values to non null values
    colum.loc[colum.notnull()]=np.squeeze(encode_catcol)
    return colum

for  c in cat_cols:
    #print(cat_cols[c])
    encode_each_column(df_enc[c])
imputer=KNNImputer(n_neighbors=4)
df_imp=imputer.fit_transform(df_enc)
df_imputed=pd.DataFrame(np.round(df_imp))
df_imputed.columns=df1.columns.values

df_imputed.head() #No more missing values found in the similarity ratings data


prop_missing=(np.count_nonzero(df_imputed.isnull().any(axis=1).values)/len(df_imputed)) *100
print(f'Records with missing values are {prop_missing} % of total records')

pairs=['Apple - Samsung','Apple - Xiaomi','Apple - Oppo','Apple - RealMe','Samsung - Xiaomi','Samsung - Oppo','Samsung - RealMe','Xiaomi - Oppo',
          'Xiaomi - RealMe','Oppo - RealMe']

dimensions = ['Computing Speed (Smoothness)','Battery Backup','Memory & Storage','Display & Camera Quality','After Sales Support & Warranty/Claims']

#len(pairs)
#len(dimensions)

#creating a new dataframe to store pairs as rows and dimensions as columns.
df_mds=pd.DataFrame(index = dimensions,columns = pairs)
df_mds=df_mds.fillna(0)
df_mds.head()
df_mds.shape

#a=df['Rate the following pairs of brands from most similar (1) to least similar (10) according to your perception in terms of Memory & Storage.- Apple - Oppo'].value_counts()
#b=a.to_dict()
#keymax=max(b,key=b.get)
#keymax

#df1[f'Rate the following pairs of brands from most similar (1) to least similar (10) according to your perception in terms of {dimensions[1]}.- {pairs[0]}']        

for d in range(len(dimensions)):
    #print(dimensions[d])
    for p in range(len(pairs)):
        #print(dimensions[d],pairs[p])
        freq_ratings=df_imputed[f'Rate the following pairs of brands from most similar (1) to least similar (10) according to your perception in terms of {dimensions[d]}.- {pairs[p]}'].value_counts()
        rating_counts=freq_ratings.to_dict()
        rating_maxcounts=max(rating_counts,key=rating_counts.get)
        df_mds.iloc[d,p]=rating_maxcounts

df_mds.head()       

DF_MDS=df_mds.T

#DF_MDS.iloc[:,1].astype(np.int64)
for i in range(DF_MDS.shape[1]):
    #print(i)
    DF_MDS.iloc[:,i]=DF_MDS.iloc[:,i].astype(np.int64)
    
#df_mds2=[DF_MDS.iloc[:,i].astype(np.int64) for i in range(DF_MDS.shape[1])]
#df_mds2

DF_MDS

#####PERCEPTUAL MAPS#######

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
#levels=['Apl-Samsung','Apl-Xiomi','Apl-Oppo','Apl-RealMe','Samgsung-Xiaomi','Samsung-Oppo','Samsung-RealMe','Xiaomi-Oppo','Xiaomi-RealMe','Oppo-RealMe']
#colors=['b','g','r','c','m','y','k','#eeefff','#feffb3','#777777']

#2D plot of Perceptual Map : Smootheness,Memory
plt.figure(figsize=(10,8))
plt.scatter(DF_MDS.iloc[:,0],DF_MDS.iloc[:,2],facecolors='none',edgecolors='none')
labels=['Apl-Samsung','Apl-Xiomi','Apl-Oppo','Apl-RealMe','Samgsung-Xiaomi','Samsung-Oppo','Samsung-RealMe','Xiaomi-Oppo','Xiaomi-RealMe','Oppo-RealMe']
#colors=['b','g','r','c','m','y','k','#eeefff','#feffb3','#777777']
for label,x,y in zip(labels,DF_MDS.iloc[:,0],DF_MDS.iloc[:,2]):
    plt.annotate(label, (x,y))
plt.xlabel('Computing Speed (Smoothness)')
plt.ylabel('Memory & Storage')
plt.title('2D Perceptual Map')
plt.show()
#plt.savefig()   

DF_MDS
#3D plot of Perceptual Map - Display Quality,Battery,AfterSales
fig=plt.figure(figsize=(10,7))
#ax=plt.axes(projection='3d')
ax=Axes3D(fig)
#ax.plot3D(DF_MDS.iloc[:,0],DF_MDS.iloc[:,1],DF_MDS.iloc[:,2])
for i in range(len(DF_MDS)):
    ax.scatter(DF_MDS.iloc[i,3],DF_MDS.iloc[i,1],DF_MDS.iloc[i,4], cmap='Greens');
    #ax.text(DF_MDS.iloc[i,0],DF_MDS.iloc[i,1],DF_MDS.iloc[i,2],'%s' % (str(i)), size=20, zorder=1,color='k')
    ax.text(DF_MDS.iloc[i,3],DF_MDS.iloc[i,1],DF_MDS.iloc[i,4],labels[i],size=12, zorder=1,color='k')
    #for label,x,y,z in zip(labels,DF_MDS.iloc[:,0],DF_MDS.iloc[:,1],DF_MDS.iloc[:,2]):
    #plt.annotate(label, (x,y,z))
ax.set_xlabel('Display & Camera Quality',linespacing = 3.1)
ax.set_ylabel('Battery Backup',linespacing = 3.4)
ax.set_zlabel('After Sales Support & Warranty/Claims',linespacing = 3.4);
plt.title('3D Perceptual Map')
plt.show()

#3D plot of Perceptual Map - Memory,Smootheness,AfterSales
fig=plt.figure(figsize=(10,7))
#ax=plt.axes(projection='3d')
ax=Axes3D(fig)
#ax.plot3D(DF_MDS.iloc[:,0],DF_MDS.iloc[:,1],DF_MDS.iloc[:,2])
for i in range(len(DF_MDS)):
    ax.scatter(DF_MDS.iloc[i,2],DF_MDS.iloc[i,0],DF_MDS.iloc[i,4], cmap='Greens');
    #ax.text(DF_MDS.iloc[i,0],DF_MDS.iloc[i,1],DF_MDS.iloc[i,2],'%s' % (str(i)), size=20, zorder=1,color='k')
    ax.text(DF_MDS.iloc[i,2],DF_MDS.iloc[i,0],DF_MDS.iloc[i,4],labels[i],size=12, zorder=1,color='k')
    #for label,x,y,z in zip(labels,DF_MDS.iloc[:,0],DF_MDS.iloc[:,1],DF_MDS.iloc[:,2]):
    #plt.annotate(label, (x,y,z))
ax.set_xlabel('Memory & Storage',linespacing = 3.1)
ax.set_ylabel('Computing Speed (Smoothness)',linespacing = 3.4)
ax.set_zlabel('After Sales Support & Warranty/Claims',linespacing = 3.4);
plt.title('3D Perceptual Map')
plt.show()




#Analysis and Visualization of brand by Rank Order Preferences
#len(df.columns)    
df2=df.iloc[:,155:163]
df2.isnull().any() # Column wise check for null values.
df2.isnull().all
df2.isnull().sum() #No null values/missing values obtained in any column of Brand Preference Rank data.
#l=df2.columns.tolist()

#extracting Brand names from overall column names.
def get_names_from_cols(cols):
    l=cols.tolist()
    brand_names=[]
    for j in range(len(l)):
        #print(j)
        brand=l[j].split('- ')[1]
        brand_names.append(brand)
    return brand_names

brands=get_names_from_cols(df2.columns)    
brands 


brand_ranks=pd.DataFrame(index =[i for i in range(1,len(brands)+1)],columns = brands)
brand_ranks=brand_ranks.fillna(0)
brand_ranks
'''
brand_ranks.iloc[:,0]
a=df2.iloc[:,0].value_counts().sort_index()
a.index
brand_ranks.index
b=a.to_dict()
b.get(1)

temp=pd.DataFrame(np.arange(11))
temp.iloc[:,0]
temp2=[]
#for i in (a.index.astype(np.int64)):
    #print(i)
for j in (brand_ranks.index):
    #print(j)
    if j in (a.index.astype(np.int64)):
        #temp.iloc[j,0]=b.get(i)
        temp.iloc[j,0]=b.get(j)
    else:
        temp.iloc[j,0]=0
        #break    
     
temp.drop(temp.index[0],inplace=True)
temp.drop(temp.index[9],inplace=True)
a
temp
brand_ranks.iloc[:,0]

for q in range(len(brand_ranks)):
    #print(q)
    brand_ranks.iloc[q,0]=temp.iloc[q,0]

brand_ranks
max_rank=max(rank_counts_dict,key=rank_counts_dict.get)
max_rank
'''


def rank_counts(col):           #function takes each brand ranking column from dataset df2,counts value for each rank in the column,maps the counts to each rank for each brand in a new dataframe
    a=col.value_counts().sort_index()
    b=a.to_dict()
    temp=pd.DataFrame(np.arange(11))
    #temp.iloc[:,0]
    for j in (brand_ranks.index):
        if j in (a.index.astype(np.int64)):
            temp.iloc[j,0]=b.get(j)
        else:
            temp.iloc[j,0]=0
    temp.drop(temp.index[0],inplace=True)
    temp.drop(temp.index[9],inplace=True)    
    return temp

for k in range(len(df2.columns)):
    #print(k)
    brand_ranks[brands[k]]=rank_counts(df2.iloc[:,k]) 
brand_ranks
brand_ranks.T.plot(kind='bar',stacked=True,title='Ranking proprotions for each Brand',figsize=(12,10))

BrandRanks=brand_ranks.T
BrandRanks
HighestRank_eachBrand={}
for ind in range(len(BrandRanks)):
    #print(ind)
    HighestRank_eachBrand[BrandRanks.columns[ind]]=[BrandRanks.iloc[:,ind].idxmax(axis=0),BrandRanks.iloc[:,ind].max()]

HighestRank_eachBrand
Rankings=pd.DataFrame(list(HighestRank_eachBrand.values()))
R2=Rankings.set_index(Rankings.iloc[:,0])
R2=R2.drop([0],axis=1)
R2['Ranks']=HighestRank_eachBrand.keys()
R2.rename(columns={1:'Highest votes'},inplace=True)
Brandwise_FinalRankings=R2

Brandwise_FinalRankings
#Brandwise_FinalRankings.to_csv('D:/IISWBM_Business_Analytics/PGDM_PROJECT/brand_rankings.csv') #final preference data rank wise
 

#2D plot of Preference Rankings
plt.figure(figsize=(10,8))
plt.plot(Brandwise_FinalRankings.iloc[:,1],Brandwise_FinalRankings.iloc[:,0])
labels=Brandwise_FinalRankings.index
#colors=['b','g','r','c','m','y','k','#eeefff','#feffb3','#777777']
for label,x,y in zip(labels,Brandwise_FinalRankings.iloc[:,1],Brandwise_FinalRankings.iloc[:,0]):
    plt.annotate(label, (x,y))
plt.xlabel('Ranks')
plt.ylabel('Highest votes')
plt.title('Brand Preference  Rankings')
plt.show()




####Analysis and visualizations of Rankings of Activities

#Visualization of brand by rank order preferences
#len(df.columns)    
df4=df.iloc[:,16:22]
df4
df4.isnull().any() # Column wise check for null values.
df4.isnull().all
df4.isnull().sum() #No null values/missing values obtained in any column of Activity Rank subset data.
#df4.columns[3].split('- ')[1].split('.')[0]
def get_names_from_cols2(cols):
    names=cols.tolist()
    activities=[]
    for j in range(len(names)):
        #print(j)
        actv=names[j].split('- ')[1].split('.')[0]
        activities.append(actv)
    return activities

activity=get_names_from_cols2(df4.columns)    
activity #extracting Activity names from large textual column names.


activity_ranks=pd.DataFrame(index =[i for i in range(1,len(activity)+1)],columns = activity)
activity_ranks=activity_ranks.fillna(0)
activity_ranks.index

def rank_counts2(col):                     #function takes each activity ranking column from dataset df4,counts value for each rank in the column,maps the counts to each rank for each activity in a new dataframe
    c=col.value_counts().sort_index()
    d=c.to_dict()
    temp=pd.DataFrame(np.arange(7))
    #temp.iloc[:,0]
    for j in (activity_ranks.index):
        if j in (c.index.astype(np.int64)):
            temp.iloc[j,0]=d.get(j)
        else:
            temp.iloc[j,0]=0       
    temp.drop(temp.index[0],inplace=True)     
    return temp

for m in range(len(df4.columns)):
    #print(k)
    activity_ranks[activity[m]]=rank_counts2(df4.iloc[:,m]) 
activity_ranks
t=activity_ranks.T.plot(kind='bar',stacked=True,title='Ranking proprotions for each Brand',figsize=(12,6))

ActivityRanks=activity_ranks.T
Rank_for_brand={}
for ind in range(len(ActivityRanks)):
    #print(ind)
    Rank_for_brand[ActivityRanks.columns[ind]]=[ActivityRanks.iloc[:,ind].idxmax(axis=0),ActivityRanks.iloc[:,ind].max()]

Rank_for_brand
Act_Rankings=pd.DataFrame(list(Rank_for_brand.values()))
R3=Act_Rankings.set_index(Act_Rankings.iloc[:,0])
R3=R3.drop([0],axis=1)
R3
R3['Ranks']=Rank_for_brand.keys()
R3.rename(columns={1:'Activity Preference votings'},inplace=True)
Activitywise_FinalRankings=R3

#Activitywise_FinalRankings.to_csv('D:/IISWBM_Business_Analytics/PGDM_PROJECT/activity_rankings.csv') #final activity data rank wise
Activitywise_FinalRankings

#2D plot of Activity Rankings
plt.figure(figsize=(10,8))
plt.plot(Activitywise_FinalRankings.iloc[:,1],Activitywise_FinalRankings.iloc[:,0])
labels=Activitywise_FinalRankings.index
#colors=['b','g','r','c','m','y','k','#eeefff','#feffb3','#777777']
for l,x,y in zip(labels,Activitywise_FinalRankings.iloc[:,1],Activitywise_FinalRankings.iloc[:,0]):
    plt.annotate(l,(x,y))
plt.xlabel('Ranks')
plt.ylabel('Highest Activity votes')
plt.title('Activity Rankings')
plt.show()
