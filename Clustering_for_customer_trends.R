#####Importing the sheets#####

library(openxlsx)

Transactions<-read.xlsx("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/KPMG_raw_data.xlsx",sheet = "Transactions_1")
Customer_det<-read.xlsx("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/KPMG_raw_data.xlsx",sheet = "Customer_Demographic")
Customer_add<-read.xlsx("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/KPMG_raw_data.xlsx",sheet = "Customer_Address")
New_list<-read.xlsx("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/KPMG_raw_data.xlsx",sheet = "New_cust_list_final")

-----------------------------------------------------------------------------------------------------------------------------------
  ----------------------------------------------------------------------------------------------------------------------------
  
  ##### MODULE_1 #####

###Data Preprocessing ###

##customer id 34 removed, since DOB is an outlier that is 1843 dated.

View(Customer_det)
Customer_det<-Customer_det[-c(34),] 

Customer_det["gender"]

#####Checking Missing values in all data sets#####

is.na(Transactions$brand)
all(is.na(Transactions$brand))
any(is.na(Transactions$brand))
apply(Transactions,2,function(x){any(is.na(x))})#checking the columns containing missing values
apply(Transactions,2,function(x){sum(is.na(x))})#count of missing values in these columns
nrow(Transactions)
apply(Transactions,1,function(x){(all(is.na(x)))})#to check for empty row
apply(Transactions,2,function(x){(all(is.na(x)))})#to check for empty column

apply(Customer_det,2,function(x){any(is.na(x))})#checking the columns containing missing values
apply(Customer_det,2,function(x){sum(is.na(x))})#count of missing values in these columns
apply(Customer_det,1,function(x){any(all(is.na(x)))})#to check for empty row
apply(Customer_det,2,function(x){any(all(is.na(x)))})#to check for empty column

apply(Customer_add,2,function(x){any(is.na(x))})#checking the columns containing missing values
apply(Customer_add,2,function(x){sum(is.na(x))})#count of missing values in these columns
apply(Customer_add,1,function(x){all(is.na(x))})
apply(Customer_add,1,function(x){any(all(is.na(x)))})#to check for empty row
apply(Customer_add,2,function(x){all(is.na(x))})
apply(Customer_add,2,function(x){any(all(is.na(x)))})#to check for empty column

apply(New_list,1,function(x){any(is.na(x))})
apply(New_list,2,function(x){any(is.na(x))})#checking the columns containing missing values
apply(New_list,2,function(x){sum(is.na(x))})#count of missing values in these columns
apply(New_list,1,function(x){all(is.na(x))})
apply(New_list,1,function(x){any(all(is.na(x)))})#to check for empty row
apply(New_list,2,function(x){all(is.na(x))})
apply(New_list,2,function(x){any(all(is.na(x)))})#to check for empty column



##Outlier Detection for a particular data set##
New_list
summary(New_list)
names(New_list)
data<-data.frame(c(New_list[,c(6,12,13)]))
data
#data["tenure"]
data
New_list[,c(6,12,13)]
###This function is to calculate outlier range for a particular column
d1<-function(x){
  q1<-quantile(x,1/4,na.rm = TRUE)
  q3<-quantile(x,3/4,na.rm = TRUE)
  iqr<-IQR(x,type = 7,na.rm = TRUE)
  init_range<-q1-1.5*iqr
  final_range<-q3+1.5*iqr
  result<-which(x>final_range|x<init_range)
  result
}
##Function that plots outliers for a particular column.    
bxplt<-function(q){
  plt<-boxplot(q,horizontal = T,xlab = colnames(q))
  plt
}

###Apply function to calculate outlier range for all columns for a data set, using the function 'outlier_ran_per_column'
names(New_list)
data1<-New_list[,c(11,18,19,20,23)]
data1
apply(data1,2,d1) #shows the outlier data
par(mfrow=c(2,2))
apply(data1,2,bxplt) # shows the visualization of the outliers for all the columns
d<-data1$Value[data1$Value<1.71]
d
boxplot(d,horizontal = T)

names(Transactions)
head(Transactions)
data2<-Transactions[,c(4,11,12,13)]
apply(data2,2,d1)
apply(data2,2,bxplt)
nrow(data2)

data3<-Customer_add[,c(3,6)]
apply(data3,2,d1)

data4<-Customer_det[,c(5,13)]
apply(data4,2,bxplt)
apply(data4,2,d1)


##Check for duplicate records.
duplicated(New_list$job_title)
which(duplicated(New_list$job_title))

apply(Transactions,2,function(x){which(duplicated(x))})

#returns indices vector of original records which have got duplicated.

which(duplicated(Transactions)|duplicated(Transactions[nrow(Transactions):1,])[nrow(Transactions):1])
which(duplicated(Customer_add)|duplicated(Customer_add[nrow(Customer_add):1,])[nrow(Customer_add):1])
which(duplicated(Customer_det)|duplicated(Customer_det[nrow(Customer_det):1,])[nrow(Customer_det):1])
which(duplicated(New_list)|duplicated(New_list[nrow(New_list):1,])[nrow(New_list):1])


Transactions
View(Transactions)
mj<-list('Transactions','Customer_add','Customer_det','New_list')
print(mj[1])

#####Checking percentage of missing values#####

##Rowwise for each data set

(sum(apply(Transactions,1,anyNA))/nrow(Transactions)) * 100 # contains 2.7% of records with missing values
(sum(apply(Customer_add,1,anyNA))/nrow(Customer_add)) * 100
(sum(apply(Customer_det,1,anyNA))/nrow(Customer_det)) * 100 # contains 33.12 % ofrecords with missing values
(sum(apply(New_list,1,anyNA))/nrow(New_list)) # 0.24 % of records containing missing values

##Column wise for each data set##

perc_na_colwise<-function(x){
  na<-(sum(is.na(x))/length(x)) * 100
  na
}
apply(Customer_det,2,perc_na_colwise) #job industry category 16 %,job title 12.6 %
#apply(Transactions,2,perc_na_colwise)
#apply(Customer_add,2,perc_na_colwise)
#apply(New_list,2,perc_na_colwise) #job title 10.6 %, job_industry_category 16.5 %

Customer_det[7,]
anyNA(Customer_det[7,])


#####Cleaning Data#####

##Removing rows with missing values for Transactions and New_list data sets.

Transactions_1<-na.omit(Transactions)

New_list_1<-na.omit(New_list)
#(sum(apply(New_list_1,1,anyNA))/nrow(Transactions)) * 100
#apply(New_list_1,2,perc_na_colwise)

names(Customer_det)
Customer_det_1<- Customer_det[,-c(8)]
names(Customer_det_1)
Customer_det
(sum(apply(Customer_det_1,1,anyNA))/nrow(Customer_det_1)) * 100 #excluding job industry(16% NAs),NA values containing 
#records reduced to 20.2%

c2<-Customer_det[,-c(7)]
(sum(apply(c2,1,anyNA))/nrow(c2)) * 100 # excluding job title(10% NAs),NA values containing reduced to 24 %

Customer_det_2<-Customer_det[,-c(7,8)]
(sum(apply(Customer_det_2,1,anyNA))/nrow(Customer_det_2)) * 100 # excluding both job title and job industry, this reduce
#to 8.87%, hence in this case we can elimininate/remove the records containing NA ,now.


#####Removing Outliers from NA removed updated data sheets#####
#Function for the same per column
ranges<-function(m){
  q1<-quantile(m,1/4,na.rm = TRUE)
  q3<-quantile(m,3/4,na.rm = TRUE)
  iqr<-IQR(m,type = 7,na.rm = TRUE)
  init_range<-q1-1.5*iqr
  final_range<-q3+1.5*iqr
  s<-data.frame(rbind(init_range,final_range))
  s
}
low<-ranges(Transactions_1$list_price)[1,]
low
high<-ranges(Transactions_1$list_price)[2,]
high
#New_list_1
#New_list_2<-subset(New_list_1,New_list_1$Value>low & New_list_1$Value<high)
#boxplot(New_list_2$Value,horizontal = T) # to check whether outliers removed or not

Transactions_1$list_price
Transactions_2<-subset(Transactions_1,Transactions_1$list_price>low & Transactions_1$list_price<high)
boxplot(Transactions_2$standard_cost,horizontal = T) # to check whether outliers removed or not

#Customer_add is as it is

#####Removal and Imputation#####

##Customer_det_2 Removal of NAs##

Customer_det_3<-na.omit(Customer_det_2)
#names(Customer_det_2)
## Imputation of NAs ##

#install.packages("VIM")
library(VIM)
Customer_Det_imputed<-kNN(Customer_det)
head(Customer_Det_imputed)
Customer_Det_imputed<-subset(Customer_Det_imputed,select = customer_id:tenure)
summary(Customer_det)
summary(Customer_Det_imputed)
Customer_Det_imputed

which(Transactions_1$customer_id == 5034)
Transactions_1[which(Transactions_1$customer_id == 5034)]



View(Transactions_2)

pro_siz_Clas<-table(Transactions_2$product_size,Transactions_2$product_class)
round((prop.table(pro_siz_Clas)*100),1)
##most of the transactional products:'medium' product size and class:44%.next is large size,medium class: 17%:
#over all transactions include approx 70% medium class products and medium size in terms to 67% products approx.
#next its 'large' product size :around 20 % of total products sold.
pro_mid_clas<-subset(Transactions_2,Transactions_2$product_class=='medium')
pro_mid_clas
(sum(pro_mid_clas$standard_cost)/sum(Transactions_2$standard_cost))*100 #std cost approxes to about 68.4% of total transactionl costs.
#this is large, hence we can check forward for other features of this subset.


pro_line_Clas<-table(Transactions_2$product_line,Transactions_2$product_class)
round((prop.table(pro_line_Clas)*100),1)
##'standard' product line and 'medium' product class,both individually amounts to around 70 % of products.
##Post standard , its 'road' product line which corresponds to 20 % of data/products.
pro_std_line<-subset(Transactions_2,Transactions_2$product_line =='Standard')
(sum(pro_std_line$standard_cost)/sum(Transactions_2$standard_cost))*100
#this also amounts to approx 67 %, standard product line products total std cost is 67 % of the overall std cost.

pro_line_size<-table(Transactions_2$product_line,Transactions_2$product_size)
round((prop.table(pro_line_size)*100),1)
pro_mid_size<-subset(Transactions_2,Transactions_2$product_size=='medium')
(sum(pro_mid_size$standard_cost)/sum(Transactions_2$standard_cost))*100
##when checking by 'medium' size, this amounts to 52% of total std cost.


##checked for brand also, it gives no such absolute information.

##First thing inferred is that,products are of 'standard','medium','medium'(pro line,pro class,pro size) that contributes the cost mostly.
#Second is of 'road','medium','large'.


Customer_det_3[c(30:40),]
Transactions_2[Transactions_2$customer_id==5034]
summary(Customer_det_3$tenure)
Customer_det_3['Tenure_Cat']<-cut(Customer_det_3$tenure,breaks = c(1,7,14,22),include.lowest = TRUE,labels=c('lessthan 7','between 7 and 13','between 14 and 22'))
Customer_det_3['Tenure_Cat5']<-cut(Customer_det_3$tenure,breaks = c(0,7,14,22),include.lowest = TRUE,labels=c('between0 and 7','between 8 and 14','between 15 and 22'))
View(Customer_det_3)
#####Trying to check impact of time on standard cost and also distribution.#####
hist(Transactions_2$list_price)
summary(Transactions_2$list_price)
plot(density(Transactions_2$list_price))
getmode(Transactions_2$list_price)
hist(Transactions_2$standard_cost)
plot(density(Transactions_2$standard_cost))

#install.packages("lubridate")
#install.packages("generics")
#library(lubridate)
#Transactions_2$transaction_date=lubridate::dmy(Transactions_2$transaction_date)
#dplyr::arrange(Transactions_2,transaction_date)

t4<-Transactions_2[order(Transactions_2$transaction_date),]
#Transactions_3<-dplyr::arrange(Transactions_2,transaction_date)
hist(t4$standard_cost)
plot(density(t4$standard_cost))
Transactions_2
summary(Transactions_2$standard_cost) #from summary  mean is greater than median, and also the histogram shows that the 
#standard cost dist is right/positively skewed. It is a unimodal data dist(only a single class/interval of std cost
#with maximum frequency)

par(mfrow=c(1,2))
summary(Transactions_2$list_price)
plot(density(Transactions_2$list_price),main = "Density distribution of list price")
polygon(density(Transactions_2$list_price),col = 'green',border = blues9)

summary(Transactions_2$standard_cost)
plot(density(Transactions_2$standard_cost),main = "Density distribution of standard cost")
polygon(density(Transactions_2$standard_cost),col = 'orange',border = blues9)

mode(Transactions_2$standard_cost)
getmode<-function(x){
  uniq<-unique(x)
  uniq[which.max(tabulate(match(x,uniq)))]
}  
getmode(Transactions_2$list_price)
getmode(Transactions_2$standard_cost) ##mode=388.9,the shape indicates that there are no. of data points /outliers 
#that are greater than the mode.
#the mode lies to the left of the grph and is smaller than the median and the mean.
# the longer tail on the right side of the mode indicates that as the std cost of products goes on increasing,
#the number of transactions/buying corresponding to this cost keeps falling/decreasing
#while the highest/most buying has occured near the mode/highest peak of the curve/histogram.
#however there is no gradual falling of the tail of the dist to the right, and buying increases between
#500-600, then again falls and increases, until it finally hits a low buying at 1100.
#therefore most of transactions occured in 0-900 range.

#####Finding Weighted mean of list price per customers#####
order(Transactions_2$customer_id)
Transac2_sort<-Transactions_2[order(Transactions_2$customer_id),]
#q=subset(Transac2_sort,Transac2_sort$customer_id==1)
#mean(270.30)
head(Transac2_sort)
list_bycost<-data.frame(matrix(nrow = 3489,ncol = 2))
colnames(list_bycost)<-c('customer_id','Mean_List_Price')
#head(Transac2_sort)
for (i in 1:length(unique(Transac2_sort$customer_id))){
  #print (i)
  temp_data<-subset(Transac2_sort,Transac2_sort$customer_id==i)
  mean_price<-mean(temp_data[,11])
  #print(mean_cost)
  list_bycost[i,1]<-i
  list_bycost[i,2]<-mean_price
}
nrow(Transac2_sort)
nrow(list_bycost) ##unique customers are 3491 who have done some transactions.
#top_1000_customers[1,2]

#####Finding top 1000 customers#####
top_1000_customers<-head((list_bycost[order(-list_bycost$Mean_List_Price),]),2900)
#top_1000_customers[1,1]
#length(unique(Customer_det_3$customer_id))
head(top_1000_customers)

#####Merging of data sets for all details#####
#top_1000_cust_details<-data.frame(matrix(nrow = 1000,ncol = 12))
#top_1000_cust_details
#for (i in 1:length(top_1000_customers$Cust_id)){
  #print(i)
#  for (j in 1:length(unique(Customer_det_3$customer_id))){
#    if(top_1000_customers[i,1]==Customer_det_3[j,1]){
#      top_1000_cust_details[i,]<-Customer_det_3[Customer_det_3[j,1],]
#      #top_1000_cust_details[i,12]<-top_1000_customers[,2]
#    }
#  }
#}
#Customer_det_3[2,1]
#top_1000_cust_details[1,]
#Transac2_sort
#Transac2_sort[Transac2_sort$customer_id==1667,]
#Customer_det_3[Customer_det_3$customer_id==1667,]
#names(Customer_det_3)
#colnames(top_1000_cust_details)<-c('customer_id','first_name','last_name','gender','past_3_years_bike_related_purchases','DOB','wealth_segment','deceased_indicator','default','owns_car','tenure')
#head(top_1000_cust_details)
#top_1000<-na.omit(top_1000_cust_details)
#head(top_1000)

#Top1000_customers<-merge(top_1000_customers,Customer_det_3,by="customer_id")
#Top1000custs_details<-merge(Top1000_customers,Customer_add,"customer_id")
#head(Top1000custs_details)
#names(Top1000custs_details)
#names(New_list_2)
#names(New_list)
#Transactions_3
Merge1<-merge(top_1000_customers,Transactions_2,by="customer_id")
Merge2<-merge(Merge1,Customer_det_3,by="customer_id")
Final_merge_fortop1000<-merge(Merge2,Customer_add,"customer_id")

head(Final_merge_fortop1000)
#(unique(Final_merge_fortop1000$online_orde)
  
write.csv(Final_merge_fortop1000,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/Top_1000_Cust_details.csv",row.names=FALSE)  
sum(Final_merge_fortop1000$list_price)/sum(Transactions_2$list_price)

#####Applying PAM with Silhoute coefficient#####

###Now we consider the data set which includes :
#2900 out of 3491 unique customers data who covers 3/4th of the total transactions performed, which amounts to 82% of total sell.
#This data is in descending order of the weighted list price corresponding to customers transactions, that is from highest to lowest.
#On merging the customer demographic and customer address sheets to get necessary customer details, 
#finally 2637 customers details are found on merging the data sets to get customer details.
#all of whom contribute the most part of the total sell price(from highest to lowest order) and this final data covers 65 % of customer details.
customers<-merge(top_1000_customers,Customer_det_3,by="customer_id")
customers<-na.omit(customers)
custs_details<-merge(customers,Customer_add,"customer_id")
custs_details<-na.omit(custs_details)
nrow(custs_details)
class(custs_details['DOB'])
write.csv(custs_details,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/Cust_details.csv",row.names=FALSE) 

#c1<-merge(top_1000_customers,Customer_Det_imputed,by="customer_id")
#c1<-na.omit(c1)
#View(c1)
#c2<-merge(c1,Customer_add,"customer_id")
#c2<-na.omit(c2)
#View(c2)
#write.csv(c2,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/Cust_details_new.csv",row.names=FALSE)
#df1<-read_csv("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/Cust_details_new.csv",col_names = TRUE)
###Now label encoded in python from Cust_details.csv file

library(cluster)
library(dplyr)
library(ggplot2)
library(readr)
library(Rtsne)

#install.packages("tsne")
library(tsne)
df<-read_csv("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/Cust_details.csv",col_names = TRUE)
df['Age_Categ']<-cut(df$Age,breaks = c(18,31,47,65,90),labels = c('above18','above30','above46','above65'))
View(df)
names(df)
df['Age_Cat_2']<-type.convert(df[,21],'factor')
df['wel_seg2']<-type.convert(df[,8],'factor')
df['gender_2']<-type.convert(df[,5],'factor')
df['ownscar_2']<-type.convert(df[,11],'factor')
df['state_2']<-type.convert(df[,16],'factor')
df['tenure_Cat_2']<-type.convert(df[,13],'factor')
#summary(df1$Age)
#df1['Age_Cat']<-cut(df1$Age,breaks = c(18,31,47,65,90),labels = c('above18','above30','above46','above65'))
#df1['Age_Cat_2']<-type.convert(df1$Age_Cat,'factor')
#df1['wel_seg2']<-type.convert(df1$wealth_segment,'factor')
#df1['gender_2']<-type.convert(df1$gender,'factor')
#df1['ownscar_2']<-type.convert(df1$owns_car,'factor')
#df1['state_2']<-type.convert(df1$state,'factor')
#df1['tenure_Cat_2']<-type.convert(df1$tenure,'factor')
##Compute Gower Distance
#View(df1)
df_encoded<-read_csv("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/Cust_details_encoded.csv",col_names = TRUE)
names(df)


#df_encoded['wel_Fact']<-as.factor(df_encoded[,9])
#df_encoded<-df_encoded[,-c(23)]
class(df_encoded[,8])
summary(df_encoded$Age)
df_encoded['Age_Category']<-cut(df_encoded$Age,breaks = c(18,31,47,65,90),labels = c('above18','above30','above46','above65'))
nrow(New_list)
New_list['Age Cat2']<-cut(New_list$Age,breaks = c(18,31,47,65,90),labels = c('above18','above30','above46','above65'))
New_list['Tenure_Cat2']<-cut(New_list$tenure,breaks = c(0,7,14,22),include.lowest = TRUE,labels=c('lessthan 7','between 7 and 14','between 15 and 22'))
View(New_list)
write.csv(New_list,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/New Customer List2.csv",row.names = FALSE)
summary(New_list$tenure)
names(df_encoded)
View(df_encoded)
df_encoded['Age_Cat_2']<-type.convert(df_encoded[,25],'factor')
df_encoded['wel_seg2']<-type.convert(df_encoded[,9],'factor')
df_encoded['gender_2']<-type.convert(df_encoded[,6],'factor')
df_encoded['ownscar_2']<-type.convert(df_encoded[,12],'factor')
df_encoded['state_2']<-type.convert(df_encoded[,17],'factor')
df_encoded['tenure_Cat_2']<-type.convert(df_encoded[,24],'factor')
View(df_encoded)

df2<-df_encoded[,c(7,26,27,28,29,30,31)]
df3<-df[,c(6,22,24,25,27)]
names(df3)
View(df3)
#df1_dashed<-df1[,c(6,14,19,22,23,24,25,27)]
#?cut
write.csv(df_encoded,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/full_data_final.csv",row.names = FALSE)
write.csv(df2,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/mixeddata_for_model.csv",row.names = FALSE)
names(df2)
View(df2)
gower_dist<-daisy(df3,metric = "gower")
#gower_dist2<-daisy(df1_dashed,metric = "gower")
gower_dist

gower_mat <- as.matrix(gower_dist)
#gower_mat2<-as.matrix(gower_dist2)
gower_mat

#Print most similar clients
df3[which(gower_mat==min(gower_mat[gower_mat!=min(gower_mat)]),arr.ind=TRUE)[1,],]

#Print most dissimilar clients
df3[which(gower_mat == max(gower_mat[gower_mat != max(gower_mat)]), arr.ind = TRUE)[1, ], ]

#Finding number of clusters required
sil_wid<-c(NA)
for (i in 2:8){
  pam_fit<-pam(gower_dist,diss = TRUE,k=i)
  sil_wid[i]<-pam_fit$silinfo$avg.width
 }

sil_wid
plot(1:8,sil_wid,xlab = "Number of Clusters",ylab = "Silhouette Width")
lines(1:8,sil_wid) #sine the highest silhouette coefficient for k=4, hence taking the same.


#sil_wid1<-c(NA)
#for (i in 2:8){
#  pam_fit1<-pam(gower_dist2,diss = TRUE,k=i)
#  sil_wid1[i]<-pam_fit1$silinfo$avg.width
#}

#sil_wid1
#plot(1:8,sil_wid1,xlab = "Number of Clusters",ylab = "Silhouette Width")
#lines(1:8,sil_wid1) 


#Interpretation of each cluster

k<-6
pam_fit_final<-pam(gower_dist,diss = TRUE,k)
#pam_fit_final$silinfo$avg.width
pam_results<-df3 %>%
  mutate(cluster = pam_fit_final$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))

pam_results$the_summary
#hist(df2$past_3_years_bike_related_purchases)

#k<-2
#pam_fit_final2<-pam(gower_dist2,diss = TRUE,k)
#pam_fit_final$silinfo$avg.width
#pam_results1<-df1_dashed %>%
#  mutate(cluster = pam_fit_final2$clustering) %>%
#  group_by(cluster) %>%
#  do(the_summary = summary(.))

#pam_results1$the_summary

##Visualization in a lower dimensional space.
tsne_obj<-Rtsne(gower_dist,is_distance = TRUE)
tsne_obj
#Rtsne(gower_mat,is_distance = TRUE)
#?Rtsne
tsne_obj

tsne_data<-tsne_obj$Y %>%
  data.frame() %>%
  setNames(c("X","Y")) %>%
  mutate(cluster=factor(pam_fit_final$clustering))

tsne_data

ggplot(aes(x = X, y = Y), data = tsne_data)+geom_point(aes(color = cluster))

#####Adding age category to the new customer list#####

NewCustomerList<-read.xlsx("D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/KPMG_VI_New_raw_data_update_final.xlsx",sheet = "NewCustomerList")
View(NewCustomerList)
NewCustomerList['Age Category']<-cut(NewCustomerList$Age,breaks = c(18,31,47,65,90),labels = c('above18','above30','above46','above65'))
NewCustomerList
write.csv(NewCustomerList,"D:/IISWBM_Business_Analytics/KPMG Virtual Internship/Module 2/NewCustomerList_final.csv",row.names=FALSE)

#df2
#df2%>%mutate(cluster=pam_fit_final$clustering)%>%group_by(cluster)%>%do(the_summary=summary(.))
#pam_results



#####AGNES CLUSTERING#####
#df3<-na.omit(df2)
#df3<-scale(df3)
#m<-c("average", "single", "complete", "ward")
#names(m)<-c("average", "single", "complete", "ward")
#m
#function calculating aglomerative coefficient
#library(purrr) ##has function map_dbl
#ac<-function(x){
#  agnes(df3,method = x)$ac
#}
#map_dbl(m,ac)
##agnes clustering method selected based on aglomerative coefficient , that is , ward method
#agnes_clust<-agnes(df3,method = "ward")
##Dendogram-cluster tree formed
#pltree(agnes_clust,cex=0.6,hang=-1,main = "Dendogram of Agnes Clustering")
##Determining the number of clusters
#library(factoextra) ### for visulaizing cluster results
#p1<-fviz_nbclust(df3,FUN=hcut,method = "wss",k.max = 10)+ggtitle("Elbow Method")
#p2<-fviz_nbclust(df3,FUN=hcut,method = "silhouette",k.max = 10)+ggtitle("Silhouette method")
#p3<-fviz_nbclust(df3,FUN=hcut,method = "gap_stat",k.max = 10)+ggtitle("Gap Statistic")
#gridExtra::grid.arrange(p1,p2,p3,nrow = 1)
##Assign clusters to data points ---cuts a tree to several groups/clusters based on k(no. of clusters)
#sub_grp <- cutree(agnes_clust, k = 10)
#sub_grp
#Number of members in each cluster
#table(sub_grp)
##Plot the full dendogram and highlight with the clusters
#fviz_dend(
#  agnes_clust,
#  k = 5,
#  horiz = TRUE,
#  rect = TRUE,
#  rect_fill = TRUE,
#  rect_border = "jco",
#  k_colors = "jco",
#  cex = 0.1
#)
#dend_plt<-fviz_dend(agnes_clust) #create full dendogram
#dend_plt
#dend_data<-attr(dend_plt,"dendogram") #extract plot info
#dend_data
#dend_cuts<-cut(dend_data,h=35) #cut the dendogram at designated height
 #Create sub dendogram plots
#p4<-fviz_dend(dend_cuts$lower[[1]])
#p5<-fviz_dend(dend_cuts$lower[[1]], type = 'circular')
#Side by side plots
#gridExtra::grid.arrange(p4,p5,nrow=1)


