# End-to-end-analysis-of-Customer-churn-and-retention-of-a-telecom-company

### Tools used: Power BI, Python, MS Excel
## Introduction

Telecommunication industry is a dynamic and competitive landscape and understanding the 
customer behavior for this industry is of great significance. Customer churn is a phenomenon of 
discontinuing their service with the telecom provider which poses a significant challenge for the 
companies who strive to maintain their market position and profits. On the other hand, retention 
of customer focuses on strategies to keep existing customers at utmost satisfaction and retain 
their loyalty to the service. Thus, the two way focus on churn and retention is considered critical 
aspect for the sustenance of growth of the company and therefore ensuring a stable revenue.


Telecom companies are always in a fierce battle with high churn rates which are driven by a 
variety of factors which includes competitive pricing, technological advancements and shift in 
customer expectations. In the juggle of acquiring new customers, retaining the existing ones is 
often neglected. Thus, efforts should be made to manage churn in a more cost-effective approach 
so that a robust customer database is maintained. Therefore, a thorough analysis of churn and 
retention of customers for such an industry becomes inevitable.


This analysis comprises of the examination of the historical data to identify the various patterns 
and predict the KPIs for churn. The dataset includes various data points: demographic variables, 
services used, billing information etc. of the service provider company. One can build predictive 
models and visualize the same by employing some of the advanced statistical techniques and 
machine learning techniques. This can help us in addressing the customer concerns.


Having the understanding and at the same time leveraging the features such as Tenure, Types of 
services and the payment methods used is crucial for the study. Noting an example: Customers 
with shorter tenure or who use high demand services: Streaming TV or Streaming movies may 
exhibit different churn behaviors when compared to those with long term or basic service users.
By focus on these, telecom companies can tailor their retention efforts to address the unique 
needs and preferences of different segments of customers.


## Objectives of the study:


1. To identify key drivers of customer churn for the telecommunication industry.
   
2. To analyze the impact of customer demographics and usage patterns.

3. To segment the customers based on churn risk.

4. To develop a model for customer churn.

5. To use Reporting and visualization techniques.


## Secondary data available for study:

The data available is a secondary data that is collected by the company for analysis purposes. The detailed explanation about the data collected is as follows:

 Customer ID: This consists of the unique Customer ID which is one of the important data point required for data analysis.

 Gender: This consists the details of the gender of the customer- Male/Female/Other.

 Senior Citizen: This evaluates whether the customer who opts for the services is a senior citizen or not.

 Partner: This evaluates whether the customer has already opted for the services or not.

 Dependents: This evaluates whether the customer has dependents or not.

 Tenure: This evaluates for how long has the customer been availing the services in months.

 Phone Service: This evaluates whether the customer has opted for phone services or not.

 Internet Service: This evaluates whether the customer has opted for internet services or not.

 Online Security: This evaluates whether the Online security is enabled for the customer or not.

 Online Backup: This evaluates whether the customer has opted for online backup option or not.

 Device Protection: This evaluates whether the device protection is enabled for the customer or not.

 Tech Support: Collects data on whether tech support is available for the customer.

 Streaming TV: Collects data on whether customer has availed TV streaming services.

 Streaming Movies: Collects data on whether customer has availed Movie streaming services.


 Contract: This throws insights into how frequently is the contract renewed for a customer.

 Paperless billing: Does the customer encourage the paperless billing method?

 Payment method: This gathers insights on the various modes of payment while availing the telecom services.

 Monthly Charges: It aims to gather details on the monthly chargers paid by the customer.

 Total Charges: It aims to gather insights on the charges paid by a customer for the entire year.

 Renewal Status: A dependent variable for the analysis where the churn/no churn is determined for a customer.


## Data Collection and Hypotheses:

Based on the secondary data collected, there are certain hypotheses related to the Telecom industry which focuses on different aspects of customer behavior, service usage 
and financial metrics.

Let‟s go through few of the hypotheses one by one.

### Hypothesis 1:
Impact of Contract type on customer churn.
Null hypothesis (H0): The type of contracts (month-to-month,one year, two year) does not significantly affect the customer churn rates.

Alternative hypothesis (H1): The type of contract significantly affects customer churn rates.

We can determine if longer-term contracts can lead to lower churn rates.


### Hypothesis 2: 
Relationship between Monthly charges and Churn rate.

Null hypothesis (H0): Monthly charges do not significantly affect churn rate.

Alternative hypothesis (H1): Monthly charges significantly affect churn rate.

We can assess if higher monthly charges relate to higher monthly churn.


### Hypothesis 3: 
Influence of Senior Citizen Status on Customer Retention.

Null hypothesis (H0): Senior citizen status does not significantly influence the customer retention.

Alternative hypothesis (H1): Senior citizen status significantly influences the customer retention.

We can understand if senior citizens are more likely to be retained if the tech support is enhanced.


### Hypothesis 4: 
Effect of Payment Method on Churn rate.

Null hypothesis (H0): The method of payment does not significantly affect the churn rate.

Alternative hypothesis (H1): The method of payment significantly affects the churn rate.

We can explore if customers using digital payment methods have lower churn rates and can promote these methods to retain customers.


By analyzing above hypotheses, we can gain insights into key factors influencing churn and develop targeted strategies to improve customer retention.
For each of the above hypothesis, statistical tests are performed to analyze the hypotheses. It includes:

 Chi-square tests for independence for categorical variables.

 t-test or ANOVA for comparing means between groups.

Based on the p-values and test statistics, we can determine whether to reject or fail to reject the null hypothesis for each case.


## Data collection and Hypothesis:

The data collected is unfit for analysis as it consists of inconsistent data. Thus it requires the preprocessing of the data to make it fit for the analysis. Data pre-processing involves Data cleaning 
and Data transformation. There are 7046 records before pre-processing. Once, the pre-processing 
is complete we number of useful records is noted to be 7016. 

### Data cleaning: 
This process is carried out by making use of MS Excel. We achieve:

 Removal of formatted tables.

 Removal of special characters.

 Removal of blanks.

 Checking for the duplicates using conditional formatting.

 Merging of Columns – Customer ID columns.

![image](https://github.com/user-attachments/assets/c81a3cbe-865b-4290-adc5-b5203a7b7f87)

### Data transformation:
Once the data is cleaned, Power BI is used for transformation of the data. 

Through data transformation, we achieve:

 Checking for the Data types.

 Following the naming convention for each column.

 Addition of custom columns.


### Exploratory Data Analysis:
#### Objective 1: The following analysis is carried out to identify the key drives for customer churn.
This step is carried out to analyze the key factors for the customer churn. EDA is a critical step in data analysis process, where the dataset is explored to uncover initial patterns, anomalies, and 
relationships to form hypothesis for further analysis. It is an iterative and open ended process that allows analysts to explore the data without preconceived notions, leading to better insights and 
informed decision-making.

Quantitative methods:
Descriptive statistics: This is performed using M S Excel data analysis tool pack.

#### Tenure:

![image](https://github.com/user-attachments/assets/3f9fe0da-e3c5-4f80-931a-01a6fa3a74ec)

Interpretation: The average tenure of the customers with the telecom company is 32 months with a right skewed distribution since median is less than mean. 
Standard deviation of 24 months indicates a high variability in the tenure of customers. The tenure is spread out over a wide range of values. 


#### Monthly charges:

![image](https://github.com/user-attachments/assets/f75345b5-dd6b-4c2e-976f-855b809c28e4)

Interpretation: The average monthly charges that the 7016 respondents pay is Rs.64.72 and it is a left skewed distribution since median is greater than mode. Standard deviation of 30.105 indicates a high variability in the monthly charges paid by the customers. 


#### Total charges:

![image](https://github.com/user-attachments/assets/e9fa3c7d-0aac-4fc0-bb55-1351d348c6a5)


Interpretation: The above statistics shows that the average total charges paid by the customers for the entire tenure of subscription. There is a high deviation from mean indicating high variation on total charges paid by customers.


#### Inferential statistics:
#### t-test:

![image](https://github.com/user-attachments/assets/177518c3-e591-4d7a-b9cc-9a1fd5f0b5e9)
 
The t-stat value if -69.78 is used to determine whether to reject the null hypothesis.

#### P-values:

One tail: the p-value is 0. Since is less than 0.05(significance level) we reject the null hypothesis for one tail test.

Two tail: the p value is 0. Since is less than 0.05(significance level) we reject the null hypothesis for two tail test.

#### Critical values: 

One tail: the critical value for one tail at 0.05 significance level is 1.644.

Two tail: the critical value for two tail at 0.05 significance level is 1.96.

#### Result: Since the absolute value for t-statistic(-69.78) is much greater than the critical t-values and the p values are 0, we reject the null hypothesis. This indicates this is a statistically significant relationship between the means of Tenure and Monthly charges.




#### Multiple Linear Regression Analysis:

![image](https://github.com/user-attachments/assets/549aa318-0d70-429f-bb21-bf557f9cc4b3)


#### Interpretation:

Multiple R=0.9102 is a correlation coefficient and indicates a strong positive linear relationship between the observed and predicted values of the dependent variable which is Tenure.

R square value of 0.8284 suggests that 82.84% of variation in Tenure can be explained by the model.

Adjusted R square is a more accurate measure of goodness of fit which is 82.82%. It is very close to R square value indicating model fits data well.

#### ANOVA:

Significance f = 0 indicates p-value associated with f statistic showing model is highly significant.

#### Conclusion:
The model is highly significant as indicated by statistic and its associated p-value. Both Total charges and Monthly charges are significant predictors of the dependent variable.

The positive co-efficient for Total_charges suggests that higher total charges are associated with an increase in Tenure. The negative coefficient for monthly charges indicates that higher monthly charges are associated with a decrease in Tenure.

The high R Square value indicates that the model explains a substantial portion of the variance in the Tenure.
Overall, the regression model appears to fit the data well and both predictors are statistically significant.

Thus, it can be noted that tenure, monthly charges, total charges are the key drivers in determining customer churn.


#### Qualitative methods:

#### Objective 2: The following steps help to analyze  the impact on customers due to demographics.

#### Univariate Analysis:
It is the simplest form of data analysis that involves examining the distribution and characteristics of a single variable. The main objective of univariate analysis is to describe and summarize the dataset for that particular variable. This type of analysis is crucial as it provides insights into the basic properties and structure of the data. 

Let us analyze few variables using Python codes using Google colaboratory for the same.
#### Gender: 
 
![image](https://github.com/user-attachments/assets/3e843567-1f9a-4f98-a535-b2445f789223)

![image](https://github.com/user-attachments/assets/a5fd9fb0-7c43-4cf6-81a2-57ab4d648689) 


From the above bar chart and frequency table, the data suggests that there is a fair distribution between male and female, with males slightly outnumbering females in the data provided for the telecom company.



#### Internet Service:

![image](https://github.com/user-attachments/assets/56a846e2-72b6-41c7-a762-449b69087e81)

![image](https://github.com/user-attachments/assets/0f0a017a-18f0-4392-a960-5318f5059025)

From the analysis, one can infer that,
•	Fiber optic is the most common type of internet service among the respondents, accounting for nearly 44%.
•	DSL follows with about 34%.

Approximately 22% of the respondents are reported to have no internet service.
This breakdown provides insights into the prevalence of different types of internet services among the surveyed population.
#### Contract:

![image](https://github.com/user-attachments/assets/e1c0cac9-a13d-4e32-9eed-2690bcf83939)

![image](https://github.com/user-attachments/assets/46985df1-d168-4af3-88dc-1602d87dc939)

From the analysis, it is seen that most of the customers availing the telecom services renew the contract as follows:

•	Month-to-month contracts are most common, accounting for  about 55% of the respondents.

•	Two year contracts accounting for 24% of the respondents.

•	One year contracts accounting for 21% of the total respondents.

This analysis shows a preference for month-to-month contracts to long term contracts.


#### Objective 3: The analysis below helps to segment the customers based on churn risk.
#### Bivariate and Multivariate analysis:
Correlation analysis is typically used to examine the relationship between two or more variables. It measures the strength and direction of the linear relationship between two or more variables at a time. It helps to determine whether and to what extent changes in one variable are associated with other variables. 

This can be achieved using a simple Python code:
print(correlation_matrix)

Each cell in the matrix represents the correlation coefficient between variables. The coefficient ranges from -1 to +1.
•	A coefficient close to 1 indicates a strong positive  correlation, meaning that as one variable increases, the other variables tend to increase as well.
•	A coefficient close to -1 indicates a strong negative correlation,meaning that as one variable increases, the other variables tend to decrease.
•	A coefficient close to 0 indicates little to no linear relationship between the variables.

![image](https://github.com/user-attachments/assets/a8ab3355-2a45-41e0-9080-fa17923e30ee)

![image](https://github.com/user-attachments/assets/e2ffd603-477e-4370-bd94-955e00bb0c29)

![image](https://github.com/user-attachments/assets/34c81296-c670-4f92-b2fb-6f9aaa7c944c)

![image](https://github.com/user-attachments/assets/34700f12-6107-4de4-89ac-b0ecde32fedf)

![image](https://github.com/user-attachments/assets/f72ed787-a5bb-4f6e-bf37-c0848b5c3fca)


In order to proceed with the above correlation analysis, the nominal data is converted to binary form in order to get primitive correlation results.
The results obtained as follows:

•	Senior_Citizen  shows very low correlation with other variables, suggesting it may not strongly influence other aspects measured.

•	Partner and Dependents have moderate positive correlations (0.452 and 0.159 respectively) with other variables like tenure, suggesting possible relationships with customer loyalty or household dynamics. 

•	Tenure correlates positively with several service related variables such as Online_Security, Online_Backup, Device_Protection, Tech_Support, Streaming_TV, Streaming_Movies indicating that longer tenure customers may opt for more services.

•	Monthly_Charges and Total_Charges show moderate positive correlations with most service-related variables, indicating  that customers who use more services tend to have higher charges.

The above correlation can be visualized using a heat map by using Python codes as follows:
correlation_matrix=df_numerical.corr()
sns.heatmap(correlation_matrix,cmap='Greens')

 ![image](https://github.com/user-attachments/assets/736b3f3e-f667-4900-8aac-a103cd4b889a)

The darker areas suggest higher positive correlation while the lightly shaded regions show negative correlation between the variables.

The above analysis performed helps us note that Contract, Tenure, Monthly charges, Total Charges, Senior Citizen and Gender can be used as a basis for segmentation of customers churn risk. This also throws light on the fact that demographic factors do not have much impact on churn and retention. But the quantitative variables definitely affect the same.


#### Data normalization:
Data normalization refers to the process of scaling individual data points to fall within a specified range often [0,1]. Normalization is useful when one wants to ensure that the features contribute equally to the distance measurements in algorithms that are sensitive to the scale of the data such as K-nearest neighbours and neural networks.

There are categorical variables such as Gender, Senior Citizen, Dependents, Phone Service, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies, Contract, Paperless Billing and Payment Method which need to be treated with dummies (0 and 1 format) to build models. This step ensures that the models can interpret the data effectively and learn from the categorical data.


### Model building:

#### Objective 4: The steps below are to build various models and predict the customer churn.

It involves creating a mathematical framework that describes relationship between input variables and output variables. The goal of model building is to make predictions or infer insights from data based on patterns and relationships captured during training process. 

Various supervised learning methods such as Logistic Regression, KNN, Decision Tree, Random forest and Naïve Bayes are used to build models to infer data driven and actionable insights. 

#importing the libraries for model building

from sklearn.linear_model import LogisticRegression  #importing LR

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

The above libraries need to be imported in Python before building any of the above models.


Let’s go through building each model one by one.
#### Logistic Regression:
#Creating an instance for Logistic Regression using Python

lr_model = LogisticRegression()  #Now fitting the model

lr_model.fit(X_train,y_train)

lr_pred=lr_model.predict(X_test)

print(confusion_matrix(y_test,lr_pred)) #gives true positive and true negative

print(classification_report(y_test,lr_pred))   #gives sensitivity

print(accuracy_score(y_test,lr_pred))  #gives precision

 
![image](https://github.com/user-attachments/assets/2d3dde7d-5bae-43e6-9fb8-1ad70dea90fc)

#### Interpretation:

True Positives: 196

False Positives: 146

False Negatives: 139

True Negatives: 923

The model correctly predicts 1119(923+196) cases as negative and 342(139+203) cases as positive..
It incorrectly classifies 285 cases(146+139).

#### Conclusion: 
The  model shows good performance in predicting non-churning cases with high precision and recall. However, predicting the churn cases where precision and recall are lower can be still improved. The model is 79.7% accurate.


#### K-nearest neighbors:

knn_model = KNeighborsClassifier(n_neighbors=2)  # n_neighbors = 2 is k=2

#we choose k=2 because accuracy is highest at k=2

knn_model.fit(X_train,y_train)

knn_pred=knn_model.predict(X_test)

print(confusion_matrix(y_test,knn_pred))

print(classification_report(y_test,knn_pred))

print(accuracy_score(y_test,knn_pred))

![image](https://github.com/user-attachments/assets/579441ea-9b69-492b-80fb-7b589264920f)

#### Interpretation:

True positives: 110

False positives: 109

False negatives: 225

True negatives: 960

The KNN model has an accuracy of 76% indicating the overall correctness of the model across all classes. The model correctly predicts 1070(960+110) cases as negative and 319(225+94) cases as positive. It incorrectly classifies 334 cases(109+225).

The KNN model shows decent performance in predicting non-churn cases with high precision and recall. But the prediction of churn can still be improved. 


#### Decision Tree Classifier:

tree_model=DecisionTreeClassifier(criterion = 'gini') # or entropy

tree_model.fit(X_train,y_train)

tree_pred=tree_model.predict(X_test)

print(confusion_matrix(y_test,tree_pred))

print(classification_report(y_test,tree_pred))

print(accuracy_score(y_test,tree_pred))

 ![image](https://github.com/user-attachments/assets/3bbad3f1-fb95-413f-af0f-1f899f80e639)

#### Interpretation:

True positives: 168

False positives: 224

False negatives: 167

True negatives: 845

The decision tree classifier model has the accuracy of 72.15% indicating the overall accuracy of the model across all classes.
The model correctly predicts 1013(845+168) cases as negative and 392(224+168) cases as positive. It incorrectly classifies 375 cases(224+167).

The decision tree model shows a reasonable performance in predicting non churn cases and a decent precision and recall. It can be improvised for predicting churn cases.


#### Random Forest Classifier:

rf_model=RandomForestClassifier(criterion = 'gini', n_estimators=100,min_impurity_decrease=0.02) # It is an ensemble method

rf_model.fit(X_train,y_train)

rf_pred=rf_model.predict(X_test)

print(confusion_matrix(y_test,rf_pred))

print(classification_report(y_test,rf_pred))

print(accuracy_score(y_test,rf_pred))

 ![image](https://github.com/user-attachments/assets/56b7707f-454a-4bdf-b1aa-d382372ec77e)

#### Interpretation:

True positives: 0

False positives: 0

False negatives: 335

True negatives: 1069

The model is 76.13% accurate  which measures the overall correctness of the model across all classes.

The model performs well in predicting non-churn cases but completely fails to predict any churn cases. This indicates a significant disability to distinguish between churn and non-churn cases. Thus this model cannot be used further.


#### Naïve Bayes:

NB_model=GaussianNB() # It is a bayes model

NB_model.fit(X_train,y_train)

NB_pred=rf_model.predict(X_test)

print(confusion_matrix(y_test,NB_pred))

print(classification_report(y_test,NB_pred))

print(accuracy_score(y_test,NB_pred))

![image](https://github.com/user-attachments/assets/59e7dd34-882f-438c-b7e1-93da2a697173)

#### Interpretation:

True positives: 173

False positives: 108

False negatives: 162

True negatives: 961
 
The model has an accuracy of 81% which measures the overall correctness of the model across all classes. 
The model correctly predicts 1134(961+173) cases as negative and 270(108+162) cases as positive. It incorrectly classifies 270 cases(108+162).

The Guassian NB model shows a very good performance in predicting both churn and non churn cases with an accuracy of 81%.


#### Summary table:

![image](https://github.com/user-attachments/assets/beaf0a11-dc4d-4c4f-92c4-26264d32ee3d)


Thus, from the above table, Naïve Bayes is the most accurate model that can be used to predict churn of customers for the telecom company.

The models built help us achieve the objective of predicting the risk of churn and possibility of retention of the customers.

### Sentimental Analysis:

The feedback was collected from customers in the form of Smiles in which Green means the customers are highly satisfied and can be called as Loyal customers; Yellow means the customers are somewhat satisfied and are having the chances of switching to other service providers and can be named as Satisfied customers; Red indicates that the customers are not at all happy with the service and can be named as Dissatisfied customers.

Certain measures were created in Power BI to assess the sentiments of the customers towards the service. They are listed as follows:

Total Cust = CALCULATE(DISTINCTCOUNT('Customer Delight'[Customer_ID]))

Green = CALCULATE(COUNT('Customer Delight'[Customer_ID]),'Customer Delight'[Delight] = "Green")

Yellow = CALCULATE(COUNT('Customer Delight'[Customer_ID]), 'Customer Delight'[Delight] = "Yellow")

Red = CALCULATE(COUNT('Customer Delight'[Customer_ID]),'Customer Delight'[Delight] = "Red")

Loyalists = CALCULATE('Customer Delight'[Green]/'Customer Delight'[Total Cust])

Satisfied = CALCULATE('Customer Delight'[Yellow]/'Customer Delight'[Total Cust])

Dissatisfied = CALCULATE('Customer Delight'[Red]/'Customer Delight'[Total Cust])

![image](https://github.com/user-attachments/assets/1c90c951-ba99-4c2b-87b1-183d13e2fde5)
 
For 7016 customers, it can be evaluated that there are 

57.37% Loyal customers,

9.61% Satisfied customers,

18.49% Dissatisfied customers.

It is these 9.61% of customers that the company needs to focus on in order to improve the customer experience and services offered to retain them and convert them to loyal customers.

For the already churned customers, it can be seen that,

![image](https://github.com/user-attachments/assets/8a7286a1-cf65-4854-b8a6-1d7ae8cdf261)

1.29% of the customers were already loyal but churned out due to reasons such as no more need of service or relocation etc.

16.53% of the customers were satisfied and still churned out maybe because they found better services than the current ones.

69.62% of the customers were dissatisfied with the service and discontinued which maybe due to bad service experiences. As already analyzed, the issues with payment method, mainly the electronic check method might be one of the leading reasons for the current churn rate.

All of the above sentimental analysis can be better understood and visualized using the Power BI Dashboard.


### Reporting and Visualization:

#### Objective 5: This objective helps to gather insights through Power BI reports and dashboard which is mainly designed for the end user.

In order to draw actionable insights and make data driven decisions, the pre-processed data from Power BI is used to build reports and dashboards. Several variables are used to visualize various aspects of the dataset. 

Several measures listed below help build the visuals using DAX calculations.

Monthly Contract = CALCULATE(DISTINCTCOUNT(Retention_Dataset[Customer_ID]),Retention_Dataset[Contract] = "Month-to-month")

One Year Contract = CALCULATE(DISTINCTCOUNT(Retention_Dataset[Customer_ID]),Retention_Dataset[Contract] = "One year")

Two Year Contract = CALCULATE(DISTINCTCOUNT(Retention_Dataset[Customer_ID]),Retention_Dataset[Contract] = "Two years")

Retained = CALCULATE(COUNT(Retention_Dataset[Customer_ID]),Retention_Dataset[Churn_Status] = "No churn")

Churned = CALCULATE(COUNT(Retention_Dataset[Customer_ID]),Retention_Dataset[Churn_Status] = "Churn")

The above DAX calculations help in counting number of customers for a particular duration of contract and their churn status. This helps us to analyze  the relation between the contract duration and retention/churn.
The dashboard is as shown below.

![image](https://github.com/user-attachments/assets/9a55d1a8-95dd-458b-ac1a-1dd69e86a71e)

![image](https://github.com/user-attachments/assets/62bec2a6-bc1a-4616-9ddf-c6fa5a053553)


### Findings and suggestions:
•	It is seen that customers who opt for longer term contracts such as One-year and Two-year contracts have lesser churn rates compared to those with month-to-month contracts. Thus, we fail to accept the null hypothesis. It can be suggested from the dashboard that long term contracts lead to lesser churn rate.

It is therefore suggested that when the company approaches the customers for entering into contract, the customers should be convinced to enter into long term contracts by giving more discounts on long term basis.

•	From the MLR output, it is seen that Monthly charges and Tenure have a negative correlation between them.Which means, increase in monthly charges would result in customers opting for lesser tenure for services which would further indicate more churns. From the first finding, it is clear that longer tenure has lesser churn rates. Thus, we fail to accept the 2nd null hypothesis.

It is therefore suggested that the company should continue with the same monthly charges without increasing it in near future.

•	From the reports and dashboards created using Power BI, it is seen that there is not much impact of Senior Citizens on the customer churn or retention. Thus we accept the null hypothesis. Irrespective of the age of customers, efforts should be made to retain them by offering discounts and services.

•	It is seen that customers making payment through Electronic checks are at a greater risk of churning out. This may be due to the glitches that they encounter during the making of payment or delay in payments or bad experience of the user interface.

While the customers who make payments via mailed check, bank transfers and credit cards are having lesser churn rate. Also, it is seen that mailed checks and bank transfer payments happen on automatic basis oftenly for month-to-month contracts.

Thus, more focus should be on retaining the customers who churn out due to issues encountered during the Electronic check method. Thus we fail to accept the null hypothesis.






   














