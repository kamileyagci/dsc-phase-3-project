# SyriaTel Customer Churn Study

**Author: Kamile Yagci**


## Overview
In this study, I will analyze the 'SyriaTel Customer Churn' data. The SyriaTel is a telecommunication company. The purpose of the study is to predict whether a customer will ("soon") stop doing business with SyriaTel.

## Business Problem
The telecommincation company, SyriaTel, hired me to analyze the Chustomer Churn data. The company wants to understand the customer's decision to discontinue their business with SyriaTel. The results of the analysis will be used make business decisions for improving the company finances.

This study will

Search for the predictable pattern for customer decision on stop or continue doing business with SyriaTel
Choose a model which will best identify the customers who will stop doing business with SyriaTel


## Method

I followed the following steps in this project:

1. Data
    * Load
    * Scrub/Explore
3. Model
    * Pre-Process
    * Evaulation Metrics
    * Logictic Regression
    * K-Nearest Neighbor
    * Decision Trees
    * Random Forest
    * XGBoost
3. Interpret 
4. Future work


## Data

### Load
I used SyriaTel Customer Churn data for this study. The data file is downloaded from Kaggle.

The file name is 'bigml_59c28831336c6604c800002a.csv'.

Tha raw data has 3333 entries and 21 columns.

The column/variable names are:
* state            
* account length
* area code
* phone number 
* international plan
* voice mail plan
* number vmail messages
* total day minutes
* total day calls
* total day charge
* total eve minutes
* total eve calls
* total eve charge
* total night minutes
* total night calls
* total night charge
* total intl minutes
* total intl calls
* total intl charge
* customer service calls
* churn

The data doesn't have any missing values.

I removed the column 'phone number' from dataset. Most digits in the phone number is random, and it will not have much use in modeling. This variable will also be a problem in dummy variable creation, because all values will be unique.

The distribution of variables:
<img src="/images/histograms_All.png" width=1500/>

The target variable for this study is 'churn'. The rest of the variables in the dataset will be predictors. 

'churn': activity of customers leaving the company and discarding the services offered

The scatter graphs for 'churn' vs predictors:
![Scatters](/images/scatters_All.png)


## Model

### Pre-process

In this study, we are trying to predict customer's decision on stopping the business with the company. The prediction will be True (1) or False (1). Therefore we will use binary classification model.

Before modeling, I divided the dataset into target data series (y) and predictor dataframe (X).

y: DataSeries of 'churn' 
X: DataFrame of all predictors

I also created dummy variables from categorical variables. The ned X DataFrame has 73 variables together with dummies.

Then, I seperated the data into train and test splits. I allocated 25% of the data for testing. I also assigned a random state for repeatability.

The shape of the splits:

X_train shape =  (2499, 73)
y_train shape =  (2499,)
X_test shape =  (834, 73)
y_test shape =  (834,)

The next step is standardization. The data values have different ranges, so I did normalize/scale each variable in train and test data (X) before modeling. I used Scikit-Learn StandardScaler.

### Evaluation Metrics

In the next steps, I will use several classifiers to model the data. I will check their performance using the evaluation metrics:

precision: 
* Number of True Positives / Number of Predicted Positives
* How precise our predictions are?

recall: 
* Nuber of True Positives / Number of Actual Total Positives
* What percentage of the classes we're interested in were actually captured by the model?

accuracy: 
* (Number of True Positives + Number of True Negatives) / (Number of Total Observations)
* Out of all the predictions our model made, what percentage were correct?

f1-score: 
* 2 * (Precision * Recall) / (Precision + Recall)
* Harmonic Mean of Precision and Recall.

*Source: Flatiron Data Science Curriculum, Evaluation Metrics*

Since my business problem is focusing on identfying the customers who stop doing business, I am interested mainly on the 'recall' metrics. However, when optimizing my model, I should also pay attention to the 'precision'. I want my predictions to be true, to be precise. The recall and precision are inversely proportional. Therefore, I choose to  use the f1-score, Harmonic Mean of Precision and Recall, as the main metric for evaluating the performance of the model.

### Logistic Regression

I started with Logistic Regression. I instantiated the model with default parameters and fit on training data. Then I checked the evaluation metrics both for training and testing data.

| | f1-score | recall |
| :- | -: | :-: |
| Train | 0.37 | .27 
| Test | 0.32 | .22


## Interpret


## Future Work

