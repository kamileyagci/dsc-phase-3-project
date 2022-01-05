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

I removed the 'phone number' column from dataset. This variable is not useful. Even though, the beginning three numbers are considered area codes, they may be misleading. Nowadays, with the use of cell phones, the area codes do not always reflect the residence/work location of the person. 

## Model



## Interpret


## Future Work

