#!/usr/bin/env python
# coding: utf-8

# In[68]:


# Loading the packages I will be using
import os
import numpy as np
import pandas as pd
import matplotlib as mp
import seaborn as sns
import statsmodels as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import stats
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D


# In[69]:


# Getting my working directory 
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[86]:


# Reading the data from the csv file into a data frame
df= pd.read_csv('/Users/ursulapodosenin/Desktop/2019_games_athletes.csv')
df.head()


# In[71]:


# Dropping rows with missing values
columns_to_check = ['overallscore']
df_cleaned = df.dropna(subset=columns_to_check)

# Removing duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Converting overallrank to numeric and coercing invalid values to Na's
df_cleaned['overallrank'] = pd.to_numeric(df_cleaned['overallrank'], errors='coerce')

# Droping columns that won't be used 
columns_to_drop = ['profilepics3key', 'bibid', 'affiliateid', 'countryoforiginname', 'affiliatename', 'status', 'competitorid', 'firstname', 'age', 'lastname', 'countryoforigincode', 'competitorname', 'overallscore', 'division']
df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')

# Filtering out invalid or extreme values for height and weight
df_cleaned = df_cleaned[df_cleaned['height'] > 1.0]  
df_cleaned = df_cleaned[df_cleaned['weight'] > 0]  

# Handling inconsistent gender values
valid_genders = ['M', 'F']
df_cleaned = df_cleaned[df_cleaned['gender'].isin(valid_genders)]

# Resetting the index
df_cleaned.reset_index(drop=True, inplace=True)

df.head()


# In[72]:


# Printing the total number of individuals per gender
total_per_gender = df.groupby('gender').size()
print(total_per_gender)


# In[73]:


# Getting the average height by gender
grouped_stats_height = df.groupby('gender')['height'].agg(['mean', 'median', 'std'])
print(grouped_stats_height)


# In[74]:


# Looking for outliers in height 
q1 = df['height'].quantile(0.25)
q3 = df['height'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df['height'] < lower) | (df['height'] > upper)]
print(outliers)


# In[75]:


# Dropping rows with height 0
df = df[df['height'] != 0]

# Separaing the data for men and women
df_men = df[df['gender'] == 'M']
df_women = df[df['gender'] == 'F']

# Creating a plot
plt.figure(figsize=(12, 6))

# Plotting the distribution for men
plt.subplot(1, 2, 1)
sns.histplot(df_men['height'], bins=30, kde=True, color='blue')
plt.title('Height Distribution for Men')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')

# Plotting the distribution for women
plt.subplot(1, 2, 2)
sns.histplot(df_women['height'], bins=30, kde=True, color='pink')
plt.title('Height Distribution for Women')
plt.xlabel('Height (m)')
plt.ylabel('Frequency')

# Printing the plots
plt.tight_layout()
plt.show()


# In[76]:


# Getting the average weight for each gender
grouped_stats_weight= df.groupby('gender')['weight'].agg(['mean', 'median', 'std'])
print(grouped_stats_weight)


# In[77]:


# Looking for outliers in the weight
q1 = df['weight'].quantile(0.25)
q3 = df['weight'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df['weight'] < lower) | (df['weight'] > upper)]
print(outliers)


# In[78]:


# Plotting the graph
plt.figure(figsize=(12, 6))

# Plotting the distribution for men
plt.subplot(1, 2, 1)
sns.histplot(df[df['gender'] == 'M']['weight'], bins=10, kde=True, color='blue')
plt.title('Weight Distribution for Men')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')

# Plotting the distribution for women 
plt.subplot(1, 2, 2)
sns.histplot(df[df['gender'] == 'F']['weight'], bins=10, kde=True, color='pink')
plt.title('Weight Distribution for Women')
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')

# Printing the plots
plt.tight_layout()
plt.show()


# In[79]:


# Findinging the gender count
print(df['gender'].unique())

# Making sure the gender column is in a string format
df['gender'] = df['gender'].astype(str)

# Making sure all the values for gender are uppercase
df['gender'] = df['gender'].str.strip().str.upper()

# Finding the rows where the gender has the value 'X' and filtering to 'M', 'F' values only
df = df[df['gender'].isin(['M', 'F'])]

# Mapping 'M' to 1 and 'F' to 0
df['gender'] = df['gender'].map({'M': 1, 'F': 0})

# Looking at the results
df.head()


# In[84]:


# Ensuring that 'height', 'weight', and 'overallrank' columns are numeric
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['overallrank'] = pd.to_numeric(df['overallrank'], errors='coerce')

# Dropping rows with Na values
df = df.dropna(subset=['height', 'weight', 'overallrank'])

# Filtering out rows with height less than 1
df = df[df['height'] >= 1.0]

# Finding the correlation between height and overallrank
try:
    overall_corr_height, overall_p_height = pearsonr(df['height'], df['overallrank'])
    print(f"Overall Correlation (Height vs OverallRank): {overall_corr_height:.2f}, P-value: {overall_p_height:.2e}")
except Exception as e:
    print(f"Error calculating correlation for height: {e}")

# Finding the correlation between weight and overallrank
try:
    overall_corr_weight, overall_p_weight = pearsonr(df['weight'], df['overallrank'])
    print(f"Overall Correlation (Weight vs OverallRank): {overall_corr_weight:.2f}, P-value: {overall_p_weight:.2e}")
except Exception as e:
    print(f"Error calculating correlation for weight: {e}")

# Finding the correlation for each gender
gender_groups = df.groupby('gender')
for gender, group in gender_groups:
    try:
        gender_corr_height, gender_p_height = pearsonr(group['height'], group['overallrank'])
        print(f"Gender: {gender}, Correlation (Height vs OverallRank): {gender_corr_height:.2f}, P-value: {gender_p_height:.2e}")
    except Exception as e:
        print(f"Error calculating correlation for gender {gender} (height): {e}")
    try:
        gender_corr_weight, gender_p_weight = pearsonr(group['weight'], group['overallrank'])
        print(f"Gender: {gender}, Correlation (Weight vs OverallRank): {gender_corr_weight:.2f}, P-value: {gender_p_weight:.2e}")
    except Exception as e:
        print(f"Error calculating correlation for gender {gender} (weight): {e}")

# Creating scatterplots with regression lines
plt.figure(figsize=(12, 6))
sns.lmplot(x='height', y='overallrank', hue='gender', data=df, aspect=1.5, ci=None, scatter_kws={'alpha': 0.6})
plt.title('Height vs OverallRank by Gender in CrossFit Rankings')
plt.xlabel('Height')
plt.ylabel('OverallRank')
plt.show()

plt.figure(figsize=(12, 6))
sns.lmplot(x='weight', y='overallrank', hue='gender', data=df, aspect=1.5, ci=None, scatter_kws={'alpha': 0.6})
plt.title('Weight vs OverallRank by Gender in CrossFit Rankings')
plt.xlabel('Weight')
plt.ylabel('OverallRank')
plt.show()


# In[87]:


# Removing rows where gender is 'X'
df = df[df['gender'] != 'X']

# Ensuring numeric columns have proper numeric values
df['overallrank'] = pd.to_numeric(df['overallrank'], errors='coerce')
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# Dropping rows with missing values in required columns
df = df.dropna(subset=['overallrank', 'height', 'weight'])

# Separate the data by gender
df_male = df[df['gender'] == 'M']
df_female = df[df['gender'] == 'F']

# Check if male data is non-empty before fitting the model
if not df_male.empty:
    X_male_height = sm.add_constant(df_male['height'])  # Adding intercept
    y_male = df_male['overallrank']
    model_male_height = sm.OLS(y_male, X_male_height).fit()
    print("Male Height Model Summary:")
    print(model_male_height.summary())

    X_male_weight = sm.add_constant(df_male['weight'])
    model_male_weight = sm.OLS(y_male, X_male_weight).fit()
    print("Male Weight Model Summary:")
    print(model_male_weight.summary())
else:
    print("No male data available for modeling.")

# Check if female data is non-empty before fitting the model
if not df_female.empty:
    X_female_height = sm.add_constant(df_female['height'])  # Adding intercept
    y_female = df_female['overallrank']
    model_female_height = sm.OLS(y_female, X_female_height).fit()
    print("Female Height Model Summary:")
    print(model_female_height.summary())

    X_female_weight = sm.add_constant(df_female['weight'])
    model_female_weight = sm.OLS(y_female, X_female_weight).fit()
    print("Female Weight Model Summary:")
    print(model_female_weight.summary())
else:
    print("No female data available for modeling.")


# In[88]:


# Converting columns to numeric values
df['overallrank'] = pd.to_numeric(df['overallrank'], errors='coerce')  # Convert to numeric
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# Filtering out rows where height is below 1
df = df[df['height'] >= 1.0]

# Dropping rows with missing values
cleaned_data = df.dropna(subset=['overallrank', 'height', 'weight'])

# Splitting the data by gender
men_data = cleaned_data[cleaned_data['gender'] == 'M']
women_data = cleaned_data[cleaned_data['gender'] == 'F']

# Creating a model for men
X_men = men_data[['height', 'weight']]
y_men = men_data['overallrank']
X_men = sm.add_constant(X_men) 
model_men = sm.OLS(y_men, X_men).fit()

# Creating a model for women
X_women = women_data[['height', 'weight']]
y_women = women_data['overallrank']
X_women = sm.add_constant(X_women)  # Add intercept
model_women = sm.OLS(y_women, X_women).fit()

# Printing the regression model 
print(model_men.summary())
print(model_women.summary())


# In[89]:


# Creating a multiple regression function
def fit_multiple_regression(data, predictors, target):
    X = data[predictors]
    X = sm.add_constant(X) 
    y = data[target]
    model = sm.OLS(y, X).fit()
    return model

# Fitting a multiple regression model for men and women
model_male = fit_multiple_regression(df_male, ['height', 'weight'], 'overallrank')
model_female = fit_multiple_regression(df_female, ['height', 'weight'], 'overallrank')

# Printting the regression models
print(model_male.summary())
print(model_female.summary())

# Storing the predictions
df_male['predicted_overallrank'] = model_male.predict(sm.add_constant(df_male[['height', 'weight']]))
df_female['predicted_overallrank'] = model_female.predict(sm.add_constant(df_female[['height', 'weight']]))

# Creating a plot
plt.figure(figsize=(12, 6))

# Creating a scatter plot with a regression line for men
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_male, x='height', y='overallrank', color='blue', label='Male Actual')
sns.lineplot(data=df_male, x='height', y='predicted_overallrank', color='red', label='Male Predicted')
plt.title("Male: Height + Weight vs OverallRank")
plt.xlabel('Height')
plt.ylabel('Overall Rank')
plt.legend()

# Creating a scatter plot with a regression line for women
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_female, x='height', y='overallrank', color='red', label='Female Actual')
sns.lineplot(data=df_female, x='height', y='predicted_overallrank', color='blue', label='Female Predicted')
plt.title("Female: Height + Weight vs OverallRank")
plt.xlabel('Height')
plt.ylabel('Overall Rank')
plt.legend()

# Printing the plot
plt.tight_layout()
plt.show()

# Creating a plot of the residuals 
plt.figure(figsize=(12, 6))

# Creating a residual plot for men 
plt.subplot(1, 2, 1)
sns.residplot(x=model_male.predict(), y=model_male.resid, lowess=True, color='blue')
plt.title("Residual Plot for Males (Height + Weight vs OverallRank)")
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# # Creating a residual plot for women
plt.subplot(1, 2, 2)
sns.residplot(x=model_female.predict(), y=model_female.resid, lowess=True, color='red')
plt.title("Residual Plot for Females (Height + Weight vs OverallRank)")
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

# Printing the plot
plt.tight_layout()
plt.show()

