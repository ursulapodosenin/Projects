#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This dataset provides a comprehensive look at competitive meets held across various regions, capturing details such as meet locations, dates, and unique identifiers.
# By analyzing this data, I explore patterns in meet frequency by country, state, and town, gaining insights into geographical trends and potential regional popularity of events. 
# Additionally, temporal aspects, such as meet IDs over time, offer a glimpse into the history and scheduling patterns of these events. 
# Through statistical summaries and visualizations, I aim to find key trends, identify high-frequency locations, and better understand the distribution and structure of meets, which can provide valuable insights for organizers, participants, and regional stakeholders.


# In[2]:


# Loading the packages I will be using
import os
import numpy as np
import pandas as pd
import matplotlib as mp
import seaborn as sns
import statsmodels as st
import matplotlib.pyplot as plt


# In[3]:


# Printing the current working directory 
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[5]:


# Reading the file 
df= pd.read_csv('/Users/ursulapodosenin/Desktop/meets.csv')
df.head()


# In[6]:


# Getting summary statistics including means, medians, quartiles
summary_stats = df.describe(include='all').T
summary_stats['median'] = df.median(numeric_only=True)
summary_stats['q1'] = df.quantile(0.25, numeric_only=True)
summary_stats['q3'] = df.quantile(0.75, numeric_only=True)

# Evaluating the missing value information
missing_values = df.isnull().sum()

# Looking at ata types and unique counts
data_info = pd.DataFrame({
    'Data Type': df.dtypes,
    'Unique Values': df.nunique(),
    'Missing Values': missing_values,
})
print("Summary Statistics:")
print(summary_stats)
print("\nMissing Values Information:")
print(missing_values)
print("\nAdditional Data Information:")
print(data_info)


# In[8]:


# Dropping duplicates values
df.drop_duplicates(inplace=True)

# Managing missing values
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

# Filling incmissing values in categorical columns with 'Unknown'
df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# Filling in missing values in numerical columns with the median
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Converting date columns to datetime format
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Check for any rows where 'Date' could not be converted and replace with a placeholder if necessary
    df['Date'] = df['Date'].fillna(pd.to_datetime('1900-01-01'))

# Standardizing text data 
for col in categorical_columns:
    df[col] = df[col].astype(str).str.lower()

# Removing any outliers in the data
for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds and filter outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
df.reset_index(drop=True, inplace=True)
print("Cleaned Data Preview:")
print(df.head())


# In[9]:


# Creating a bar plot for Top 10 Meet Countries
top_countries = df['MeetCountry'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Top 10 Countries by Meet Frequency', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Meets', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Frequency'], loc='upper right', fontsize=10)

# Creating a scatter plot of MeetID vs Date
plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], df['MeetID'], color='purple', marker='o', edgecolor='black', s=30, alpha=0.6)
plt.title('MeetID Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Meet ID', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Meet IDs'], loc='upper left', fontsize=10)
plt.annotate('Earliest Meet', xy=(df['Date'].min(), df['MeetID'].min()), 
             xytext=(df['Date'].min(), df['MeetID'].min() + 1000),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='red')

# Creating a multi-plot for Meet Frequency by State
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
top_states = df['MeetState'].value_counts().head(10)
top_states.plot(kind='bar', color='green', edgecolor='black', linewidth=1.2)
plt.title('Top 10 States by Meet Frequency', fontsize=14, fontweight='bold')
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Meets', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.subplot(1, 2, 2)
top_towns = df['MeetTown'].value_counts().head(10)
top_towns.plot(kind='bar', color='coral', edgecolor='black', linewidth=1.2)
plt.title('Top 10 Towns by Meet Frequency', fontsize=14, fontweight='bold')
plt.xlabel('Town', fontsize=12)
plt.ylabel('Number of Meets', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.figlegend(['Meet Frequency'], loc='upper center', fontsize=12, ncol=2)
plt.tight_layout()
plt.show()


# In[10]:


# Setting up Seaborn styles for the plots
sns.set(style="whitegrid")

# Bar plot for Top 10 Meet Countries
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.index, y=top_countries.values, palette="Blues_d", edgecolor='black')
plt.title('Top 10 Countries by Meet Frequency', fontsize=16, fontweight='bold')
plt.xlabel('Country', fontsize=14)
plt.ylabel('Number of Meets', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Frequency'], loc='upper right', fontsize=10)

# Scatter plot of MeetID vs Date
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['Date'], y=df['MeetID'], color='purple', marker='o', edgecolor='black', s=60, alpha=0.6)
plt.title('MeetID Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Meet ID', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Meet IDs'], loc='upper left', fontsize=10)
plt.annotate('Earliest Meet', xy=(df['Date'].min(), df['MeetID'].min()), 
             xytext=(df['Date'].min(), df['MeetID'].min() + 1000),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='red')

# Multi-plot for Meet Frequency by State and Town
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
sns.barplot(x=top_states.index, y=top_states.values, palette="Greens_d", edgecolor='black')
plt.title('Top 10 States by Meet Frequency', fontsize=14, fontweight='bold')
plt.xlabel('State', fontsize=12)
plt.ylabel('Number of Meets', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.subplot(1, 2, 2)
sns.barplot(x=top_towns.index, y=top_towns.values, palette="Oranges_d", edgecolor='black')
plt.title('Top 10 Towns by Meet Frequency', fontsize=14, fontweight='bold')
plt.xlabel('Town', fontsize=12)
plt.ylabel('Number of Meets', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.figlegend(['Meet Frequency'], loc='upper center', fontsize=12, ncol=2)
plt.tight_layout()
plt.show()


# In[ ]:


# Creating plots in Matplotlib and Seaborn differ in terms of syntax simplicity, styling, and default aesthetics. 
# While Matplotlib provides a foundational, versatile library with more manual control over plot elements, it often requires additional lines of code for customization, such as setting color schemes and adjusting layout properties. 
# Seaborn, on the other hand, is built on top of Matplotlib and offers high-level abstractions, making it quicker to produce aesthetically pleasing plots with less code. 
# For instance, Seaborn automatically applies themes and color palettes, which streamline customization, and includes specialized functions that simplify creating plots like bar and scatter plots with advanced styling, such as smooth color gradients and refined grid layouts. 
# In the above plots, Seaborn helped produce smoother color schemes and grid styling with minimal effort, while Matplotlib required more manual adjustments to achieve similar visual effects.


# In[ ]:


# In conclusion, the analysis of this dataset reveals clear trends in the distribution and frequency of competitive meets across different geographical regions and over time. 
# High-frequency meet locations identified by country, state, and town suggest certain regions are central to hosting these events, potentially reflecting local interest and infrastructure for competitions. 
# The temporal analysis of meet IDs provides insights into event scheduling and growth, indicating an expansion or consistency in meet organization over the years. 
# These findings could help organizers in targeting locations with high demand or exploring underserved areas to broaden event reach. 
# Overall, the data underscores the geographic and temporal dynamics of competitive meets, providing a basis for informed decision-making in future event planning and resource allocation.

