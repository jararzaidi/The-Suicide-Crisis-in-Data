#!/usr/bin/env python
# coding: utf-8

# # Suicide Research
# 
# What is the relationship between the age of suicide and the number of suicides?
# Is one age group more inclined to suicide, or is their correlation?
# What is the relationship between the sex and the number of suicides? 
# What underlying patterns do we recognize between each of the categories? 
# Which countries are more inclined to suicide? 
# These are just a few of the questions we will be diving into.
# 

# Things we will Do:

# Peform some basic tasks to explore our dataframe.

# Peform some data visualization using Seaborn 

# Clean / Scrape some data from Kaggle, inorder to answer some questions with it 

#  Feature Engineering which is to create new data columns which help support my data analysis

# In[280]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Read who_suicide_statistics.csv as a dataframe called suici

# In[274]:


csv_file = 'who_suicide_statistics.csv'
suici = pd.read_csv(csv_file)
suici 


# Peform simple tasks/methods to get basic info on the Data Frame

# In[276]:


# check out the head of the DataFrame
suici.head()


# In[90]:


# check out the tail of the DataFrame
suici.tail()


# In[92]:


# Use the .info method to find out how many entires there are
suici.info()


# In[108]:


# The number of columns 
len(suici.columns)


# In[109]:


# The number of rows 
len(suici.index)


# In[110]:


# The Average suicides_no 
suici['suicides_no'].mean()


# In[223]:


# The Maximum/ highest number of suicides
suici['suicides_no'].max()


# In[225]:


# The index of the max
suici.country[suici.suicides_no.idxmax()]


# In[127]:


# The amount of Males & Females used in this data
suici['sex'].value_counts()


# In[ ]:





# In[132]:


# The different age groups
suici['age'].unique()


# In[133]:


# the Number of different age groups
suici['age'].nunique()


# In[135]:


# the Number of entries for each Country
suici['country'].value_counts()


# In[150]:


# Data of U.S. 
suici[suici['country'] == 'United States of America']


# In[154]:


suici[suici['country'] == 'United States of America']


# Data Visulization Before Scaping with Seaborn

# In[ ]:


sns.set_style('whitegrid')


# In[242]:


sns.jointplot(x='population',y='suicides_no',data=suici,alpha=0.2)


# In[260]:


# Lm Plot showing that ages 35-54 has the highest amount of suicies
sns.lmplot(x='population',y='suicides_no',data=suici,col='age',hue='sex',aspect=0.5,height=3)


# In[257]:


# Box Plot showing that ages 35-54 has the highest amount of suicies
sns.boxplot(x='age',y='suicides_no',data=suici,palette='rainbow')


# In[263]:


# This boxplot shows that males are more inclined to suicide
sns.boxplot(x='sex',y='suicides_no',data=suici,palette='rainbow')


#  Clean / Scrape some data from Kaggle, inorder to answer some questions with it 

# In[297]:


#.dropna() is a method which drops rows where atleast 1 item is missing
# this will eliminate filler rows with a suicides_no of "NA"

suici = suici.dropna()
suici


# Feature Engineering which is to create new data columns which help support my data analysis

# In[442]:


sample = suici.sample(5)


# In[443]:


sample['Youth or Adult'] = sample['age'].str[:2]
sample


# In[444]:


def AgeGroup(suici):
    sample['Youth or Adult'] = sample['age'].str[:2]
    sample['Type'] = 'None'
    sample['Type'][sample['Youth or Adult'].astype(int) >= 25 ] = 'Adult'
    sample['Type'][sample['Youth or Adult'].astype(int) < 25 ] = 'Adolescent'
    return sample


# Feature Engineering used where we create a new data column to support my analysis

# In[445]:


suici = AgeGroup(suici)
suici.sample(5)


# Now we must use the new Columns  we created to confirm our analysis!

# In[455]:


sns.boxplot(x='Type',y='suicides_no',data=suici,palette='rainbow')


# In[454]:



sns.boxplot(x='age',y='suicides_no',data=suici,palette='rainbow')


# In[457]:


# Lm Plot showing that ages 35-54 has the highest amount of suicies
sns.lmplot(x='population',y='suicides_no',data=suici,col='age',hue='Type',aspect=0.5,height=3)


# In[ ]:





# It's apparent that the data visualizations above illustrates that Adults are at a higher risk of committing suicide. 
# From the box plot above, it shows that the age group of 35-54 years old had the greatest number of suicides.
# With the 15-24 age group being the 2nd most inclined, while 75+ years had the least number of suicides.

# In[ ]:





# In[ ]:




