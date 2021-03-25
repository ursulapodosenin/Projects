#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This code requests the user's Crossfit id number and uses web scraping toolkits to extract the user's profile picture.


# In[4]:


import requests
import bs4
from bs4 import BeautifulSoup as bs


# In[ ]:


Crossfit_ID_Number= input('Input Crossfit ID Number: ')
url= 'https://games.crossfit.com/athlete/'+ Crossfit_ID_Number
r= requests.get(url)
soup= bs(r.content, 'html.parser')
Profile_Picture= soup.find('img',{'class': 'pic img-circle'})['src']
print(Profile_Picture)


# In[ ]:




