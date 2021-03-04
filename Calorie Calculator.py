#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This program calculates your daily caloric intake based on your macros.


# In[ ]:


carbohydrates=int(input("Carbohydrates in grams "))
protein=int(input("Protein in grams "))
fat=int(input("Fat in grams "))
alcohol=int(input("Alcohol in grams "))

Daily_Caloric_Intake=(carbohydrates*4)+(protein*4)+(fat*9)+(alcohol*9)

print("Your caloric intake today is", Daily_Caloric_Intake)

