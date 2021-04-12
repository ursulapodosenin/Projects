#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[ ]:


def main():
    i= 0
    path= "C:/Users/Ursula Podosenin/Desktop/Sample_Folder"
    for filename in os.listdir(path):
        named_files= "document"+ str(i)+ ".pdf"
        file_source= path+filename
        file_destination= path+file_destination
        os.rename(file_source, file_destination)
        i+=1
        
if __name__ == '__main__':
    main()

