#!/usr/bin/env python
# coding: utf-8

# # Tutorial 8: Deep Energy-Based Generative Models
# 
# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)
# 
# **Filled notebook:** 
# [![View on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial8/Deep_Energy_Models.ipynb)
# [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial8/Deep_Energy_Models.ipynb)  
# **Pre-trained models:** 
# [![View files on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/phlippe/saved_models/tree/main/tutorial8)
# [![GoogleDrive](https://img.shields.io/static/v1.svg?logo=google-drive&logoColor=yellow&label=GDrive&message=Download&color=yellow)](https://drive.google.com/drive/folders/11ZI7x2sfCNtaZUNpe4v08YXWN870spXs?usp=sharing)  
# **Recordings:** 
# [![YouTube - Part 1](https://img.shields.io/static/v1.svg?logo=youtube&label=YouTube&message=Part%201&color=red)](https://youtu.be/E6PDwquBBQc)
# [![YouTube - Part 2](https://img.shields.io/static/v1.svg?logo=youtube&label=YouTube&message=Part%202&color=red)](https://youtu.be/QJ94zuSQoP4)    
# **Author:** Phillip Lippe

# ## Load Libraries

# In[1]:


from Classifier.Evaluation import evaluate
from DataHandlers.DataLoading import loadOriginalTrainSet, loadTestSet, loadGeneratedSet

## PyTorch
import torch
from DataHandlers.DetailsDisplayer import detailsDisplayer
from ShowImages import showImages


# ## Choice of data

# In[2]:


USE_ORIGINAL_DATA = False # If false, the classifier gets trained only on the generated set


# ## Used Device

# In[3]:


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


# ## Load Data

# ### Load test loader

# In[4]:


test_loader = loadTestSet()
detailsDisplayer(test_loader)


# ### Load train loader

# In[5]:


if USE_ORIGINAL_DATA:
    train_loader = loadOriginalTrainSet()
    print('Using the original train set')
else:
    train_loader = loadGeneratedSet()
    print('Using the generated set')

detailsDisplayer(train_loader)


# ## Display train images

# In[6]:


showImages(train_loader.dataset, "Generated Images")


# ## Classify

# In[7]:


evaluate(train_loader, test_loader, device)


# ---
# 
# [![Star our repository](https://img.shields.io/static/v1.svg?logo=star&label=⭐&message=Star%20Our%20Repository&color=yellow)](https://github.com/phlippe/uvadlc_notebooks/)  If you found this tutorial helpful, consider ⭐-ing our repository.    
# [![Ask questions](https://img.shields.io/static/v1.svg?logo=star&label=❔&message=Ask%20Questions&color=9cf)](https://github.com/phlippe/uvadlc_notebooks/issues)  For any questions, typos, or bugs that you found, please raise an issue on GitHub. 
# 
# ---
