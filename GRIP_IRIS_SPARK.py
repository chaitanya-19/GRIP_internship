#!/usr/bin/env python
# coding: utf-8

# # Task # 4 - To Explore Decision Tree Algorithm
# 

# ### 1. Defining the problem statement

# *For the given ‘Iris’ dataset, create the Decision Tree classifier and
# visualize it graphically. The purpose is if we feed any new data to this
# classifier, it would be able to predict the right class accordingly.*

# In[1]:


from IPython.display import Image
Image(url= "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png")


# ### 2. Collecting the data
# training data set and testing data set are avaliable in the sklearn datasets.we can directly load the dataset.

# In[2]:


from sklearn.datasets import load_iris
#load the dataset
data_raw= load_iris()
data_raw


# ### 3. Exploratory data analysis

# In[3]:


#converting data into dataframe
import seaborn as sb
data=sb.load_dataset("iris")
data


# In[4]:


#data for traing and testing
#data_x represents the data required for building the decision tree
import pandas as pd
data_x=pd.DataFrame(data_raw.data,columns=data_raw.feature_names)
data_x


# In[5]:


#data for test and training encoded or converted
#data y represents the target variable or the decision
data_y=pd.DataFrame(data_raw.target,columns=["species"])

data_y


# In[6]:


data.info()


# We can see the data type of various columns

# In[7]:


data.describe()


# We can see various statistical descriptin of the data

# In[8]:


data.isnull().sum()


# As we can see there are no null values

#  
# **0-'setosa'**
# 
# **1-'versicolor'**
# 
# **2-'virginica'**

# ### 4] Visulization of the data

# In[9]:


import matplotlib.pyplot as plt
#plotting a graph 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('species')
plt.xlabel('sepal length')
plt.scatter(data_x[("sepal length (cm)")],data[("species")],label="sepal length (cm)",color="red")

plt.legend()


# **Insights gained**
# 
# This graph represents the distribution of the sepal lenght of the data set with respect to their target
# 
# As we can see the 'setosa' species sepal lenght lies between 4 to 6 
# 
# As we can see the 'versicolor' species sepal lenght lies between 4.7 to 7.3 
# 
# As we can see the 'virginica' species sepal lenght lies between 5.5 to 8 
# 

# In[14]:


import matplotlib.pyplot as plt
#plotting a graph 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('species')
plt.xlabel('sepal_width	')
plt.scatter(data_x[("sepal width (cm)")],data[("species")],label="sepal width (cm)",color="blue")
plt.legend()


# **Insights gained**
# 
# This graph represents the distribution of the sepal width of the data set with respect to their target
# 
# As we can see the 'setosa' species sepal width lies between 2.9 to 4.5 
# 
# As we can see the 'versicolor' species sepal width  lies between 2 to 3.5 
# 
# As we can see the 'virginica' species sepal width lies between 2.5 to 4

# In[11]:


import matplotlib.pyplot as plt
#plotting a graph 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('species')
plt.xlabel('petal length (cm)')
plt.scatter(data_x[("petal length (cm)")],data[("species")],label="petal length (cm)",color="green")
plt.legend()


# **Insights gained**
# 
# This graph represents the distribution of the petal lenght of the data set with respect to their target
# 
# As we can see the 'setosa' species petal lenght lies between 1 to 2 
# 
# As we can see the 'versicolor' petal sepal lenght lies between 3 to 5.5 
# 
# As we can see the 'virginica' species petal lenght lies between 4.5 to 7 

# In[12]:


import matplotlib.pyplot as plt
#plotting a graph 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ylabel('species')
plt.xlabel('petal width (cm)')
plt.scatter(data_x[("petal width (cm)")],data[("species")],label="petal width (cm)",color="yellow")
plt.legend()


# **Insights gained**
# 
# This graph represents the distribution of the petal width of the data set with respect to their target
# 
# As we can see the 'setosa' species petal width lies between 0.1 to 0.6 
# 
# As we can see the 'versicolor' petal sepal width  lies between 1 to 1.9 
# 
# As we can see the 'virginica' petal sepal width lies between 1.5 to 2.5

# In[13]:


data.hist(bins=50, figsize=(20,15))
plt.show()


# The above graph represents the distribution of various columns 

# ### 5]Creating a decision tree

# In[15]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()


# In[17]:


#First we split the data into training and testing data to evaluate the performance of our model
#splitting the data into testing and traingin datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_x,data[("species")],random_state=42,test_size=0.2)


# In[18]:


x_train


# In[19]:


x_train.shape


# In[20]:


x_test


# In[21]:


x_test.shape


# As expected and mentioned 20% of our total data is used for testing

# In[22]:


dtree.fit(x_train,y_train)


# In[23]:


y_pred=dtree.predict(x_test)


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))


# In[34]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score


# As we can see we have **100%** acuraccy

# ### 6]visualize the Decision Tree to understand it better.

# In[43]:


from sklearn import tree
plt.figure(figsize=(25,10))
tree.plot_tree(dtree, filled=True, 
              rounded=True, 
              fontsize=14)


# In[ ]:




