#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[48]:


movie_review = pd.read_csv("/Users/madankc/Desktop/Sentiment analysis/movie review/movie-sentiment-polarity.csv")


# In[49]:


movie_review


# In[50]:


movie_review


# In[51]:


X_review = movie_review.iloc[:,0]


# In[52]:


X_review


# In[53]:


y_sentiment = movie_review.iloc[:,1]


# In[54]:


y_sentiment


# In[55]:


tfidf = TfidfVectorizer()
matrix_X = tfidf.fit_transform(X_review)


# In[56]:


matrix_X


# In[57]:


matrix_X.toarray()


# In[58]:


train_X, test_X, train_y, test_y = train_test_split(matrix_X, y_sentiment, shuffle = "True", test_size = 0.2)


# In[59]:


train_X


# In[60]:


test_X


# In[ ]:





# In[61]:


#Random Forest Classifier
model_RF = RandomForestClassifier().fit(train_X, train_y)
y_predict_RF = model_RF.predict(test_X)
accuracy_RF = accuracy_score(test_y, y_predict_RF)
accuracy_RF


# In[62]:


#Naive Bayes Classifier
model_NB = MultinomialNB().fit(train_X, train_y)
y_predict_NB = model_NB.predict(test_X)
accuracy = accuracy_score(test_y, y_predict_NB)
accuracy


# In[63]:


# Logistic Regression
model_LR = LogisticRegression().fit(train_X, train_y)
y_predict_LR = model_LR.predict(test_X)
accuracy = accuracy_score(test_y, y_predict_LR)
accuracy


# In[64]:


#KNeigbors Classifier 
model_KN = KNeighborsClassifier().fit(train_X, train_y)
y_pred_KN = model_KN.predict(test_X)
accuracy = accuracy_score(test_y, y_pred_KN)
accuracy


# In[66]:


#SVC Classifier
model_SVC = SVC().fit(train_X, train_y)
y_pred_SVC = model_SVC.predict(test_X)
accuracy = accuracy_score(test_y, y_pred_SVC)
accuracy


# In[67]:


#Decision Tree Classifier
model_DT = DecisionTreeClassifier().fit(train_X, train_y)
y_pred_DT = model_DT.predict(test_X)
accuracy_DT = accuracy_score(test_y, y_pred_DT)
accuracy_DT


# In[ ]:


#While training the model and predicting the sentiments of the movie reviews using multiple classifiers above, I found Naive-Bayes Classifier has the maximum accuracy of the prediction and Decision Tree Classifier has the least.

