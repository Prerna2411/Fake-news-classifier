#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import sklearn
print(sklearn.__version__)
get_ipython().system('pip install --upgrade scikit-learn')


# In[3]:


df=pd.read_csv('train.csv')


# In[4]:


df.head()


# In[5]:


#get independent features
X=df.drop('label',axis=1)
X


# In[6]:


Y=df['label']
Y


# In[7]:


df.shape


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer


# In[9]:


df=df.dropna()#drop nan
df


# In[10]:


messages=df.copy()
messages.head(10)


# In[11]:


messages.reset_index(inplace=True)
messages.head()


# 
# 

# In[12]:


messages['title'][6]


# In[13]:


#text preprocessing
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-z-A-Z]',' ',messages['title'][i])
    review=review.lower()
    review=review.split()
    
   
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
    


# In[14]:


print(corpus)


# In[35]:


#applying countvectorizer
#creating bow
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
X=cv.fit_transform(corpus).toarray()
d=cv.get_feature_names_out()[:20]
print(d)


# In[16]:


X.shape


# In[17]:


X


# In[18]:


y=messages['label']
y


# In[19]:


#
from sklearn.model_selection  import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)


# In[36]:


cv.get_feature_names_out()[:20]


# In[21]:


cv.get_params()


# In[45]:


count_df=pd.DataFrame(X_train,columns=cv.get_feature_names_out())


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    


# # MULTINOMIALNB ALGORITHM

# In[26]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[27]:


from sklearn import metrics
import numpy as np
import itertools


# In[28]:


classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy score: %0.3f" % score)
cm=metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes=['Fake','Real'])


# # Passive aggressive classifier algorithm(for text data)

# In[29]:


from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf=PassiveAggressiveClassifier(max_iter=50)


# In[30]:


linear_clf.fit(X_train,y_train)
pred=linear_clf.predict(X_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy score: %0.3f" % score)
cm=metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes=['Fake','Real'])


# In[31]:


#Multinomial classifier with 
prev_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score=metrics.accuracy_score(y_test,y_pred)
    if score>prev_score:
        classifier=sub_classifier
    print("Alpha: {},Score:{}".format(alpha,score))


# In[ ]:


feature_names=cv.get_feature_names


# In[ ]:


classifier.coef_[0]


# In[50]:


##most real
sorted(zip(classifier.coef_[0],feature_names),reverse=True)[:20]


# In[51]:


# Get the feature names
feature_names = cv.get_feature_names_out()

# Get the log probabilities for each feature
log_probs = classifier.feature_log_prob_[0]

# Combine feature names and log probabilities
feature_log_probs = list(zip(feature_names, log_probs))

# Print the first 20 feature names and their log probabilities
print(feature_log_probs[:20])


# In[49]:


#most  fake
sorted(zip(classifier.coef_[0],feature_names))[:100]


# In[ ]:





# In[ ]:





# In[ ]:




