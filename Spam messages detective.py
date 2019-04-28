#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[4]:


#nltk.download_shell()


# In[5]:


messages = [ line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[6]:


print(len(messages))


# In[7]:


import pandas as pd


# In[14]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label','message'])


# In[15]:


messages.head()


# In[17]:


messages.describe()


# In[18]:


messages.groupby('label').describe()


# In[19]:


messages["length"] = messages["message"].apply(len)


# In[20]:


messages.head()


# In[22]:


from nltk.corpus import stopwords


# In[26]:


stopwords.words('english')


# In[27]:


import string 


# In[38]:


mess = 'Sample message! I notice: punctuation"'


# In[39]:


nopunc = [c for c in mess if c not in string.punctuation]


# In[30]:


nopunc


# In[40]:


nopunc = ''.join(nopunc)


# In[32]:


nopunc


# In[41]:


nopunc.split()


# In[42]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[43]:


clean_mess


# In[44]:


def text_process(mess):
    """
        1. remove punc
        2. remove stop words
        3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
     


# In[49]:


messages['message'].head().apply(text_process)


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer


# In[51]:


bow_transformer  = CountVectorizer(analyzer = text_process).fit(messages['message'])


# In[52]:


mess4 = messages['message'][3]


# In[53]:


print(mess4)


# In[54]:


bow4 = bow_transformer.transform([mess4])


# In[55]:


print(bow4)


# In[56]:


bow_transformer.get_feature_names()[4068]


# In[57]:


message_bow = bow_transformer.transform(messages['message'])


# In[61]:


print(' Shape of Sparse Matrix : ', message_bow.shape)


# In[64]:


message_bow.nnz


# In[65]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[66]:


tfidf_transformer = TfidfTransformer().fit(message_bow)


# In[67]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[69]:


print(tfidf4)


# In[70]:


message_tfidf = tfidf_transformer.transform(message_bow)


# In[71]:


from sklearn.naive_bayes import MultinomialNB


# In[72]:


spam_detect_model = MultinomialNB().fit(message_tfidf, messages['label'])


# In[74]:


spam_detect_model.predict(tfidf4)[0]


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages["label"])


# In[77]:


from sklearn.pipeline import Pipeline


# In[78]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer= text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[79]:


pipeline.fit(msg_train, label_train)


# In[80]:


predictions =pipeline.predict(msg_test)


# In[82]:


predictions


# In[84]:


label_test


# In[87]:


from sklearn.metrics import classification_report, confusion_matrix


# In[86]:


print(classification_report(label_test, predictions))


# In[88]:


print(confusion_matrix(label_test, predictions))


# In[ ]:




