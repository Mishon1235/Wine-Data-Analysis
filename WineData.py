#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics


# In[2]:


from sklearn.datasets import load_wine
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['target_name'] = df['target'].apply(lambda x: data.target_names[x])


# In[3]:


df


# In[4]:


df=df.drop('target_name',axis=1)
pca = PCA(n_components=3)

pca.fit(df)
reduced_data = pca.transform(df)


# In[5]:


df[['alcohol', 'ash']].hist()


# In[51]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c=reduced_data[:, 0]/max(reduced_data[:, 0])
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],c=c, cmap = 'inferno')

plt.figure(figsize=(20, ))
plt.show()


# In[7]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(reduced_data)
clusters = kmeans.predict(reduced_data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='rainbow')
plt.show()


# In[10]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(reduced_data)
clusters = kmeans.predict(reduced_data)
labels = kmeans.labels_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=clusters, cmap='rainbow')


plt.show()


# In[57]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


X_train, X_test, y_train, y_test = train_test_split(reduced_data, labels, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
from sklearn.metrics import rand_score


# pred = kmeans.labels_
# rand_score(df['ash'],pred)

# In[66]:


pred = kmeans.labels_
rand_score(df['flavanoids'],pred)


# In[ ]:





# In[12]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

plot_tree(clf, filled=True, rounded=True)


# In[ ]:





# In[ ]:




