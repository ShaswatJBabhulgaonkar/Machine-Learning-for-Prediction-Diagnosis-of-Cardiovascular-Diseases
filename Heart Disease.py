#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease

# In[4]:


import pandas
import matplotlib.pyplot as plot
from visuals import disease_stats
import numpy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


dataframe = pandas.read_csv('cleveland.csv')


# In[6]:


dataframe.shape


# In[7]:


dataframe.head(5)


# ### Check for null values

# In[8]:


dataframe.isnull().values.any()


# In[9]:


print("Size before Dropping Rows with Missing Values:", len(dataframe))


# In[10]:


dataframe = dataframe.dropna()


# In[11]:


print("Size after Dropping Rows with Missing Values:", len(dataframe))


# ### Only predict If they will or will not have heart disease

# In[12]:


dataframe.loc[dataframe['num'] != 0, 'num'] = 1


# ### Dataset Features

# In[13]:


figure, axes = plot.subplots(nrows=7, ncols=2, figsize=(20,50) )
plot.suptitle("Heart Disease Data", fontsize=20)

axes[0,0].hist(dataframe.age)
axes[0,0].set_xlabel("Age (years)")
axes[0,0].set_ylabel("Number of Patients")

axes[0,1].hist(dataframe.sex)
axes[0,1].set_xlabel("Sex (0=female,1=male)")
axes[0,1].set_ylabel("Number of Patients")

axes[1,0].hist(dataframe.cp, bins=4, range=(0.5,4.5), rwidth=0.80)
axes[1,0].set_xlim(0.0,5.0)
axes[1,0].set_xlabel("Type of Chest Pain [cp]")
axes[1,0].set_ylabel("Number of Patients")

axes[1,1].hist(dataframe.trestbps)
axes[1,1].set_xlabel("Resting Blood Pressure [trestbps]")
axes[1,1].set_ylabel("Number of Patients")

axes[2,0].hist(dataframe.chol)
axes[2,0].set_xlabel("Serum Cholesterol [chol]")
axes[2,0].set_ylabel("Number of Patients")

axes[2,1].hist(dataframe.fbs)
axes[2,1].set_xlabel("Fasting Blood Sugar [fbs]")
axes[2,1].set_ylabel("Number of Patients")

axes[3,0].hist(dataframe.restecg)
axes[3,0].set_xlabel("Resting Electrocardiography [restecg]")
axes[3,0].set_ylabel("Number of Patients")

axes[3,1].hist(dataframe.thalach)
axes[3,1].set_xlabel("Maximum Heart Rate Achieved [thalach]")
axes[3,1].set_ylabel("Number of Patients")

axes[4,0].hist(dataframe.exang)
axes[4,0].set_xlabel("Exercise Induced Angina [exang]")
axes[4,0].set_ylabel("Number of Patients")

axes[4,1].hist(dataframe.oldpeak)
axes[4,1].set_xlabel("Exercise Induced ST Depression [oldpeak]")
axes[4,1].set_ylabel("Number of Patients")

axes[5,0].hist(dataframe.slope)
axes[5,0].set_xlabel("Slope of Peak Exercise ST Segment [slope]")
axes[5,0].set_ylabel("Number of Patients")

axes[5,1].hist(dataframe.ca,bins=4,range=(-0.5,3.5),rwidth=0.8)
axes[4,1].set_xlim(-0.7,4.7)
axes[5,1].set_xlabel("Major Vessels colored by Fluoroscopy [ca]")
axes[5,1].set_ylabel("Number of Patients")

axes[6,0].hist(dataframe.thal)
axes[6,0].set_xlabel("Thal")
axes[6,0].set_ylabel("Number of Patients")

axes[6,1].hist(dataframe.num,bins=5,range=(-0.5,4.5),rwidth=0.8)
axes[6,1].set_xlabel("Heart Disease [num]")
axes[6,1].set_ylabel("Number of Patients")


# In[14]:


predictions = dataframe.num
features = dataframe.drop('num', axis=1)


# In[15]:


print("Cleveland dataset now has {} data points with {} variables each.".format(*dataframe.shape))


# ### Correlation

# In[16]:


dataframe.corr()


# In[17]:


def plot_correlation(dataframe, size=10):
    corr = dataframe.corr() # Data Frame correlation function
    figure, axes = plot.subplots(figsize=(size, size))
    axes.matshow(corr) # Color code the rectangles by correlation value
    plot.xticks(range(len(corr.columns)), corr.columns) # Draw x tick marks
    plot.yticks(range(len(corr.columns)), corr.columns) # Draw y tick marks


# In[18]:


plot_correlation(dataframe)


# In[16]:


for key in features.columns.values:
    disease_stats(features, predictions, key)


# In[19]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, predictions, test_size=0.30, random_state=42)


# ## Logistic Regression

# In[45]:


import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

best = {"mean_score": 0}
for length in range(9, 14):
    for columns in itertools.combinations(features.columns.values, length):
        model = LogisticRegression(penalty="l1",C=0.8, random_state=37)
        scores = cross_val_score(model, features[list(columns)], predictions, cv=10, scoring='accuracy')
        result = {"columns": columns, "mean_score": scores.mean(), "scores": scores}
        if best["mean_score"] < result["mean_score"]:
            best = result


# In[46]:


print(best)


# In[47]:


columns = list(best['columns'])
model = LogisticRegression(penalty="l1",C=4.2, random_state=37)
accuracies = cross_val_score(model, features[list(columns)], predictions, cv=10, scoring='accuracy')
fscores = cross_val_score(model, features[list(columns)], predictions, cv=10, scoring='f1')


# In[48]:


print("\nBest Results using:")
print(columns)

print(accuracies)
print(accuracies.mean())
print(fscores)
print(fscores.mean())


# ### Tuning Params

# In[49]:


from sklearn.model_selection import GridSearchCV


# In[50]:


c_values = numpy.arange(0.1,10,0.1).tolist()


# In[51]:


param_grid = dict(C=c_values)
print(param_grid)


# In[52]:


lr = LogisticRegression(penalty="l1",C=0.8, random_state=37)
grid = GridSearchCV(lr, param_grid, cv=10, scoring='accuracy', n_jobs=-1)


# In[53]:


columns = ['age', 'sex', 'cp', 'chol', 'thalach', 'exang', 'oldpeak', 'ca', 'thal']

grid.fit(features[columns], predictions)


# In[54]:


grid.cv_results_


# In[55]:


plot.plot(c_values, grid.cv_results_['mean_test_score']*100)
plot.xlabel("C Val")
plot.ylabel("Accuracy")


# In[56]:


grid.best_score_


# In[57]:


grid.best_params_


# In[58]:


grid.best_estimator_


# ## Naive Bayes

# In[59]:


from sklearn.naive_bayes import GaussianNB


# In[60]:


best = {"mean_score": 0}
for length in range(9, 14):
    for columns in itertools.combinations(features.columns.values, length):
        model = GaussianNB()
        scores = cross_val_score(model, features[list(columns)], predictions, cv=10, scoring='accuracy')
        result = {"columns": columns, "mean_score": scores.mean(), "scores": scores}
        if best["mean_score"] < result["mean_score"]:
            best = result


# In[61]:


print(best)


# ## SVM

# In[21]:


from sklearn import svm
import itertools
from sklearn.cross_validation import cross_val_score

best = {"mean_score": 0}
for length in range(9, 14):
    for columns in itertools.combinations(features.columns.values, length):
        model = svm.SVC()
        scores = cross_val_score(model, features[list(columns)], predictions, cv=10, scoring='accuracy')
        result = {"columns": columns, "mean_score": scores.mean(), "scores": scores}
        if best["mean_score"] < result["mean_score"]:
            best = result


# In[22]:


print(best)


# ## Neural Network

# In[43]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.cross_validation import cross_val_score

#best = {"mean_score": 0}
#for length in range(9, 14):
#for columns in itertools.combinations(features.columns.values, length):
columns = ('sex', 'cp', 'trestbps', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
model = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(13,20), random_state=1)
scores = cross_val_score(model, features[list(columns)], predictions, cv=10, scoring='accuracy')
result = {"columns": columns, "mean_score": scores.mean(), "scores": scores}


# In[44]:


print(result)

