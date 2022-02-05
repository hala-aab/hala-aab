#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data mining project_Halah Bahanshal


# In[92]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # data visualization
import warnings
warnings.filterwarnings('ignore')  # supress warnings
sns.set_style('whitegrid')
from pickle import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import statsmodels.discrete.discrete_model as sm

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ttest_ind, wilcoxon, shapiro, mannwhitneyu


# In[93]:



# load all data
dt = pd.read_excel('Concrete_Data.xls')


# In[95]:


#Cement (component 1) -- quantitative -- kg in a m3 mixture
#Blast Furnace Slag (component 2) kg in a m3 mixture -- Input Variable 
#Fly Ash (component 3) kg in a m3 mixture -- Input Variable 
#Water (component 4) kg in a m3 mixture -- Input Variable 
#Superplasticizer (component 5) kg in a m3 mixture -- Input Variable 
#Coarse Aggregate (component 6) kg in a m3 mixture -- Input Variable 
#Fine Aggregate (component 7) kg in a m3 mixture -- Input Variable 
#Age -- quantitative -- Day (1~365) -- Input Variable 
#Concrete compressive strength  MPa -- Output Variable 


# In[96]:


dt.info()


# In[97]:


dt.isnull().sum()


# In[98]:


#The dataset consists of 1030 instances with 9 attributes. 
#There are 8 input variables and 1 output variable. 
#Seven input variables represent the amount of raw material (measured in kg/m³) and one represents Age (in Days). 
#The target variable is Concrete Compressive Strength measured in (MPa — Mega Pascal). 


# In[99]:


print("Data set contains {} rows and {} columns".format(dt.shape[0], dt.shape[1]))


# In[100]:


req_col_names = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
                 "CoarseAggregate", "FineAggregare", "Age", "CC_Strength"]
curr_col_names = list(dt.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

dt = dt.rename(columns=mapper)


# In[101]:


dt.head()


# In[102]:


dt.describe(include = 'all')


# In[103]:


plt.rcParams['figure.figsize']=25,10               #Selecting size and width of the plot
dt.hist()                                 #Choosing bar/histogram for visualization
plt.show()


# In[44]:


#No missing value so replace 800 in coarseaggregate with na values
dt['CoarseAggregate']= dt['CoarseAggregate'].replace(800,np.nan)

mean_price = dt['CoarseAggregate'].mean()

dt['CoarseAggregate'].fillna(mean_price, inplace =True)

dt.head()


# In[218]:





# In[104]:


import matplotlib.pyplot as plt

for column in dt:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=dt, x=column)


# In[105]:


corr = dt.corr()

sns.heatmap(corr, annot=True, cmap='Blues')
b, t = plt.ylim()
plt.ylim(b+0.5, t-0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[106]:


dt.head()


# In[107]:


ax = sns.distplot(dt.CC_Strength)
ax.set_title("Compressive Strength Distribution")


# In[108]:


dt.insert(9, "Water to cement Ratio", dt['Water']/dt['Cement'])


# In[109]:


dt.head()


# In[110]:


dt.describe(include = 'all')


# In[111]:


dt.head()


# In[112]:


#Independent variables
X = dt.iloc[:,:-1]
#dependent variable
y = dt.iloc[:,-1] 


# In[113]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[114]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[115]:


# import linear regression models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr=LinearRegression()
fit=lr.fit(X_train,y_train)
score = lr.score(X_test,y_test)
print('predcted score is : {}'.format(score))
print('..................................')
y_predict = lr.predict(X_test)
print('mean_sqrd_error is ==',mean_squared_error(y_test,y_predict))
rms = np.sqrt(mean_squared_error(y_test,y_predict)) 
print('root mean squared error is == {}'.format(rms))


# In[116]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[117]:



print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_predict)),mean_squared_error(y_test, y_predict),
            mean_absolute_error(y_test, y_predict), r2_score(y_test, y_predict)))


# In[142]:


fig, (ax1) = plt.subplots(1, figsize=(8,4))
ax1.scatter(y_predict, y_test, s=20) 
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
ax1.set_ylabel("True") 
ax1.set_xlabel("Predicted") 
ax1.set_title("Linear Regression") 


# In[118]:


#SVM


# In[119]:



dt["Concrete strong or weak"] =np.where(dt["CC_Strength"]> 45.66,1,0)


# In[120]:


dt.head()


# In[121]:


X2=dt.iloc[:,[0,1,2,3,4,5,6,7]]
X3=dt["Cement"], dt["Age"]
y2=dt["Concrete strong or weak"]


# In[122]:


X4=X2.to_numpy()
y4=y2.to_numpy()


# In[123]:


X4


# In[124]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X4, y4, test_size=0.3, random_state=2)


# In[125]:


np.size(X2_test,0)


# In[126]:


np.size(X2_test,1)


# print(y4)

# In[127]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X2_train, y2_train)

#Predict the response for test dataset
y2_pred = clf.predict(X2_test)


# In[128]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y2_test, y2_pred))


# In[ ]:





# In[130]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y2_test, y2_pred)
print(cm)
accuracy_score(y2_test, y2_pred)


# In[84]:



ax.set_title('confusion_matrix');
import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['False','True']); ax.yaxis.set_ticklabels(['True','False']);
plt.show()


# In[85]:


#Logistic Regression


# In[131]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X2_train,y2_train)

#
y2_pred=logreg.predict(X2_test)


# In[132]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y2_test, y2_pred)
cnf_matrix


# In[133]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[89]:


print("Accuracy:",metrics.accuracy_score(y2_test, y2_pred))
print("Precision:",metrics.precision_score(y2_test, y2_pred))
print("Recall:",metrics.recall_score(y2_test, y2_pred))


# In[134]:


y2_pred_proba = logreg.predict_proba(X2_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y2_test,  y2_pred_proba)
auc = metrics.roc_auc_score(y2_test, y2_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:


#Thankyou


# In[ ]:




