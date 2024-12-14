# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/bmi.csv

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("drive/MyDrive/bmi.csv")

df.head()

df.dropna()

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

df.columns

df.shape

x=df.drop('Survived',axis=1)
y=df['Survived']

df=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df.columns

df['Age'].isnull().sum()


df['Age'].fillna(method='ffill')

df['Age']=df['Age'].fillna(method='ffill')
df['Age'].isnull().sum()

data=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data

for column in['Sex','Cabin','Embarked']:
   if x[column].dtype=='object':
             x[column]=x[column].astype('category').cat.codes
k=5
selector=SelectKBest(score_func=chi2,k=k)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

x.info()

x=x.drop(["Sex","Cabin","Embarked"],axis=1)
x

from sklearn.feature_selection import SelectKBest, f_regression
selector=SelectKBest(score_func=f_regression,k=5)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector=SelectKBest(score_func=mutual_info_classif,k=5)
X_new=selector.fit_transform(x,y)

selected_features_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)

from sklearn.feature_selection import SelectPercentile,chi2
selector=SelectPercentile(score_func=chi2,percentile=10)
x_new=selector.fit_transform(x,y)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance = model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance > threshold]
print("Selected Features:")
print(selected_features)

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")
df.columns

df

df.isnull().sum()

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

df = pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()

contigency_table=pd.crosstab(tips["sex"],tips["time"])
contigency_table

chi2,p,_,_=chi2_contingency(contigency_table)
print(f"chi-Squared Statistic: {chi2}")
print(f"p-value: {p}")

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

  data={
'Feature1':[1,2,3,4,5],
'Feature2':['A','B','C','A','B'],
'Feature3':[0,1,1,0,1],
'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector = SelectKBest(score_func=f_classif, k=2)
selector.fit(x, y)
selector_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selector_feature_indices]
print("Selected Features:")
print(selected_features)
print("selected_Features:")
print(selected_features) # Assuming selected_features holds the desired value
# RESULT:
       # INCLUDE YOUR RESULT HERE
