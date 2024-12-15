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

![Screenshot 2024-12-14 204226](https://github.com/user-attachments/assets/e84178b8-444a-43c3-9749-b16e3e21c527)

df.dropna()

max_vals=np.max(np.abs(df[['Height','Weight']]))

max_vals

![Screenshot 2024-12-14 204242](https://github.com/user-attachments/assets/e6ec0243-107b-4e25-84ae-63cdd147c560)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])

df.head(10)

![Screenshot 2024-12-14 204251](https://github.com/user-attachments/assets/efbfd9b1-e636-45d8-ac50-8d8fd371bf02)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head(10)

![Screenshot 2024-12-14 204301](https://github.com/user-attachments/assets/1ef9c001-46ff-4502-8f30-ce454fd62b18)

from sklearn.preprocessing import Normalizer

scaler=Normalizer()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df

![Screenshot 2024-12-14 204312](https://github.com/user-attachments/assets/27123cf6-04c9-4100-ad8a-cd3f858f3f2d)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler

scaler=MaxAbsScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df

![Screenshot 2024-12-14 204322](https://github.com/user-attachments/assets/8d35111a-45c6-4363-8817-9c99b1542ba5)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])

df.head()

![Screenshot 2024-12-14 204332](https://github.com/user-attachments/assets/c500dda0-e947-45dd-8294-7c4c29035bcd)

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

![Screenshot 2024-12-14 204346](https://github.com/user-attachments/assets/65cb2f8b-4043-41ed-9560-e81855277acc)

df.shape

![Screenshot 2024-12-14 204355](https://github.com/user-attachments/assets/3d58dc36-6467-45fe-b406-69f1797b29ce)

x=df.drop('Survived',axis=1)

y=df['Survived']

df=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df.columns

![Screenshot 2024-12-14 204414](https://github.com/user-attachments/assets/eaf5ce50-b622-4b8c-9e9d-570315100b30)

df['Age'].isnull().sum()

![Screenshot 2024-12-14 204422](https://github.com/user-attachments/assets/0865129b-14e8-419e-b6f3-42fcd31f505e)


df['Age'].fillna(method='ffill')

![Screenshot 2024-12-14 204448](https://github.com/user-attachments/assets/7a46d7b3-26cb-4cb7-8e11-0c2bdf6c852b)

df['Age']=df['Age'].fillna(method='ffill')

df['Age'].isnull().sum()

![Screenshot 2024-12-14 204458](https://github.com/user-attachments/assets/a518a8b5-eca6-4577-ba3f-0677c5b07547)

data=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)

y=data['Survived']

x

![Screenshot 2024-12-14 204509](https://github.com/user-attachments/assets/93bf1a9c-d69a-4605-80b3-4ac7ac084d5e)

data["Sex"]=data["Sex"].astype("category")

data["Cabin"]=data["Cabin"].astype("category")

data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes

data["Cabin"]=data["Cabin"].cat.codes

data["Embarked"]=data["Embarked"].cat.codes

data

![Screenshot 2024-12-14 204526](https://github.com/user-attachments/assets/51a3da3b-7bda-4e64-b306-9ae29b508d56)

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

![Screenshot 2024-12-14 204539](https://github.com/user-attachments/assets/3faa2d58-dbd4-4962-91d4-b166ce43f08b)

x.info()

![Screenshot 2024-12-14 204551](https://github.com/user-attachments/assets/5d3ef382-2b55-47d6-9a69-f9c96f3f972d)

x=x.drop(["Sex","Cabin","Embarked"],axis=1)

x

![Screenshot 2024-12-14 204559](https://github.com/user-attachments/assets/d2d73289-87e5-41be-9e35-a58a11cf210a)

from sklearn.feature_selection import SelectKBest, f_regression

selector=SelectKBest(score_func=f_regression,k=5)

X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]

print("Selected Features:")

print(selected_features)

![Screenshot 2024-12-14 204613](https://github.com/user-attachments/assets/5c8eb9ab-8cf6-4faa-b8ea-9fede56b02a6)

from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector=SelectKBest(score_func=mutual_info_classif,k=5)

X_new=selector.fit_transform(x,y)

selected_features_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_features_indices]

print("Selected Features:")

print(selected_features)

![Screenshot 2024-12-14 204622](https://github.com/user-attachments/assets/46eb50ce-fa16-4782-9738-8220189e92cf)

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

![Screenshot 2024-12-14 204630](https://github.com/user-attachments/assets/6af747c5-279d-45cd-8003-74980c47896e)

model = RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x,y)

feature_importance = model.feature_importances_

threshold=0.15

selected_features=x.columns[feature_importance > threshold]

print("Selected Features:")

print(selected_features)

![Screenshot 2024-12-14 204637](https://github.com/user-attachments/assets/acabc3fa-9c47-4347-882e-4fd6c759a194)

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

df.columns

![Screenshot 2024-12-14 204645](https://github.com/user-attachments/assets/a8d0eefa-aa5d-41aa-a46a-300e75b94ef6)

df

![Screenshot 2024-12-14 204705](https://github.com/user-attachments/assets/f8f40920-d518-42ee-8efb-17528bf02eb6)

df.isnull().sum()

![Screenshot 2024-12-14 204720](https://github.com/user-attachments/assets/6015930a-8181-4ac7-95ca-b9f2d22b1aa9)

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.impute import SimpleImputer

df = pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

from scipy.stats import chi2_contingency

import seaborn as sns

tips=sns.load_dataset("tips")

tips.head()


![Screenshot 2024-12-14 204730](https://github.com/user-attachments/assets/8ec6db2f-78d3-4d20-84d1-30d2a88c9f76)

contigency_table=pd.crosstab(tips["sex"],tips["time"])

contigency_table

![Screenshot 2024-12-14 204740](https://github.com/user-attachments/assets/eb2eeb4c-ade4-4dc4-838a-21ab354949fb)

chi2,p,_,_=chi2_contingency(contigency_table)

print(f"chi-Squared Statistic: {chi2}")

print(f"p-value: {p}")

![Screenshot 2024-12-14 204747](https://github.com/user-attachments/assets/716fa77b-8dba-4102-986a-5acf78d4e164)

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

![Screenshot 2024-12-14 204759](https://github.com/user-attachments/assets/c938654f-eb76-4df7-bde6-164a89a4d515)

# RESULT:
   The code is run successfully.
