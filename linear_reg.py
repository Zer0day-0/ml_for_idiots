#ml practice linear regression
#saurav Roy
#!/bin/bash

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


#loading datasets
df = pd.read_csv('/home/saurav/Desktop/MLworks/BostonHousing.csv')

#dependent and independent features are kept in different datasets
Y=df.medv #medianvalue
X=df.drop("medv", axis=1) #dependeant features

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#standerdizing the dataset
scalar = StandardScaler()

#fitting and transfroming
X_train = scalar.fit_transform(X_train)

"""it_transform() is used on the training data to learn the scaling or transformation parameters and 
then applies the same transformation to the training data.transform() is used on new data (e.g. test data) 
to apply the same transformation that was learned on the training data."""

#transforming test data
X_test = scalar.transform(X_test)

#calling regression
regression=LinearRegression()
regression.fit(X,y)

#linear regression cross validation 5 times
mse = cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)

#predictiong the values
reg_pred = regression.predict(X_test)

#mean of mse 
mean_mse = np.mean(mse)

#plotting difference
plt.hist(reg_pred-y_test)

print(df.columns)

print("libraries loded")



