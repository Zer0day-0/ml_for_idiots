#import necessary libaries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#loading dataset
from sklearn.datasets import load_breast_cancer

#naming dataset
cancer = load_breast_cancer()

#head
print(cancer.keys())

#loading dataframe
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df.head())

#rescaling(std. normalization)
from sklearn.preprocessing import StandardScaler

#refitting df
scaler = StandardScaler()
scaler.fit(df)

#scaled data
scaler = StandardScaler()
scaler.fit(df)

#we find PCs using the fit method, then apply the rotation and dimensionality reduction by calling transform
scaled_data = scaler.transform(df)


#imprting PCA
from sklearn.decomposition import PCA

#pca of two componets
pca = PCA(n_components=2)

#fitting PCA
pca.fit(scaled_data)


#transforming intp two PCs
x_pca = pca.transform(scaled_data)


print(x_pca.shape)
print(scaled_data.shape)

#printing image data
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()

#showing data
