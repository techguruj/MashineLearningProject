import numpy as np
from sklearn.decomposition import PCA
# from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

tc=pd.read_csv('turnover_clean.csv')

# features and target
target = tc['event']
# dropping target, and categories 'age','stag','gender'
features = tc.drop(['event','age','stag','gender'],axis=1)


for i in range (0,10):
    pca = PCA(n_components=i)
    X_transformed = pca.fit_transform(features)
    print(i,'_components', (pca.explained_variance_ratio_).round(2))

print(i, X_transformed.shape, features.shape)

df=pd.DataFrame(X_transformed)
# df_objects=df.select_dtypes(include=['bool']).copy()
print(df)
