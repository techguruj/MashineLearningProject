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
import plotly.express as pex

tc=pd.read_csv('turnover_clean.csv')
df=pd.DataFrame(tc)
# features and target
target = tc['event']
# dropping the target, and categories
# extraversion - which -ive-ly correlated (corr=-0.54) to selfcontrol
# independent - which -ive-ly correlated (corr=-0.54) to anxiety
# novator - which -ive-ly correlated (corr=-0.57) to selfcontrol


df_corr=df.corr().round(1)
fig = pex.imshow(df_corr, color_continuous_scale='rainbow', text_auto = True, labels=dict(color="Correlation"), width=600, height=600)
# plt.imshow(df_corr,cmap='cool')
# fig.show()


features1 = df.drop(['event','extraversion','independ','novator' ],axis=1)
features2=df.drop(['event'],axis=1)
features3=df.drop(['event','age','stag','gender','coach','greywage'  ],axis=1)

X_train, X_test, y_train, y_test = train_test_split(features1, target, test_size=0.2, random_state=42)


# train_set, test_set = train_test_split(tc, test_size=0.2, random_state=42)
X_train.head()

print(f"Train set: {X_train.shape[0]},Test set: {X_test.shape[0]}")

''' additional visualisation'''
tc.plot(kind="scatter", x="stag", y="age", cmap="cool", c="event", grid=True)
# plt.show()

plt.rc('font', size=8)
plt.rc('axes', labelsize=8, titlesize=8)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
tc.hist(bins=40, figsize=(12, 8))
plt.show()
'''Gaussian ------'''
from sklearn.naive_bayes import GaussianNB

GausNB=GaussianNB(var_smoothing=1e-13)
# var_smooth=[1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3]

# fit the model on the training data
data_fit1=GausNB.fit(X_train, y_train)
# data_fit2=GausNB.fit(X_test, y_test)

score1=data_fit1.score(X_train, y_train)
score1_crossval=cross_val_score(data_fit1,X_train,y_train,cv=4).mean()

score2=GausNB.score(X_test, y_test)

'''GridSearch'''
from sklearn.model_selection import GridSearchCV
param_grid={
    'var_smoothing':[1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3]
}
grid_search = GridSearchCV(GausNB,param_grid,cv=4, scoring =['accuracy'],refit='accuracy')

grid_search.fit(X_train,y_train)
best_params= grid_search.best_params_

print('Best params grid search:',best_params)


# score2_crossval=cross_val_score(data_fit2,X_test,y_test,cv=4)
print(f" - Training Score Gauss: {score1},\nCrossval_score Gauss:{score1_crossval}")
print(f" - Test Score Gauss: {score2},\nCrossval_score Gauss:")
train_predict =GausNB.predict(X_train)
#conf matrix

# --------------------------------------Accuracy
conf_matrix=confusion_matrix(y_train, train_predict)
print(conf_matrix)

# plt.figure(figsize=(2, 1))
# sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='cool')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


class_report_train = classification_report(y_train, train_predict)
print("----------------------------------------------"
      "\n1.a.Classification Report (without scaler):")
print(class_report_train)
accuracy_score_train = GausNB.score(X_train, y_train)
print(f"1.a.Accuracy Training Data (without scaler): {accuracy_score_train}")

'''MinMAx Scaler --------- '''
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
X_train_min_max = scaler1.fit_transform(X_train)

GausNB.fit(X_train_min_max, y_train)
train_predict_min_max =GausNB.predict(X_train_min_max)
conf_matrix_minmax=confusion_matrix(y_train, train_predict_min_max)
print(conf_matrix)
class_report_train = classification_report(y_train, train_predict_min_max)
print("----------------------------------------------"
      "\n1.b.Classification Report (Min Max Scaler): ")
print(class_report_train)
accuracy_score_train_minmax = GausNB.score(X_train_min_max, y_train)
print(f"1.b.Accuracy Training Data (Min Max Scaler)\n ---------: {accuracy_score_train_minmax}")

'''Standard Scaler  '''
from sklearn.preprocessing import StandardScaler

scaler2 = StandardScaler()
X_train_std = scaler2.fit_transform(X_train)

GausNB.fit(X_train_std, y_train)
train_predict_standard =GausNB.predict(X_train_std)
conf_matrix_std=confusion_matrix(y_train, train_predict_standard)
print(conf_matrix_std)

class_report_train_std = classification_report(y_train, train_predict_standard)
print("----------------------------------------------"
     "\n1..Classification Report (Standard Scaler):")
print(class_report_train_std)
accuracy_score_train_std = GausNB.score(X_train_std, y_train)
print(f"1.c.Accuracy Training Data (Standard Scaler): {accuracy_score_train_std}")

# print("all CV-Scores",cross_val_score(GaussianNB,X_train,y_train,cv=1))

