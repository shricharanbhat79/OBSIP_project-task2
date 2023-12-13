import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics  import confusion_matrix ,accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


#------Data Preprocessing----
data=pd.read_csv('Iris.csv',index_col=0,na_values=["??","????"])

data.head(10)
data.info
data.describe()
data.isnull().sum()
data.shape


data2=data.copy()
data2

data2['Species']=pd.factorize(data2['Species'])[0]
data2

#------Visualization------
sns.distplot(data2['Species'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
plt.xlabel("count")
plt.ylabel("Density")
plt.title("Density of price")
plt.legend("Price") # Providing legend labels as a list
plt.show()

data2.plot(kind='box')
plt.xticks(rotation=30)

#-------Data Splitting and Scaling-----

y = data2.Species
data2.drop(['Species'], axis=1, inplace=True)
x = data2
data2.shape

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#-------Dimensionality Reduction with PCA---------
pca = PCA(random_state = 36)
pca.fit(X_train)
pca.components_
pca.explained_variance_ratio_


var_cumu = np.cumsum(pca.explained_variance_ratio_)
fig = plt.figure(figsize=[12,8])
plt.vlines(x=2, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.95, xmax=4, xmin=0, colors="b", linestyles="--")
plt.plot(var_cumu)
plt.ylabel("Cumulative variance explained")
plt.show()

pca1 = PCA(n_components=2, random_state= 36)
pca2 = PCA(n_components=1, random_state= 36)
X_train_pca1 = pca1.fit_transform(X_train_scaled)
X_test_pca1 = pca1.fit_transform(X_test_scaled)
X_train_pca2 = pca2.fit_transform(X_train_scaled)
X_test_pca2 = pca2.fit_transform(X_test_scaled)

def evaluateModel(y_test, y_pred):
    print("Confusion Matrix :- \n {}".format(confusion_matrix(y_test, y_pred)))
    print("Accuracy : {}".format(accuracy_score(y_test, y_pred)))
    print("\n")
    
    
#------Model Training and Evaluation -------
logreg = LogisticRegression()
logreg.fit(X_train_pca1,y_train)

y_train_pred = logreg.predict(X_train_pca1)
print("Train Score (PCA 2 comp, Logistic Regression): ")
evaluateModel(y_train,y_train_pred)

y_test_pred = logreg.predict(X_test_pca1)
print("Test Score (PCA 2 comp, Logistic Regression): ")
evaluateModel(y_test,y_test_pred)



logreg.fit(X_train_pca2,y_train)

y_train_pred = logreg.predict(X_train_pca2)
print("Train Score (PCA single comp, Logistic Regression): ")
evaluateModel(y_train,y_train_pred)

y_test_pred = logreg.predict(X_test_pca2)
print("Test Score (PCA single comp, Logistic Regression): ")
evaluateModel(y_test,y_test_pred)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_pca1, y_train)

y_train_pred = dt.predict(X_train_pca1)
print("Train Score (PCA 2 comp, Decision Tree): ")
evaluateModel(y_train,y_train_pred)

y_test_pred = dt.predict(X_test_pca1)
print("Test Score (PCA 2 comp, Decision Tree): ")
evaluateModel(y_test,y_test_pred)

dt.fit(X_train_pca2, y_train)

y_train_pred = dt.predict(X_train_pca2)
print("Train Score (PCA single comp, Decision Tree): ")
evaluateModel(y_train,y_train_pred)

y_test_pred = dt.predict(X_test_pca2)
print("Test Score (PCA single comp, Decision Tree): ")
evaluateModel(y_test,y_test_pred)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_pca1, y_train)

y_train_pred = rf.predict(X_train_pca1)
print("Train Score (PCA 2 comp, Random Forest): ")
evaluateModel(y_train,y_train_pred)

y_test_pred = rf.predict(X_test_pca1)
print("Test Score (PCA 2 comp, Random Forest): ")
evaluateModel(y_test,y_test_pred)


