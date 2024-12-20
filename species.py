import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\vanda\OneDrive\Documents\iris.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
dataset = data.copy()
X = dataset.drop(['species'], axis = 1)
print(X.head())
y = dataset['species']
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier()
rfmodel.fit(X_train, y_train)
pred = rfmodel.predict(X_test)

print(X_test)
print(y_test,pred)
print('random forest')
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))
rf=accuracy_score(y_test,pred)
from sklearn.svm import SVC
svmmodel = SVC()
svmmodel.fit(X_train, y_train)
pred = svmmodel.predict(X_test)

print(X_test)
print(y_test,pred)
print('svm alogritham')
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))
svc=accuracy_score(y_test,pred)
from sklearn.tree import DecisionTreeClassifier
dtmodel= DecisionTreeClassifier()

dtmodel.fit(X_train, y_train)
pred = dtmodel.predict(X_test)

print(X_test)
print(y_test,pred)
print('decesion tree alogritham')
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)
print("{:.0%}".format(accuracy_score(y_test,pred)))
dt=accuracy_score(y_test,pred)
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=3)

knnmodel.fit(X_train, y_train)
pred = knnmodel.predict(X_test)

print(X_test)
print(y_test,pred)
print('Knn alogritham')
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)
print("{:.0%}".format(accuracy_score(y_test,pred)))
knn=accuracy_score(y_test,pred)
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
nbmodel = GaussianNB()
nbmodel.fit(X_train, y_train)
pred = nbmodel.predict(X_test)

print(X_test)
print(y_test,pred)
print(' navie bayes alogritham')
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)
print("{:.0%}".format(accuracy_score(y_test,pred)))
nb=accuracy_score(y_test,pred)
score = [rf*100,svc*100,dt*100,knn*100,nb*100]
print(score)
algo = ['RF','SVC','DT','KNN','NB']  
plt.bar(algo,score,color = 'green')  
	  
plt.title("accuracy graph")  
plt.ylabel('accuries')  
plt.xlabel(' score')  
plt.show()  

test=(5.1,3.5,1.4,0.2)
input_as_numpy=np.asarray(test)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=rfmodel.predict(input_reshaped)
print(pre1)
