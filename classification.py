import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data = load_breast_cancer()

X = np.array(data.data)
Y = np.array(data.target)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = GaussianNB()
clf3 = LogisticRegression()
clf4 = DecisionTreeClassifier()
clf5 = RandomForestClassifier()

clf1.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)
clf3.fit(X_train, Y_train)
clf4.fit(X_train, Y_train)
clf5.fit(X_train, Y_train)

print(clf1.score(X_test, Y_test))
print(clf2.score(X_test, Y_test))
print(clf3.score(X_test, Y_test))
print(clf4.score(X_test, Y_test))
print(clf5.score(X_test, Y_test))


