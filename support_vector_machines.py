from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)

model = SVC(kernel='linear', C=3)
model.fit(X_train, Y_train)

accuracy = model.score(X_test, Y_test)
print(accuracy)
