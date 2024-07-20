import pickle
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()

X = data.data
Y = data.target

best_accuracy = 0

for x in range(2500):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    model = SVC(kernel='linear', C=3)
    model.fit(X_train, Y_train)

    accuracy = model.score(X_test, Y_test)

    if accuracy > best_accuracy:
        best_accuracy = accuracy

        print(f'Best accuracy is {best_accuracy}')

        with open('model.pickle', 'wb') as file:
            pickle.dump(model, file)

