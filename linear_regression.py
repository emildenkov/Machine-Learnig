import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('resources/student-mat.csv', sep=';')
data = data[['age', 'sex', 'studytime', 'absences', 'G1', 'G2', 'G3']]

data['sex'] = data['sex'].map({'F': 0, 'M': 1})
prediction = 'G3'

X = np.array(data.drop([prediction], axis=1))
Y = np.array(data[prediction])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

model = LinearRegression()
model.fit(X_train, Y_train)

X_new = np.array([[18, 1, 3, 40, 15, 16]])
Y_new = model.predict(X_new)

plt.scatter(data['G2'], data['G3'])
plt.title('Correlations')
plt.xlabel('Second Grade')
plt.ylabel('Final Grade')
plt.show()