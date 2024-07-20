from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits


digits = load_digits()
data = scale(digits.data)

clf = KMeans(n_clusters=10, init='random', n_init=10)
clf.fit(data)

clf.predict(data)

