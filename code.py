from numpy import genfromtxt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

fn = r'C:\Users\DELL I5558\Desktop\Python\electricity_price_and_demand_20170926.csv'
my_data = genfromtxt(fn, delimiter=',')
model = KMeans(n_clusters=5)
model.fit(my_data)
labels = model.predict(my_data)
print(labels)
model = TSNE(learning_rate=100)
transformed = model.fit_transform(my_data)
xs = transformed[:, 0]
ys = transformed[:, 1]

plt.scatter(xs, ys, c=labels)
plt.show()
