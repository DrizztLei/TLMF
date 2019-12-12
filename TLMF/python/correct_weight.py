from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

import gensim as gensim
import numpy as np
from sklearn.cluster import KMeans

n_cluster = 10

MODEL_NAME = "./myprojects/word2vec/word2vec_model"
GOOGLE_NAME = "~/Documents/Word2Vec/GoogleNews-vectors-negative300.bin"

model = gensim.models.Word2Vec.load(MODEL_NAME)
# model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_NAME, binary=True)

X = model[model.wv.vocab]
name = model.wv.vocab
# print(type(name))

map = {}
count = 0

"""
for item in name.items():
	print (item)
	print (type(item))
	print (item[1])
	word = item[0]
	map.update(word=X[count])
	print (map)
	count = count + 1
	exit()
"""

classifier = KMeans(n_clusters=n_cluster, n_jobs=8, max_iter=1000, algorithm='auto')
temp = classifier.fit_transform(X)

label_pred = classifier.labels_
center = classifier.cluster_centers_
inertia = classifier.inertia_

cluster = []
molecule_dis = []

for index in range(n_cluster):
	cluster.append([])

for cls in range(n_cluster):
	choose = label_pred == cls
	# print (np.sum(choose))
	vectors = np.array(X[choose])
	cluster[cls] = vectors


def dis(x, y):
	z = np.sum(x * y) / (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
	return z


def computeR(center, vec):
	return dis(center, vec)


for index in range(n_cluster):

	vectors_set = cluster[index]
	sum = 0
	for vector in vectors_set:
		sum += dis(vector, center[index])

	molecule_dis.append(sum)

for index in range(len(label_pred)):
	label = label_pred[index]
	cluster_set = cluster[label]
	m = len(cluster_set)
	vector = np.ndarray([300])
	w = 1 - (molecule_dis[label]) / (m * computeR(center[label], vectors))

	print (w)
