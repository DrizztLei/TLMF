from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
import gensim as gensim
import numpy as np
from sklearn.cluster import KMeans

MODEL_NAME = "./word2vec_model"

WORD_TOPIC_FILE = "../../word2topic_matrix.txt"

word_topic_matrix = np.loadtxt(WORD_TOPIC_FILE, dtype=str, delimiter=',')

word_list = word_topic_matrix[::, 0]
word_list = word_list.tolist()

model = gensim.models.Word2Vec.load(MODEL_NAME)

# X = model[model.wv.vocab]
X = model[word_list]
SAMPLE_NUMBER = len(X)
neighbor = 8
n_cluster = SAMPLE_NUMBER / neighbor

# vis = PCA(n_components=2, whiten=False)
vis = TSNE(perplexity=neighbor, n_components=2, init='pca', n_iter=1000, learning_rate=10)
result = vis.fit_transform(X)


# print (pca.explained_variance_)
# print (pca.explained_variance_ratio_)

choice = np.random.choice(len(result), int(len(result)), replace=False)
# result = result[choice, ::]

# pyplot.scatter(result[choice, 0], result[choice, 1])
words = list(model.wv.vocab)

# print (len(words))


for index in choice:
    word = words[index]
    x = result[index, 0]
    y = result[index, 1]

    # pyplot.annotate(word, xy=(x, y))


X = result[choice, ::]
y_pred = KMeans(n_clusters=n_cluster, n_jobs=-1, max_iter=1000, algorithm='auto').fit_transform(X)

# pyplot.scatter(X[:, 0], X[:, 1], marker='o')
# pyplot.scatter(X[::, 0], X[::, 1], c=np.squeeze(y_pred))
# pyplot.scatter(X[::, 0], X[::, 1], marker='o')

fig, ax = pyplot.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xticks([])
ax.set_yticks([])

x = X[::, 0]
y = X[::, 1]
s = 10
c = np.argmin(y_pred, axis=1)
pyplot.scatter(x, y, s=s, c=c, linewidths=5)


for index in range(len(X)):
    x = X[index, 0]
    y = X[index, 1]
    text = word_list[index]
    # print (text)
    # pyplot.scatter(x, y, s=s, c=c)
    # pyplot.scatter(x, y, s=s, c='r', linewidths=5)
    pyplot.annotate(s=text, xy=(x, y), xytext=(x, y))


# pyplot.scatter(X[::, 0], X[::, 1], s=1, c=np.argmin(y_pred, axis=1))

pyplot.savefig("out.pdf", dpi=300, format='pdf')
pyplot.show()
