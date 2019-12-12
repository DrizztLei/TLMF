from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from matplotlib import pyplot

import gensim as gensim
import numpy as np
from sklearn.cluster import KMeans
from tqdm import *

n_cluster = 100
shape = (100,)

MODEL_NAME = "./myprojects/word2vec/word2vec_model"
# GOOGLE_NAME = "~/Documents/Word2Vec/GoogleNews-vectors-negative300.bin"
# SERVICE_SCORE_FILE = "./servie_score.txt"
DOC_TOPIC_FILE = "./doc2topics_matrix.txt"
DOC_TOPIC_FOLDER = "./service_topic/"
SERVICE_SCORE_FOLDER = "./service_score/"
WORD_TOPIC_FILE = "./word2topic_matrix.txt"

doc_topic_matrix = np.loadtxt(DOC_TOPIC_FILE, dtype=str, delimiter=',')
word_topic_matrix = np.loadtxt(WORD_TOPIC_FILE, dtype=str, delimiter=',')

word_list = word_topic_matrix[::, 0]
word2topic = word_topic_matrix[::, 1::]
word2topic = word2topic.astype(np.float)

# print (doc_topic_matrix.shape)

isbn = doc_topic_matrix[::, 0]
doc2topic = doc_topic_matrix[::, 1::]
doc2topic = doc2topic.astype(np.float)


def load_basic_score(file_name):
    choose = file_name == isbn
    return doc2topic[choose]


def write(filename, content):
    with open(filename, 'w') as f:
        f.write(content)


def load_basic_topic_score(word):
    index = word == word_list
    return word2topic[index]


# print (isbn)
# print(doc2topic.shape)

model = gensim.models.Word2Vec.load(MODEL_NAME)
# model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_NAME, binary=True)

X = model[model.wv.vocab]  # This step could be chagne to collect the specific data set model[word_list]

name = model.wv.vocab
# print(type(name))

map = {}
count = 0

classifier = KMeans(n_clusters=n_cluster, n_jobs=-1, max_iter=1000, algorithm='auto', tol=1e-4)
temp = classifier.fit_transform(X)

label_pred = classifier.labels_
center = classifier.cluster_centers_
inertia = classifier.inertia_

cluster = []
molecule_dis = []
max_dis = []

for index in range(n_cluster):
    cluster.append([])
    max_dis.append([])

for cls in range(n_cluster):
    choose = label_pred == cls
    # print (np.sum(choose))
    vectors = np.array(X[choose])
    cluster[cls] = vectors


def dis(x, y):
    # z = np.sum(x * y) / (np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    # z = cosine_distances(x.reshape(1, -1), y.reshape(1, -1))

    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)

    z = euclidean_distances(x, y)
    return z[0]


def computeR(center, vec):
    return dis(center, vec)
    # return euclidean_distances(center, vec)


for index in range(n_cluster):

    vectors_set = cluster[index]
    vector_max_dis = 0
    sum = 0
    for vector in vectors_set:
        # vector = np.array(vector)
        vec_center = center[index]
        vec_center = vec_center.reshape(1, -1)
        distance = dis(vector, vec_center)
        sum += distance

        if distance > vector_max_dis:
            max_dis[index] = distance

    # sum += dis(vector, center[index])

    molecule_dis.append(sum)


def compute_weight(vector):
    vector = vector.reshape(1, -1)
    index = classifier.predict(vector)
    label = index[0]

    # label = label_pred[index]

    cluster_set = cluster[label]
    m = len(cluster_set)

    # print (molecule_dis[label])
    # print (computeR(center[label], vector))
    # print (m)

    # w = molecule_dis[label] / m - computeR(center[label], vector) / max_dis[label]
    # vector = vector.resape(1, -1)
    vec_center = center[label]
    # w = (m * computeR(vec_center, vector)) / max_dis[label]
    w = 1 - (m * computeR(vec_center, vector) - molecule_dis[label]) / (m * max_dis[label])

    # print (w)
    return w


for file_name in tqdm(isbn):
    out_file = SERVICE_SCORE_FOLDER + file_name
    with open(DOC_TOPIC_FOLDER + file_name, 'r') as f:
        # print (file_name)
        lines = f.readlines()
        addtion_weight = np.ones(shape)

        basic_score = load_basic_score(file_name)

        for line in lines:
            line = line.strip()
            index = (line == word_list)

            vector = model[line]
            weight = compute_weight(vector)
            basic_topic_score = load_basic_topic_score(line)

            # print (basic_topic_score.shape)
            # print (weight)
            # print (basic_topic_score.shape)

            addtion_weight += weight * basic_topic_score.reshape(shape)

        # min_add = np.min(addtion_weight)
        # max_add = np.max(addtion_weight)

        # range_add = max_add - min_add
        # addtion_weight = (addtion_weight - min_add) / range_add

        score = basic_score * addtion_weight

        min_score = np.min(score)
        max_score = np.max(score)
        range_score = max_score - min_score

        sum_score = np.sum(score)

        score = score / sum_score

        # print (score)

        OUT_FILE = SERVICE_SCORE_FOLDER + file_name
        np.savetxt(OUT_FILE, score)

    # print (score)
    # print (addtion_score)
