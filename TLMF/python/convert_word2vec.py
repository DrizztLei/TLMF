# -*- coding: UTF-8 -*-

from gensim.models import Word2Vec
import collections

import gensim as gensim
import numpy as np
import numpy as np
import tensorflow as tf
import math
from preprocessing_data import *
from nltk.corpus import stopwords
from nltk import download
from nltk import word_tokenize

stop_words = stopwords.words('english')

n_cluster = 10

MODEL_NAME = "./myprojects/word2vec/word2vec_model"
GOOGLE_NAME = "/home/elvis/Documents/Word2Vec/GoogleNews-vectors-negative300.bin"
DATASET_DIR = "./scattered_content/"

vocabulary = np.loadtxt("./vocabular.txt", dtype=np.str, delimiter='\n')
vocabulary_size = len(vocabulary)

model = gensim.models.Word2Vec.load(MODEL_NAME)
google_model_word_dict = model.wv.vocab

"""
embedding_matrix = np.random.uniform(-0.25, 0.25, [len(vocabulary), 300])

for word, seq in zip(vocabulary, range(len(vocabulary))):
	if word in google_model_word_dict:
		vector = model[word]
		embedding_matrix[seq] = vector
"""


def get_data(dirname):
	if not os.listdir(dirname):
		print "Files not found-Empty Directory "
		return
	else:
		files = os.listdir(dirname)
	filenames = [dirname + "/" + files[i] for i in range(len(files))]
	train_data = [io.open(filenames[i], 'r', encoding='latin-1').read() for i in range(len(filenames))]
	return train_data


def preprocess_gensim(doc):
	""" preprocess raw text by tokenising and removing stop-words,special-charaters """
	doc = doc.lower()  # Lower the text.
	doc = word_tokenize(doc)  # Split into words.
	doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
	doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.

	return doc


def build_dataset(words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)

	dict_size = len(dictionary)
	embedding_matrix = np.random.uniform(-0.25, 0.25, [dict_size, 300])

	for word, seq in zip(dictionary.keys(), range(dict_size)):
		if word in google_model_word_dict:
			vector = model[word]
			embedding_matrix[seq] = vector

	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count,  dictionary, reverse_dictionary, embedding_matrix, dict_size


# words is the all corpus list

train_data = get_data(DATASET_DIR)
w2v_corpus = [preprocess_gensim(train_data[i]) for i in range(len(train_data))]

words = []
for item in w2v_corpus:
	words += item

data, count, dictionary, reverse_dictionary, embedding_matrix, vocabulary_size = build_dataset(words)

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window

	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen=span)

	for _ in range(span):
		# print ("iterat 1")
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	for i in range(batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]

		for j in range(num_skips):
			while target in targets_to_avoid:
				target = np.random.randint(0, span - 1)

			targets_to_avoid.append(target)

			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]

		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	return batch, labels


batch_size = 128
embedding_size = 300
skip_window = 2
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	embeddings = tf.convert_to_tensor(embedding_matrix, dtype=tf.float32)
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
	                                              stddev=1.0 / math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
	                                     biases=nce_biases,
	                                     labels=train_labels,
	                                     inputs=embed,
	                                     num_sampled=num_sampled,
	                                     num_classes=vocabulary_size))




	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_examples)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	init = tf.global_variables_initializer()

num_steps = 1000000

with tf.Session(graph=graph) as sess:
	init.run()
	print("Initialized")

	average_loss = 0
	for step in range(num_steps):
		# print ('step', step)
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		# print (batch_inputs)
		# print (batch_labels)

		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			print("Average loss at step ", step, "loss", average_loss)
			average_loss = 0

		if step % 10000 == 0:
			sim = similarity.eval()
			for i in range(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 5
				nearest = (-sim[i, :]).argsort()[1:top_k + 1]
				# print (nearest)
				log_str = "Nearest to %s:" % valid_word
				for k in range(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = "%s %s" % (log_str, close_word)
				print(log_str)
	final_embeddings = normalized_embeddings.eval()
