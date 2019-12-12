import numpy as np
import os as os
from tqdm import *
# from adam import computeS

USER_SERVICE_FILE = "book_rating_filtered.txt"
SCORE_FOLDER = "./service_score/"
SPARSE_FILE_NAME = "./sparse.txt"
ORIGIN_SPARSE_FILE_NAME = "./origin_sparse.txt"
USER_INDEX_NAME = "./user.txt"
SERVICE_INDEX_NAME = "./service.txt"
RATING_MEAN_FILE_NAME = "./mean.txt"
RATING_VAR_FILE_NAME = "./var.txt"
# TRAINING_FILE = "./train.txt"
USER2TOPIC_FILE_NAME = "./user2topic.txt"

SPARITY = 0.05

y = 10

matrix = np.loadtxt(USER_SERVICE_FILE, delimiter=',', dtype=str)
matrix = np.char.replace(matrix, "\"", "")

user_list = matrix[::, 0]
service = matrix[::, 1]
rating = matrix[::, 2].astype(np.int)

service_list = os.listdir(SCORE_FOLDER)

choose = np.zeros(len(service)).astype(np.bool)

for doc in service_list:
	doc = doc.replace(".txt", "")
	index = doc == service
	choose = choose | index

user_list = user_list[choose]
service = service[choose]
rating = rating[choose]

user_set = set(user_list)
service_set = set(service)
service_list = list(service_set)

x = len(user_set)
z = len(service_list)

rating_mean = np.zeros([z, 1])
rating_var = np.zeros([z, 1])

np.seterr(divide='ignore')

for (doc, seq) in zip(service_list, range(z)):
	doc = doc.replace(".txt", "")
	index = doc == service
	rating_result = rating[index]

	rating_mean[seq] = np.mean(rating_result)
	rating_var[seq] = np.var(rating_result)

sparse_matrix = np.zeros([x, y])
sequence = 0

def load_score(file_name):
	return np.loadtxt(file_name)

def correct_rat(doc, rat):

	rat_index = service_list.index(doc)

	mean_rat = rating_mean[rat_index]
	var_rat = rating_var[rat_index]

	rat = (rat - mean_rat) / var_rat
	nan_index = np.isnan(rat)
	inf_index = np.isinf(rat)

	all_index = nan_index | inf_index
	rat[all_index] = 0

	# rat = np.nan_to_num(rat)
	# print (rat)
	return rat

"""
for user in tqdm(user_set):
	index = user == user_list
	services = service[index]
	ratings = rating[index]

	basic = np.zeros(y)

	for doc, rat in zip(services, ratings):
		doc_name = doc + ".txt"
		full_name = SCORE_FOLDER + doc_name
		score = load_score(full_name)
		rat = correct_rat(doc, rat)
		# print (score)
		basic += score * rat

	min_value = np.min(basic)
	max_value = np.max(basic)

	# basic = (basic-min_value) / (max_value - min_value)
	# sparse_matrix[sequence]

	sparse_matrix[sequence] = basic
	# print (basic)

	sequence += 1

"""

def build_sparse_matrix(matrix, sparsity):

	number = matrix.shape[0] * matrix.shape[1]

	sample_number = np.int(number * sparsity)

	permutation = np.random.permutation(number)
	permutation = permutation.reshape(matrix.shape)
	permutation_index = permutation < sample_number

	sample_matrix = np.zeros(matrix.shape)
	sample_matrix[permutation_index] = matrix[permutation_index]

	for (data, row) in tqdm(zip(permutation_index, range(permutation_index.shape[0]))):

		basic = np.zeros(y)

		for (value, col) in zip(data, range(len(data))):
			if value:
				rat = sample_matrix[row, col]
				doc = service_list[col]
				doc_name = doc + ".txt"
				full_name = SCORE_FOLDER + doc_name
				score = load_score(full_name)
				rat = correct_rat(doc, rat)
				basic += score * rat
				# print (doc_name)
		# print (basic)
		sparse_matrix[row] = basic

	return sample_matrix, sparse_matrix

def build_origin_matrix():
	origin_sparse_matrix = np.zeros([x, z])

	count = 0

	for user in tqdm(user_set):
		index = user == user_list
		services = service[index]
		ratings = rating[index]

		for doc, rat in zip(services, ratings):
			service_index = service_list.index(doc)
			origin_sparse_matrix[count, service_index] = rat

	count += 1

	return origin_sparse_matrix

def save_sparse_matrix(sparse_matrix, file_name):
	np.savetxt(file_name, sparse_matrix)

def save_user_index(user_set, file_name):
	user_set = np.array(list(user_set)).reshape(-1, 1)
	np.savetxt(file_name, user_set, fmt='%s')

origin_matrix = build_origin_matrix()
sparse_matrix, user2topic_matrix = build_sparse_matrix(origin_matrix, SPARITY)

save_sparse_matrix(origin_matrix, ORIGIN_SPARSE_FILE_NAME)
save_sparse_matrix(sparse_matrix, SPARSE_FILE_NAME)
save_sparse_matrix(user2topic_matrix, USER2TOPIC_FILE_NAME)

save_user_index(user_set, USER_INDEX_NAME)
save_user_index(service_list, SERVICE_INDEX_NAME)

save_sparse_matrix(rating_mean, RATING_MEAN_FILE_NAME)
save_sparse_matrix(rating_var, RATING_VAR_FILE_NAME)


"""
origin_matrix = build_origin_matrix()

save_user_index(user_set, USER_INDEX_NAME)
save_user_index(service_list, SERVICE_INDEX_NAME)

train_matrix = build_sparse_matrix(sparse_matrix, SPARITY)
save_sparse_matrix(train_matrix, TRAINING_FILE)
"""