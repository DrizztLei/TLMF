import numpy as np
import os as os
from tqdm import *
# from adam import computeS

USER_SERVICE_FILE = "book_rating_filtered.txt"
SCORE_FOLDER = "./service_score/"
SPARSE_FILE_NAME = "./sparse.npy"
ORIGIN_SPARSE_FILE_NAME = "./origin_sparse.npy"
USER_INDEX_NAME = "./user.txt"
SERVICE_INDEX_NAME = "./service.txt"
RATING_MEAN_FILE_NAME = "./mean.npy"
RATING_VAR_FILE_NAME = "./var.npy"
# TRAINING_FILE = "./train.txt"
USER2TOPIC_FILE_NAME = "./user2topic.npy"
TRAIN_RECORD_FILE = "./train_record.npy"
TEST_RECORD_FILE = "./test_record.npy"

SPARITY = 0.8

y = 10

matrix = np.loadtxt(USER_SERVICE_FILE, delimiter=',', dtype=str)
matrix = np.char.replace(matrix, "\"", "")

user_matrix = matrix[::, 0]
service = matrix[::, 1]
rating = matrix[::, 2].astype(np.int)

service_list = os.listdir(SCORE_FOLDER)

choose = np.zeros(len(service)).astype(np.bool)

for doc in service_list:
	doc = doc.replace(".txt", "")
	index = doc == service
	choose = choose | index

user_matrix = user_matrix[choose]
service = service[choose]
rating = rating[choose]

user_set = set(user_matrix)
service_set = set(service)
service_list = list(service_set)
user_list = list(user_set)


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


def build_sparse_matrix(matrix, sparsity):

	number = len(rating)

	sample_number = np.int(number * sparsity)

	permutation = np.random.permutation(number)
	permutation_index = permutation < sample_number

	train_user_list = user_matrix[permutation_index]
	train_user_set = np.unique(train_user_list)
	train_service = service[permutation_index]
	train_rating = rating[permutation_index]

	train_record = np.zeros(matrix.shape, dtype=np.bool)
	sample_matrix = np.zeros(matrix.shape)

	for user in tqdm(train_user_set):
		index = user == train_user_list
		services = train_service[index]
		ratings = train_rating[index]

		sequence = user_list.index(user)

		basic = np.zeros(y)

		for doc, rat in zip(services, ratings):
			service_index = service_list.index(doc)
			sample_matrix[sequence, service_index] = rat
			train_record[sequence, service_index] = True

			doc_name = doc + ".txt"
			full_name = SCORE_FOLDER + doc_name
			score = load_score(full_name)
			rat = correct_rat(doc, rat)
			# print (score)
			basic += score * rat

		# basic = (basic-min_value) / (max_value - min_value)
		# sparse_matrix[sequence]

		sparse_matrix[sequence] = basic
		# print (basic)

	return sample_matrix, sparse_matrix, train_record

def build_origin_matrix():
	origin_sparse_matrix = np.zeros([x, z])
	all_record = np.zeros([x, z], dtype=np.bool)

	for user in tqdm(user_set):
		index = user == user_matrix
		services = service[index]
		ratings = rating[index]

		sequence = user_list.index(user)

		for doc, rat in zip(services, ratings):
			service_index = service_list.index(doc)
			origin_sparse_matrix[sequence, service_index] = rat
			all_record[sequence, service_index] = True

	return origin_sparse_matrix, all_record

def save_np_matrix(sparse_matrix, file_name):
	# np.savetxt(file_name, sparse_matrix)
	np.save(file_name, sparse_matrix)


def save_user_index(user_set, file_name):
	user_set = np.array(list(user_set)).reshape(-1, 1)
	np.savetxt(file_name, user_set, fmt='%s')

origin_matrix, data_record = build_origin_matrix()
sparse_matrix, user2topic_matrix, train_record = build_sparse_matrix(origin_matrix, SPARITY)

test_record = data_record ^ train_record

save_np_matrix(origin_matrix, ORIGIN_SPARSE_FILE_NAME)
save_np_matrix(sparse_matrix, SPARSE_FILE_NAME)
save_np_matrix(user2topic_matrix, USER2TOPIC_FILE_NAME)

save_np_matrix(train_record, TRAIN_RECORD_FILE)
save_np_matrix(test_record, TEST_RECORD_FILE)

save_user_index(user_set, USER_INDEX_NAME)
save_user_index(service_list, SERVICE_INDEX_NAME)

save_np_matrix(rating_mean, RATING_MEAN_FILE_NAME)
save_np_matrix(rating_var, RATING_VAR_FILE_NAME)


"""
origin_matrix = build_origin_matrix()

save_user_index(user_set, USER_INDEX_NAME)
save_user_index(service_list, SERVICE_INDEX_NAME)

train_matrix = build_sparse_matrix(sparse_matrix, SPARITY)
save_sparse_matrix(train_matrix, TRAINING_FILE)
"""