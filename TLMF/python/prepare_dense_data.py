import numpy as np
import os as os

from tqdm import *
# from multiprocessing import Process, Manager
import multiprocessing as mp
import threading
import sys
import gensim as gensim


# from adam import computeS

SPARITY = 0.1
EPSILON = 1e-6

if len(sys.argv) != 1:
	SPARITY = sys.argv[1]

SPARITY = str(SPARITY)

USER_LOCATION_MATRIX = "filtered_user.txt"
MODEL_NAME = "./myprojects/word2vec/word2vec_model"
USER_SERVICE_FILE = "book_rating_filtered.txt"
SCORE_FOLDER = "./service_score/"
SPARSE_FILE_NAME = "./sparse"
ORIGIN_SPARSE_FILE_NAME = "./origin_sparse"
USER_INDEX_NAME = "./user.txt"
SERVICE_INDEX_NAME = "./service.txt"
RATING_MEAN_FILE_NAME = "./mean"
RATING_std_FILE_NAME = "./var.npy"
# TRAINING_FILE = "./train.txt"
USER2TOPIC_FILE_NAME = "./user2topic"
SERVICE2TOPIC_FILE_NAME = "./service2topic"
WORD_TOPIC_FILE = "./word2topic_matrix.txt"
TRAIN_RECORD_FILE = "./train_record"
TEST_RECORD_FILE = "./test_record"

SPARSE_FILE_NAME = SPARSE_FILE_NAME + "_" + SPARITY + ".npy"
ORIGIN_SPARSE_FILE_NAME = ORIGIN_SPARSE_FILE_NAME + "_" + SPARITY + ".npy"
RATING_MEAN_FILE_NAME = RATING_MEAN_FILE_NAME + "_" + SPARITY + ".npy"
RATING_std_FILE_NAME = RATING_std_FILE_NAME + "_" + SPARITY + ".npy"
# TRAINING_FILE = "./train.txt"
USER2TOPIC_FILE_NAME = USER2TOPIC_FILE_NAME + "_" + SPARITY + ".npy"
SERVICE2TOPIC_FILE_NAME =  SERVICE2TOPIC_FILE_NAME + "_" + SPARITY + ".npy"
TRAIN_RECORD_FILE = TRAIN_RECORD_FILE + "_" + SPARITY + ".npy"
TEST_RECORD_FILE = TEST_RECORD_FILE + "_" + SPARITY + ".npy"

MODEL_NAME = "./myprojects/word2vec/word2vec_model"

model = gensim.models.Word2Vec.load(MODEL_NAME)

y = 100

user_localtion_matrix = np.loadtxt(USER_LOCATION_MATRIX, delimiter=';', dtype=str)
user_localtion_matrix = np.char.replace(user_localtion_matrix, "\"", "")
user_localtion_list = user_localtion_matrix[::, 0].astype(np.int)
localtion_list = user_localtion_matrix[::, 1]
user_localtion_list = user_localtion_list.tolist()

word_topic_matrix = np.loadtxt(WORD_TOPIC_FILE, dtype=str, delimiter=',')

word_list = word_topic_matrix[::, 0].tolist()
word2topic = word_topic_matrix[::, 1::]
word2topic = word2topic.astype(np.float)

matrix = np.loadtxt(USER_SERVICE_FILE, delimiter=',', dtype=str)
matrix = np.char.replace(matrix, "\"", "")

user_matrix = matrix[::, 0]
service = matrix[::, 1]
rating = matrix[::, 2].astype(np.float)

service_list = os.listdir(SCORE_FOLDER)

choose = np.zeros(len(service)).astype(np.bool)

zero_index = rating == 0
rating[zero_index] = EPSILON

for doc in tqdm(service_list):
	doc = doc.replace(".txt", "")
	index = doc == service

	# min_value = np.min(rating[index])
	# max_value = np.max(rating[index])

	choose = choose | index

print (np.min(rating), np.max(rating))

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
rating_std = np.zeros([z, 1])

# np.seterr(divide='ignore')


for (doc, seq) in tqdm(zip(service_list, range(z))):
	doc = doc.replace(".txt", "")
	index = doc == service
	rating_result = rating[index]

	rating_mean[seq] = np.mean(rating_result)
	rating_std[seq] = np.std(rating_result)


# global sparse_matrix
# sparse_matrix = np.zeros([x, y])
sparse_matrix = np.random.normal(loc=0, scale=0.05, size=[x, y])

# global train_record
train_record = np.zeros([x, z], dtype=np.bool)

# global sample_matrix
# sample_matrix = np.zeros([x, z])
sample_matrix = np.random.normal(loc=rating_mean.reshape(z), scale=0.05, size=[x, z])

sequence = 0

def computeS(matrix):
	num = len(matrix.reshape(-1))
	s1 = np.sum(matrix)
	s2 = np.sum(matrix ** 2)

	s2 = np.sqrt(s2)
	c = s1 / s2
	a = np.sqrt(num) - c
	b = np.sqrt(num) - 1

	return a / b


def load_score(file_name):
	return np.loadtxt(file_name)

def correct_rat(doc, rat):

	rat_index = service_list.index(doc)

	mean_rat = rating_mean[rat_index]
	std_rat = rating_std[rat_index]

	rat = (rat - mean_rat) / std_rat
	nan_index = np.isnan(rat)
	inf_index = np.isinf(rat)

	all_index = nan_index | inf_index
	rat[all_index] = 0

	# rat = np.nan_to_num(rat)
	# print (rat)
	return rat

def load_basic_topic_score(word):
	if word in word_list:
	    index = word_list.index(word)
	    return word2topic[index]
	else:
		return np.zeros(y)

def load_location_score(user):
	user_index = user_localtion_list.index(user)
	location_list = localtion_list[user_index].split(', ')

	a = location_list[0]
	b = location_list[1]
	c = location_list[2]

	return load_basic_topic_score(a)


def divided_work(train_user, train_service, train_rating, train_user_list, sample_matrix, sparse_matrix, train_record):

	# print ("temp")
	seq_index = np.zeros([len(user_list)])
	seq_index = seq_index.astype(np.bool)

	for user in tqdm(train_user):
		index = user == train_user_list
		# print ("user", train_user_list[index])

		# print (user)
		# print ("index sum:", np.sum(index), index.shape, train_rating.shape)

		services = train_service[index]
		ratings = train_rating[index]
		# print ("rating sum:", len(ratings), ratings)

		sequence = user_list.index(user)
		# print ('seq', sequence)
		seq_index[sequence] = True

		basic = np.zeros(y)
		acc_rat = 0
		acc_score = np.zeros(y)
		record_size = len(ratings)

		# basic = load_location_score(user)
		# print ("record size", record_size)

		for doc, rat in zip(services, ratings):
			service_index = service_list.index(doc)
			sample_matrix[sequence, service_index] = rat

			train_record[sequence, service_index] = True

			doc_name = doc + ".txt"
			full_name = SCORE_FOLDER + doc_name
			score = load_score(full_name)
			acc_score += score
			acc_rat += rat

			# rat = correct_rat(doc, rat)
			# print (score)
			basic += score * rat

		sparse_matrix[sequence] = basic
		# sparse_matrix[sequence] = basic / np.sum(basic)

		# basic = (basic-min_value) / (max_value - min_value)
		# sparse_matrix[sequence]
		# print (acc_rat, acc_score, record_size)
		factor = acc_rat * acc_score / record_size
		# print ("factor:", factor)
		all_size = len(service_list)

		if record_size != 0:
			for doc, col in zip(service_list, range(all_size)):
				# print(train_record[sequence, col])
				if train_record[sequence, col] == False:
					doc_name = doc + ".txt"
					full_name = SCORE_FOLDER + doc_name
					score = load_score(full_name)
					transform_rat = np.sum(score * factor)

					# transform_rat += np.random.normal(loc=0, scale=0.01)
					# print("transfor rat:%f" % transform_rat)
					sample_matrix[sequence, col] = transform_rat
					# print (transform_rat)
		else:
			# print ("run in else")
			for doc, col in zip(service_list, range(all_size)):
				print(train_record[sequence, col])
				transform_rat = np.random.normal(loc=rating_mean[col], scale=0.05)
				# transform_rat = 0
				sample_matrix[sequence, col] = transform_rat
		# print ('iter done')
	#print ("done")
	return sample_matrix, sparse_matrix, train_record, seq_index
	# print (np.sum(train_record))

def callback_func(result):
	sample, sparse, record, seq_index = result
	sample_matrix[record] = sample[record]
	# sample_matrix[0, 0] = 3123
	# sparse_matrix[record] = sparse[record]
	sparse_matrix[seq_index] = sparse[seq_index]
	global train_record
	train_record = train_record | record

def para_test(matrix, col):
	# global kk

	for x in tqdm(range(1000000)):
		temp = np.random.random([4])
		col = np.random.randint(4)
		matrix[col] = temp
	# matrix[0, 0] = col


def build_sparse_matrix(matrix, sparsity):

	number = len(rating)
	sparsity = np.float(sparsity)

	sample_number = np.int(number * sparsity)

	permutation = np.random.permutation(number)
	permutation_index = permutation < sample_number

	train_user_list = user_matrix[permutation_index]
	train_user_set = np.unique(train_user_list)
	train_service = service[permutation_index]
	train_rating = rating[permutation_index]
	# print (np.min(train_rating))

	"""
	global rating_mean
	global rating_std

	for (doc, seq) in zip(service_list, range(z)):
		doc = doc.replace(".txt", "")
		index = doc == train_service
		rating_result = train_rating[index]

		rating_mean[seq] = np.mean(rating_result)
		rating_std[seq] = np.std(rating_result)

	rating_mean = np.nan_to_num(rating_mean)
	rating_std = np.nan_to_num(rating_std)
	"""

	user_length = len(train_user_set)

	works = np.float(20)
	interval = np.int(np.ceil(user_length / works))
	task = []
	start = 0
	end = interval

	for sub in range(np.int(works)):

		sub_task = train_user_set[start:end]
		start += interval
		end += interval

		task.append(sub_task)

		if end >= user_length:
			end = user_length - 1

	process = []

	global kk
	kk = np.zeros([4, 4])
	works = np.int(works)
	pool = mp.Pool(processes=works)

	for sub_task in task:
		# p = threading.Thread(target=divided_work, args=(sub_task, train_service, train_rating, train_user_list, sample_matrix, sparse_matrix, train_record, ))
		# p = threading.Thread(target=para_test, args=(kk, 2,))
		# p = threading.Thread(target=para_test, args=(np.random.rand(), ))
		pool.apply_async(func=divided_work, args=(sub_task, train_service, train_rating, train_user_list, sample_matrix, sparse_matrix, train_record, ), callback=callback_func)

		# process.append(p)
		# p.start()

	pool.close()
	pool.join()

	# print (np.sum(train_record))
	return sample_matrix, sparse_matrix, train_record


def build_service2topic_matrix():

	service2topic_matrix = np.zeros([len(service_list), y])
	for service, sequence in tqdm(zip(service_list, range(len(service_list)))):
		doc_name = service  + ".txt"
		full_name = SCORE_FOLDER + doc_name
		score = load_score(full_name)
		service2topic_matrix[sequence] = score

	return service2topic_matrix

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
service2topic_matrix = build_service2topic_matrix()

test_record = data_record ^ train_record

save_np_matrix(origin_matrix, ORIGIN_SPARSE_FILE_NAME)
save_np_matrix(sparse_matrix, SPARSE_FILE_NAME)
save_np_matrix(user2topic_matrix, USER2TOPIC_FILE_NAME)
save_np_matrix(service2topic_matrix, SERVICE2TOPIC_FILE_NAME)

save_np_matrix(train_record, TRAIN_RECORD_FILE)
save_np_matrix(test_record, TEST_RECORD_FILE)

save_user_index(user_set, USER_INDEX_NAME)
save_user_index(service_list, SERVICE_INDEX_NAME)

save_np_matrix(rating_mean, RATING_MEAN_FILE_NAME)
save_np_matrix(rating_std, RATING_std_FILE_NAME)


"""
origin_matrix = build_origin_matrix()

save_user_index(user_set, USER_INDEX_NAME)
save_user_index(service_list, SERVICE_INDEX_NAME)

train_matrix = build_sparse_matrix(sparse_matrix, SPARITY)
save_sparse_matrix(train_matrix, TRAINING_FILE)
"""
