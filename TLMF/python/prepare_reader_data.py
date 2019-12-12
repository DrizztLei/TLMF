import numpy as np
import os as os
from tqdm import *
# from adam import computeS
import sys


SPARITY = 0.8

if len(sys.argv) != 1:
	SPARITY = sys.argv[1]

SPARITY = str(SPARITY)

USER_SERVICE_FILE = "book_rating_filtered.txt"
SCORE_FOLDER = "./service_score/"
USER_INDEX_NAME = "./user.txt"
SERVICE_INDEX_NAME = "./service.txt"
TRAIN_FILE = "./train_data"
TEST_FILE = "./test_data"

TRAIN_FILE = TRAIN_FILE + "_" + SPARITY + ".csv"
TEST_FILE = TEST_FILE + "_" + SPARITY + ".csv"

SPARITY = np.float(SPARITY)

matrix = np.loadtxt(USER_SERVICE_FILE, delimiter=',', dtype=str)
matrix = np.char.replace(matrix, "\"", "")

user_matrix = matrix[::, 0]
service = matrix[::, 1]
rating = matrix[::, 2].astype(np.int)

service_list = os.listdir(SCORE_FOLDER)

choose = np.zeros(len(service)).astype(np.bool)

for doc in tqdm(service_list):
	doc = doc.replace(".txt", "")
	index = doc == service
	choose = choose | index

user_matrix = user_matrix[choose]
service = service[choose]
rating = rating[choose]

ZERO_THRESHOLD = 1e-6
zero_index = rating == 0
rating = rating.astype(np.float)
rating[zero_index] = ZERO_THRESHOLD

rating = rating / 10.

def save_np_matrix(sparse_matrix, file_name):
	# np.savetxt(file_name, sparse_matrix)
	np.save(file_name, sparse_matrix)


def save_user_index(user_set, file_name):
	user_set = np.array(list(user_set)).reshape(-1, 1)
	np.savetxt(file_name, user_set, fmt='%s')


def build_data(sparsity):

	number = len(rating)

	sample_number = np.int(number * sparsity)

	permutation = np.random.permutation(number)
	permutation_index = permutation < sample_number
	train_index = permutation_index
	test_index = ~train_index

	train_user_matrix = user_matrix[train_index]
	train_service = service[train_index]
	train_rating = rating[train_index]

	test_user_matrix = user_matrix[test_index]
	test_service = service[test_index]
	test_rating = rating[test_index]

	train = np.stack([train_user_matrix, train_service, train_rating]).T
	test = np.stack([test_user_matrix, test_service, test_rating]).T

	np.savetxt(TRAIN_FILE, train, fmt='%s', delimiter=',')
	np.savetxt(TEST_FILE, test, fmt='%s', delimiter=',')

build_data(SPARITY)