#!/usr/bin/env python2
# -*- coding:utf-8 -*-

import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing

SPARITY = 0.6

if len(sys.argv) != 1:
	SPARITY = sys.argv[1]

SPARITY = str(SPARITY)

SPARSE_FILE_NAME = "./sparse"
ORIGIN_SPARSE_FILE_NAME = "./origin_sparse"
USER_INDEX_NAME = "./user.txt"
SERVICE_INDEX_NAME = "./service.txt"
RATING_MEAN_FILE_NAME = "./mean"
RATING_VAR_FILE_NAME = "./var.npy"
# TRAINING_FILE = "./train.txt"
USER2TOPIC_FILE_NAME = "./user2topic"
SERVICE2TOPIC_FILE_NAME = "./service2topic"
TRAIN_RECORD_FILE = "./train_record"
TEST_RECORD_FILE = "./test_record"

SPARSE_FILE_NAME = SPARSE_FILE_NAME + "_" + SPARITY + ".npy"
TRAIN_FILE = SPARSE_FILE_NAME
ORIGIN_SPARSE_FILE_NAME = ORIGIN_SPARSE_FILE_NAME + "_" + SPARITY + ".npy"
MATRIX_FILE = ORIGIN_SPARSE_FILE_NAME
RATING_MEAN_FILE_NAME = RATING_MEAN_FILE_NAME + "_" + SPARITY + ".npy"
RATING_VAR_FILE_NAME = RATING_VAR_FILE_NAME + "_" + SPARITY + ".npy"
# TRAINING_FILE = "./train.txt"
USER2TOPIC_FILE_NAME = USER2TOPIC_FILE_NAME + "_" + SPARITY + ".npy"
SERVICE2TOPIC_FILE_NAME =  SERVICE2TOPIC_FILE_NAME + "_" + SPARITY + ".npy"
TRAIN_RECORD_FILE = TRAIN_RECORD_FILE + "_" + SPARITY + ".npy"
TEST_RECORD_FILE = TEST_RECORD_FILE + "_" + SPARITY + ".npy"

LAMBDA = 1e-5
ITERATION_SIZE = 4000
num_features = 10
THRESHOLD = 1e-6
VAL_EPOCHO = 100
DECAY_RATE = np.float32(1e-2)
DECAY_STEP = 0

# matrix = np.load(MATRIX_FILE)
validation_matrix = np.load(MATRIX_FILE)
train_matrix = np.load(TRAIN_FILE)
service = np.loadtxt(SERVICE_INDEX_NAME, dtype=np.str)
rating_mean = np.load(RATING_MEAN_FILE_NAME)
rating_var = np.load(RATING_VAR_FILE_NAME)
user2topic_matrix = np.load(USER2TOPIC_FILE_NAME)
service2topic_matrix = np.load(SERVICE2TOPIC_FILE_NAME)

service2topic_matrix = service2topic_matrix.astype(np.float32)
user2topic_matrix = user2topic_matrix.astype(np.float32)

train_record = np.load(TRAIN_RECORD_FILE)
test_record = np.load(TEST_RECORD_FILE)

# record = train_record

record_size = np.sum(train_record)
all_size = train_record.shape[0] * train_record.shape[1]
rest_size = all_size - record_size

# train_record = train_record.astype(np.float)
record = train_record
train_record = tf.ones(train_record.shape)

shape = train_matrix.shape
x = shape[0]
y = shape[1]

# min_max_scaler = preprocessing.MinMaxScaler()
# val_scaler = min_max_scaler.fit(train_matrix)
# validation_matrix = val_scaler.transform(validation_matrix)
# train_matrix = val_scaler.transform(train_matrix)

# matrix = val_scaler.transform(matrix)
validation_matrix = validation_matrix / 10.
train_matrix = train_matrix / 10.
# matrix = matrix / 10.

# 构建模型

# X_parameters = tf.Variable(tf.random_normal([x, num_features], stddev=0.1))
X_parameters = tf.Variable(user2topic_matrix)
# Theta_parameters = tf.Variable(tf.random_normal([y, num_features], stddev=0.1))
Theta_parameters = tf.Variable(service2topic_matrix)

global_step = tf.Variable(0)
add_op = tf.add(global_step, 1)
update_op = tf.assign(global_step, add_op)

y_predict = tf.matmul(X_parameters, Theta_parameters, transpose_b=True)

# train_loss = tf.reduce_sum(((y_predict - train_matrix) * record) ** 2) / (np.sum(record))
train_loss = tf.reduce_sum(((y_predict - train_matrix) * train_record) ** 2) / (np.sum(record))

record_loss = tf.sqrt(tf.reduce_sum(((y_predict - train_matrix) * record) ** 2) / (np.sum(record)))
train_avg_loss = tf.reduce_sum(((y_predict - train_matrix) * train_record) ** 2) / tf.reduce_sum(train_record)
val_loss = tf.reduce_sum(((y_predict - validation_matrix) * test_record) ** 2) / np.sum(test_record)

X_l2_regular = tf.nn.l2_loss(X_parameters)
Theta_l2_regular = tf.nn.l2_loss(Theta_parameters)
l_regular = X_l2_regular + Theta_l2_regular
avg_l_regular = tf.reduce_mean(l_regular)
regular = LAMBDA * l_regular
loss = train_loss + regular

tf_rmse = tf.sqrt(tf.reduce_sum(((y_predict - validation_matrix) * test_record) ** 2) / np.sum(test_record))

# loss = loss + tf_rmse

# train = tf.train.AdamOptimizer(learning_rate=5e-3, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)
train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

with tf.variable_scope("LOSS", reuse=tf.AUTO_REUSE):
	tf.summary.scalar('train_loss', train_loss)
	tf.summary.scalar('regular', regular)
	tf.summary.scalar('val_loss', val_loss)
	tf.summary.scalar('train_normal_rmse', record_loss)
	tf.summary.scalar('train_avg_loss', train_avg_loss)

with tf.variable_scope("MSE", reuse=tf.AUTO_REUSE):
	tf.summary.scalar('rmse', tf_rmse)


summaryMerged = tf.summary.merge_all()
filename = './log/adam'
writer = tf.summary.FileWriter(filename)


# print (matrix.shape)


def computeS(matrix):
	num = len(matrix.reshape(-1))
	s1 = np.sum(np.abs(matrix))
	s2 = np.sum(matrix ** 2)

	s2 = np.sqrt(s2)
	c = s1 / s2
	a = np.sqrt(num) - c
	b = np.sqrt(num) - 1

	return a / b


print("train sparity: %f" % computeS(train_matrix))
print("eval sparity: %f" % computeS(validation_matrix))
# print("processed sparity: %f" % computeS(matrix))


ratio = tf.exp(-DECAY_RATE * tf.cast(global_step, tf.float32))
non_record_weight = tf.zeros(record.shape) + ratio
train_record = tf.where(record, train_record, non_record_weight)

"""
record_ratio = (all_size - rest_size * ratio) / record_size
record_weight = tf.zeros(record.shape) + record_ratio
train_record = tf.where(~record, train_record, record_weight)
"""

check = tf.reduce_sum(train_record)
max_value = tf.reduce_max(train_record)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

pre_loss = 0
diff_loss_threshold = 1e-7

for i in range(ITERATION_SIZE):
	_, summary, _loss, tf_rmse_, _train_loss, _regular = sess.run([train, summaryMerged, loss, tf_rmse, train_loss, l_regular])
	step = sess.run(update_op)


	# sys.stdout.write("train loss: %f, l2 loss: %f \r" % (float(_train_loss), float(_regular)))
	# sys.stdout.flush()


	# if i % VAL_EPOCHO == 0:
	if True:
		summary, _val_loss, _avg_l_regular = sess.run([summaryMerged, val_loss, avg_l_regular])
		# print("val loss: %f" % _val_loss)
		sys.stdout.write("train loss: %f, l2 loss: %f, val loss: %f \r" % (float(_train_loss), float(_avg_l_regular), float(_val_loss)))
		sys.stdout.flush()
	else:
		sys.stdout.write("train loss: %f, l2 loss: %f, val loss: %f \r" % (float(_train_loss), float(_regular), float(_val_loss)))
		# sys.stdout.write("train loss: %f, l2 loss: %f, \r" % (float(_train_loss), float(_regular)))
		sys.stdout.flush()

	# _loss = sess.run([loss])


	if step > DECAY_STEP:
		check_value = sess.run(check)
		# decay_value = sess.run(ratio)
		# print ("decay ration: %f" % decay_value)

	"""
		diff = np.abs(check_value - all_size)
		print ("diff: %f" % diff)
		print("value: %f" % sess.run(max_value))

	"""


	delta_loss = np.abs(_loss - pre_loss)
	if delta_loss < diff_loss_threshold:
		break

	pre_loss = _loss

	# print("l2 loss: %f\r" % _regular)
	if _loss < THRESHOLD:
		break

	# print("rmse: %f" % tf_rmse_)
	writer.add_summary(summary, i)
	sess.run(update_op)

# writer.add_summary(movie_summary, i)
# print (_loss.shape)
# print (_loss)

Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])
# predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean

predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T)

def compute_error(predict, ground_truth):

	errors = np.sqrt(np.sum(((predict - ground_truth)) ** 2)) / (len(ground_truth.reshape(-1)))

	mae_error = mean_absolute_error(ground_truth, predict)
	mse_error = mean_squared_error(ground_truth, predict)
	rmse_error = np.sqrt(mse_error)

	r2_coefficient = r2_score(ground_truth, predict)

	return errors, mae_error, rmse_error, r2_coefficient

errors, mae_error, rmse_error, r2_coefficient = compute_error(predicts[record], train_matrix[record])

print ('-------------------------')

print ('train_avg_error', errors)
print ('train_mae_error', mae_error)
print ('train_mse_error', rmse_error)
print ('train_r2_score', r2_coefficient)
print ("after predict sparse:", computeS(predicts))

print ('-------------------------')

errors, mae_error, rmse_error, r2_coefficient = compute_error(predicts[test_record], validation_matrix[test_record])

print ('test_avg_error', errors)
print ('test_mae_error', mae_error)
print ('test_mse_error', rmse_error)
print ('test_r2_score', r2_coefficient)
print ("after predict sparse:", computeS(predicts[test_record]))
