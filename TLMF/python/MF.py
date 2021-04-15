#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing

SPARITY = 0.1

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
SERVICE2TOPIC_FILE_NAME = SERVICE2TOPIC_FILE_NAME + "_" + SPARITY + ".npy"
TRAIN_RECORD_FILE = TRAIN_RECORD_FILE + "_" + SPARITY + ".npy"
TEST_RECORD_FILE = TEST_RECORD_FILE + "_" + SPARITY + ".npy"

LAMBDA = 1e-4
ITERATION_SIZE = 1000
num_features = 1000
THRESHOLD = 1e-2
VAL_EPOCHO = 10

matrix = np.load(MATRIX_FILE)
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

record = train_record
test_record = test_record

shape = matrix.shape
x = shape[0]
y = shape[1]


"""
min_max_scaler = preprocessing.MinMaxScaler()

val_scaler = min_max_scaler.fit(train_matrix)
validation_matrix = val_scaler.transform(validation_matrix)
train_matrix = val_scaler.transform(train_matrix)
matrix = val_scaler.transform(matrix)
"""

validation_matrix = validation_matrix / 10.
train_matrix = train_matrix / 10.

# validation_matrix = validation_matrix / 10.
# train_matrix = train_matrix / 10.


# without pre-determine technology
# X_parameters = tf.Variable(tf.random_normal([x, num_features], stddev=1))
# Theta_parameters = tf.Variable(tf.random_normal([y, num_features], stddev=1))

# with pre-determine technology
X_parameters = tf.Variable(user2topic_matrix)
Theta_parameters = tf.Variable(service2topic_matrix)

y_predict = tf.matmul(X_parameters, Theta_parameters, transpose_b=True)
decay = 1e-2
step = 0


train_loss = tf.reduce_sum(((y_predict - train_matrix) * train_record) ** 2) / np.sum(train_record)
val_loss = tf.reduce_sum(((y_predict - validation_matrix) * test_record) ** 2) / np.sum(test_record)

X_l2_regular = tf.nn.l2_loss(X_parameters)
Theta_l2_regular = tf.nn.l2_loss(Theta_parameters)
regular = LAMBDA * (X_l2_regular + Theta_l2_regular)

loss = train_loss + regular

tf_rmse = tf.reduce_sum(((y_predict - matrix) * record) ** 2) / np.sum(record)

# loss = loss + tf_rmse

train = tf.train.AdamOptimizer().minimize(loss)

with tf.variable_scope("LOSS", reuse=tf.AUTO_REUSE):
	tf.summary.scalar('train_loss', train_loss)
	tf.summary.scalar('regular', regular)
	tf.summary.scalar('val_loss', val_loss)

with tf.variable_scope("MSE", reuse=tf.AUTO_REUSE):
	tf.summary.scalar('rmse', tf_rmse)

summaryMerged = tf.summary.merge_all()
filename = './log/adam'
writer = tf.summary.FileWriter(filename)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# print (matrix.shape)
# print(record.shape)
# print(np.sum(record))


def computeS(matrix):
	num = len(matrix.reshape(-1))
	s1 = np.sum(np.abs(matrix))
	s2 = np.sum(matrix ** 2)

	s2 = np.sqrt(s2)
	c = s1 / s2
	a = np.sqrt(num) - c
	b = np.sqrt(num) - 1

	return a / b


for i in range(ITERATION_SIZE):
	_, summary, _loss, tf_rmse_, _train_loss, _regular, pred = sess.run(
		[train, summaryMerged, loss, tf_rmse, train_loss, regular, y_predict])

	if i % VAL_EPOCHO == 0:
		summary, _val_loss = sess.run([summaryMerged, val_loss])
		print("val loss: %f" % _val_loss)

	# _loss = sess.run([loss])
	print("train loss: %f" % _train_loss)
	print("l2 loss: %f" % _regular)
	if _loss < THRESHOLD:
		break

	# print("rmse: %f" % tf_rmse_)
	writer.add_summary(summary, i)

# 把训练的结果summaryMerged存在movie里
# writer.add_summary(movie_summary, i)
# print (_loss.shape)
# print (_loss)
# 把训练的结果保存下来

# 第五步：-------------------------------------评估模型

Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])
# Current_X_parameters为电影内容矩阵，Current_Theta_parameters用户喜好矩阵
# predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean

predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T)


def compute_error(predict, ground_truth):
	errors = np.sqrt(np.sum((predict - ground_truth) ** 2)) / (len(ground_truth.reshape(-1)))

	mae_error = mean_absolute_error(ground_truth, predict)
	rmse_error = mean_squared_error(ground_truth, predict)

	r2_coefficient = r2_score(y_true=ground_truth, y_pred=predict)

	return errors, mae_error, rmse_error, r2_coefficient


errors, mae_error, rmse_error, r2_coefficient = compute_error(predicts[record], matrix[record])

print('-------------------------')

print('train_avg_error', errors)
print('train_mae_error', mae_error)
print('train_rmse_error', rmse_error)
print('train_r2_score', r2_coefficient)
print("after predict sparse:", computeS(predicts))

print('-------------------------')

errors, mae_error, rmse_error, r2_coefficient = compute_error(predicts[test_record], validation_matrix[test_record])

print('test_avg_error', errors)
print('test_mae_error', mae_error)
print('test_rmse_error', rmse_error)
print('test_r2_score', r2_coefficient)
print("after predict sparse:", computeS(predicts[test_record]))

"""
# 第六步：--------------------------------------构建完整的电影推荐系统
user_id = input(u'您要想哪位用户进行推荐？请输入用户编号：')
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
# argsort()函数返回的是数组值从小到大的索引值; argsort()[::-1] 返回的是数组值从大到小的索引值
print(u'recommend result：'.center(80, '='))
# center() 返回一个原字符串居中,并使用空格填充至长度 width 的新字符串。默认填充字符为空格。
idx = 0
for i in sortedResult:
	print(u'value: %.2f, name: %s' % (predicts[i, int(user_id)] - 2, movies_df.iloc[i]['title']))
	idx += 1
	if idx == 20:
		break
"""
