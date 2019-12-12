#!/usr/bin/env python2
# -*- coding:utf-8 -*-

from surprise import *
from surprise.model_selection import *
from surprise import accuracy
import numpy as np

np.seterr(divide='ignore')


TRAIN_FILE = "./train_data"
TEST_FILE = "./test_data"

SPARITY = str(0.8)

TRAIN_FILE = TRAIN_FILE + "_" + SPARITY + ".csv"
TEST_FILE = TEST_FILE + "_" + SPARITY + ".csv"

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 1))

train_data = Dataset.load_from_file(TRAIN_FILE, reader=reader)
train_data = train_data.build_full_trainset()

test_data = Dataset.load_from_file(TEST_FILE, reader=reader)
test_data = test_data.build_full_trainset()
test_data = test_data.build_testset()


"""
param_grid = {'n_epochs': [10, 50, 100, 500, 1000], 'n_factors': [15, 100, 500, 1000]}

grid_search = GridSearchCV(NMF, param_grid, measures=['RMSE', 'MAE'], n_jobs=-1, joblib_verbose=True, refit=True)
grid_search.fit(train_data)
grid_search.test(test_data)

# grid_search.evaluate(train_data)
print (grid_search.best_score)
"""



algo = NMF(n_factors=300)
"""
n_factors=10, n_epochs=1000, biased=False, reg_pu=.06,
           reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
           init_low=0.1, init_high=1, random_state=None, verbose=False)
"""
algo.fit(train_data)
prediction = algo.test(test_data)


accuracy.rmse(predictions=prediction)
accuracy.mae(predictions=prediction)
