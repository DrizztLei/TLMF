from surprise import *
from surprise.model_selection import *
from surprise import accuracy

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

algo = SVD()

"""
n_factors=10, n_epochs=100, biased=True, init_mean=0,
           init_std_dev=.1, lr_all=.005,
           reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
           reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
           random_state=None, verbose=False)
"""

algo.fit(train_data)
prediction = algo.test(test_data)

accuracy.rmse(predictions=prediction)
accuracy.mae(predictions=prediction)
