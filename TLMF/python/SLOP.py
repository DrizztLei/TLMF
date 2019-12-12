from surprise import *
from surprise.model_selection import *
from surprise import accuracy

TRAIN_FILE = "./train_data"
TEST_FILE = "./test_data"

SPARITY = str(0.3)

TRAIN_FILE = TRAIN_FILE + "_" + SPARITY + ".csv"
TEST_FILE = TEST_FILE + "_" + SPARITY + ".csv"

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 1))

train_data = Dataset.load_from_file(TRAIN_FILE, reader=reader)
train_data = train_data.build_full_trainset()

test_data = Dataset.load_from_file(TEST_FILE, reader=reader)
test_data = test_data.build_full_trainset()
test_data = test_data.build_testset()

algo = SlopeOne()
algo.fit(train_data)
prediction = algo.test(test_data)

accuracy.mae(predictions=prediction)
accuracy.rmse(predictions=prediction)

