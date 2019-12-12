# encoding:utf-8
import sys
import numpy as np

reload(sys)

sys.setdefaultencoding('utf-8')

import csv
from tqdm import *

FILE_NAME = "./BX-Book-Ratings.csv"

# np.loadtxt(FILE_NAME, 'str', delimiter="\";\"")

f = open(FILE_NAME, "rb")  # 二进制格式读文件
i = 0

set = []
valid = []
isbn = np.loadtxt("./isbn.txt", dtype='str')

count = 0

while True:
	i += 1
	line = f.readline()

	print (count)

	if not line:
		break
	else:
		# print line
		split_result = line.split(";")
		USER_ID = split_result[0]
		ISBN = split_result[1]
		BOOK_RATING = split_result[2]

		if ISBN not in isbn:
			count += 1

print (count)

"""
valid = np.array(valid)
set = np.array(set)

# print(valid)
print(set)

valid.astype('S1')
set.astype('S1')

print (type(valid))
print (type(set))

print (valid.shape)
print (set.shape)

# result = np.array(valid)
# result = result.reshape(1, -1)
# filter = np.array(set)

# result.astype('S1')

# print (result.dtype)
# print (result.shape)

print(valid[0, ::])

np.savetxt("result.txt", valid, delimiter=',', fmt='%s')
np.savetxt("name_error.txt", set, delimiter=',', fmt='%s')

np.savetxt("isbn.txt", isbn, delimiter=',', fmt='%s')
"""