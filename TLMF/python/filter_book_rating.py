# encoding:utf-8
import sys
import numpy as np

reload(sys)

sys.setdefaultencoding('utf-8')

import csv
from tqdm import *

FILE_NAME = "./BX-Book-Ratings.csv"
OUT_FILE_NAME = "./book_rating.txt"

# np.loadtxt(FILE_NAME, 'str', delimiter="\";\"")

isbn = []
f = open(FILE_NAME, "rb")  # 二进制格式读文件

while True:
	line = f.readline()

	if not line:
		break
	else:
		split_result = line.split(";")
		USER_ID = split_result[0]
		ISBN = split_result[1]
		BOOK_RATING = split_result[2]
		# isbn.append(ISBN)

		try:
			if ISBN.decode('utf8'):
				ISBN = ISBN.decode('utf8')
				isbn.append(ISBN)
		except:
			continue

		# print line

# print (isbn)
matrix = np.array(isbn)
matrix.astype('S1')
matrix = matrix.reshape(-1, 1)
print (matrix.shape)

np.savetxt(OUT_FILE_NAME, matrix, delimiter=',', fmt='%s')