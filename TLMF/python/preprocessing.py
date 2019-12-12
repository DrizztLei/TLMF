# encoding:utf-8
import sys
import numpy as np

reload(sys)

sys.setdefaultencoding('utf-8')

import csv
from tqdm import *

FILE_NAME = "./BX-Books.csv"

"""
csv_file = csv.reader(open(FILE_NAME))

for line in tqdm(csv_file):
	print (type(line))
	print (len(line))
	print (line)
"""

# np.loadtxt(FILE_NAME, 'str', delimiter="\";\"")

f = open(FILE_NAME, "rb")  # 二进制格式读文件
i = 0

set = []
valid = []
isbn = []

while True:
	i += 1
	line = f.readline()

	if not line:
		break
	else:
		# print line
		split_result = line.split(";")
		ISBN = split_result[0]
		TITLE = split_result[1]
		# print(ISBN)
		# print(TITLE)
		AUTHOR = split_result[2]
		YEAR = split_result[3]
		PUBLISH = split_result[4]
		IMG_URL_S = split_result[5]
		IMG_URL_M = split_result[6]
		IMG_URL_L = split_result[7]

		isbn.append(ISBN)

		try:
			if TITLE.decode('utf8'):
				TITLE = TITLE.decode('utf8')
				valid.append([TITLE, ISBN])
		except:
			# TITLE = TITLE.decode('')
			set.append(ISBN)

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
	if not line:
		break
	else:
		try:

			# print(line)
			# print(line.decode('utf8'))

			line.decode('utf8')
			valid.append(lind.decode('utf8'))
		# 为了暴露出错误，最好此处不print
		except:
			set.append(str(line))
			print(str(line))
"""

"""
for line in set:
	print(line.split("\";\"")[0].replace("\"", ''))
"""