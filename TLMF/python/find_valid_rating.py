# encoding:utf-8
import sys
import numpy as np
from tqdm import *
reload(sys)

sys.setdefaultencoding('utf-8')

import csv
from tqdm import *

PREFIX = "temp/xa"
END_SET = ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
VALID_ISBN_FILE = "./isbn_sorted_uniq.txt"
OUT_FILE_NAME = "./book_rating_filtered.txt"

VALID_ISBN = np.loadtxt(VALID_ISBN_FILE, 'str')


for end in END_SET:
	HISTORY_FILE = PREFIX + end
	print ("processing the:" + HISTORY_FILE)

	f = open(HISTORY_FILE, "rb")  # 二进制格式读文件

	lines = f.readlines()

	f.close()
	filtered_data = []

	for line in tqdm(lines):

		line = line.replace('\n', '')
		split_result = line.split(";")

		USER_ID = split_result[0]
		ISBN = split_result[1]
		BOOK_RATING = split_result[2]

		try:
			if ISBN.decode('utf8'):
				ISBN = ISBN.decode('utf8')

			if ISBN in VALID_ISBN:
				temp_set = USER_ID + "," + ISBN + "," + BOOK_RATING + "\n"
				filtered_data.append(temp_set)
				# print (temp_set)
		except:
			continue

	with open(OUT_FILE_NAME, 'a+') as f:
		for line in tqdm(filtered_data):
			f.write(line)