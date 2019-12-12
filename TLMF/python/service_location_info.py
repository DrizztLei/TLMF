# encoding:utf-8
import sys
import numpy as np

reload(sys)

sys.setdefaultencoding('utf-8')

import csv
from tqdm import *

# FILE_NAME = "filtered_user.txt"
FILE_NAME = "BX-Users.csv"
OUTFILE_NAME = "filtered_user.txt"

# VALID_ISBN = np.loadtxt(FILE_NAME, 'str', delimiter=',', skiprows=1)
BOOK_RATING = "book_rating_filtered.txt"

# BOOK_RATING_MATRIX = np.loadtxt(BOOK_RATING, 'str', delimiter=',')
# USER_SERVICE = np.loadtxt(FILE_NAME, 'str', delimiter=',')

# print (USER_SERVICE.shape)
# USER = USER_SERVICE[::, 0]
# SERVICE = USER_SERVICE[::, 1]

f = open(FILE_NAME, "rb")  # 二进制格式读文件

lines = f.readlines()

f.close()
filtered_data = []
count_line = 0
valid_count = 0
invalid_count = 0

valid_set = []

for line in tqdm(lines):

    line = line.replace('\n', '')
    split_result = line.split(";")

    USER_ID = split_result[0]
    LOCATION = split_result[1]
    # AGE = split_result[2]

    # print (split_result)
    # print (LOCATION)

    count_line += 1

    try:
        if LOCATION.decode('utf8'):
            if '?' in LOCATION:
                # print (count_line)
                # print LOCATION
                continue
            else:
                LOCATION = LOCATION.decode('utf8')
                valid_set.append([USER_ID, LOCATION])

    except:
        print ("except: " + LOCATION)
        continue

# print ("invalid : " + str(invalid_count))
# print ("valid : " + str(valid_count))

matrix = np.array(valid_set)
np.savetxt(OUTFILE_NAME, valid_set, delimiter=';', fmt='%s')

