# encoding:utf-8
import sys
import numpy as np

reload(sys)

sys.setdefaultencoding('utf-8')

import csv
from tqdm import *

SERVICE_LOCATION_FREQUENCY_FILE = "service_location_filtered.txt"
USER_LOCATION_FILE = "filtered_user.txt"
USER_SERVICE_FILE = "book_rating_filtered.txt"

USER_LOCATION_MATRIX = np.loadtxt(USER_LOCATION_FILE, dtype='str', delimiter=';')
USER_SERVICE_MATRIX = np.loadtxt(USER_SERVICE_FILE, dtype='str', delimiter=',')

# print (USER_LOCATION_MATRIX)
# print (USER_SERVICE_MATRIX)

SERVICE_LOCATION_MATRIX = ""
SERVICE_LIST = USER_SERVICE_MATRIX[::, 1]
LOCATION_LIST = USER_LOCATION_MATRIX[::, 1]

# print (SERVICE_LIST[0:5])
# print (len(SERVICE_SET))
# print (LOCATION_LIST[0:5])
# print (len(LOCATION_LIST))

x = len(set(SERVICE_LIST))
y = len(set(LOCATION_LIST))

# print (x)
# print (y)
f = open(SERVICE_LOCATION_FREQUENCY_FILE, 'a+')

for service in tqdm(set(SERVICE_LIST)):
	# print (service)
	index = (service == USER_SERVICE_MATRIX[::, 1])
	invoked_users = USER_SERVICE_MATRIX[index][::, 0]

	location_list = []

	for user in invoked_users:
		invoded_city = (user == USER_LOCATION_MATRIX[::, 0])
		location_list.append(USER_LOCATION_MATRIX[invoded_city, 1])

	info = service
	location_info = ""
	# print (len(location_list))

	for location in location_list:
		if len(location) == 0:
			continue
		location_info += ";" + location[0]

	if len(location_info) == 0:
		continue

	info = info + location_info + "\n"

	f.write(info)

f.close()
