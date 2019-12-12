# encoding:utf-8
import sys
import numpy as np
from tqdm import *
import os

top_n = 3

# the next is run the java code

reload(sys)

sys.setdefaultencoding('utf-8')

DIR = "./scatter_content/"

import csv
from tqdm import *

SERVICE_LOCATION_FREQUENCY_FILE = "service_location_filtered.txt"
SERVICE_DIR = "./content/"

f = open(SERVICE_LOCATION_FREQUENCY_FILE, 'a+')
lines = f.readlines()
f.close()

download_books = os.listdir(SERVICE_DIR)
for index in range(len(download_books)):
	download_books[index] = download_books[index].replace('.txt', '')

for line in tqdm(lines):
	info = str(line).split(';')
	isbn = info[0]
	isbn = isbn.replace('\"', '')

	if isbn not in download_books:
		continue

	location_list = info[1::]

	text = ""

	for index in range(0, len(location_list)):
		city = location_list[index]
		city = city.replace('\"', '')
		city = city.replace('n/a', '')
		city = city.replace(', ', ',')
		# print (city)
		# city = city.replace('\n', '')

		location_list[index] = city
		text = text + "," + city

	text = text[1:]
	text = text.strip()
	# text = text.replace(',', '')

	# text = text.replace()

	text = text.split(',')
	text = filter(lambda x: x != "", text)

	# print (text)

	counts = {}
	for i in text:
		counts[i] = counts.get(i, 0) + 1

	iteams = list(counts.items())
	iteams.sort(key=lambda x: x[1], reverse=True)

	size = len(text)

	FILE_NAME = DIR + isbn + ".txt"

	info = ""

	for iteam in iteams:
		# print (iteam)
		# print (iteam[0])
		frequency = iteam[1] / float(size)
		# print (frequency)

		info = info + iteam[0] + ";" + str(frequency) + "\n"

	# print (info)

	f = open(FILE_NAME, 'w+')
	f.write(info)
	f.close()
