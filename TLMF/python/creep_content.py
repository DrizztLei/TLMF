# coding=utf-8
import sys

# from scipy.stats import randint_gen

reload(sys)
sys.setdefaultencoding('utf-8')
import requests, gevent, random, time
import numpy as np
from bs4 import BeautifulSoup
from gevent import monkey
from selenium import webdriver
import os
from tqdm import *

chrome_options = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument('disable-infobars')
chrome_options.add_argument('headless')
driver = webdriver.Chrome(chrome_options=chrome_options)

s = requests.session()
# s.proxies={"https":"127.0.0.1:12333", "http":"127.0.0.1:12333"}
s.keep_alive = False

monkey.patch_all()

href_prefix = "https://www.amazon.com"
search_prefix = "https://www.amazon.com/s?k="
search_suffix = "&i=stripbooks-intl-ship&ref=nb_sb_noss"

dictionary = ['with', 'from', 'to', 'that', 'in', 'one', 'to', 'it',
              'of', 'the', 'and', 'the', 'for', 'in', 'to', 'do',
              'this', 'hwo', 'to', 'in', 'for', 'by', 'an', 'a',
              'by', 'our', 'all', 'R.', 'is', 'out of', ]

FILTER_FILES = "./filters.txt"


class Amazon_books(object):

	def __init__(self):
		self.url = "https://www.amazon.com/gp/bestsellers/books/ref=sv_b_3#1"
		self.tasks_list = []  # 因为有页面请求等待时间,这个列表用来存储任务
		self.index = 1  # 记录图书序号
		# 开始解析页面,当有下一页时,改变self.url,再次调用handle_page,直到无下一页
		# self.handle_page()

	def randHeader(self):
		head_connection = ['Keep-Alive', 'close']
		head_accept = ['text/html, application/xhtml+xml, */*']
		head_accept_language = ['zh-CN,fr-FR;q=0.5', 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3']
		head_user_agent = ['Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
		                   'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36',
		                   'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; rv:11.0) like Gecko)',
		                   'Mozilla/5.0 (Windows; U; Windows NT 5.2) Gecko/2008070208 Firefox/3.0.1',
		                   'Mozilla/5.0 (Windows; U; Windows NT 5.1) Gecko/20070309 Firefox/2.0.0.3',
		                   'Mozilla/5.0 (Windows; U; Windows NT 5.1) Gecko/20070803 Firefox/1.5.0.12',
		                   'Opera/9.27 (Windows NT 5.2; U; zh-cn)',
		                   'Mozilla/5.0 (Macintosh; PPC Mac OS X; U; en) Opera 8.0',
		                   'Opera/8.0 (Macintosh; PPC Mac OS X; U; en)',
		                   'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.12) Gecko/20080219 Firefox/2.0.0.12 Navigator/9.0.0.6',
		                   'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; Win64; x64; Trident/4.0)',
		                   'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; Trident/4.0)',
		                   'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.2; .NET4.0C; .NET4.0E)',
		                   'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Maxthon/4.0.6.2000 Chrome/26.0.1410.43 Safari/537.1 ',
		                   'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.2; .NET4.0C; .NET4.0E; QQBrowser/7.3.9825.400)',
		                   'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0 ',
		                   'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.92 Safari/537.1 LBBROWSER',
		                   'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; BIDUBrowser 2.x)',
		                   'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/3.0 Safari/536.11']

		header = {
			'Connection': head_connection[0],
			'Accept': head_accept[0],
			'Accept-Language': head_accept_language[1],
			'User-Agent': head_user_agent[random.randrange(0, len(head_user_agent))]
		}
		return header

	def get_page(self, url):
		"""输入链接返回resopnse.text"""

		# time.sleep(random.randint(1, 3))
		headers = self.randHeader()
		try:
			# html = requests.get(url, headers=headers).text
			s.headers = headers
			html = s.get(url).text
			# print(type(html))
			return html
		except Exception as e:
			print(e)
			return None

	def introduction_filter(self, introduction):
		temp = ""
		for word in introduction.split():
			if word not in dictionary:
				temp = temp + " "
		return temp

	def handle_page(self):
		"""解析当前页面"""
		# 处理的是self.url 当前页资源获取完毕时,若得到下一页的链接,会改变这个值,再调用该函数

		books = np.loadtxt("./isbn_sorted_uniq.txt", dtype='str', delimiter=',', skiprows=1)

		self.filters_books = np.loadtxt(FILTER_FILES, dtype='str', delimiter='\n')
		self.filters_books = list(self.filters_books)

		self.download_books = os.listdir("./content/")
		self.download_books = np.array(self.download_books)

		for book in tqdm(books):
			book_name = book
			book_name = book_name.replace(":", "")
			book_name = book_name.replace("/", "")
			book_name = book_name.replace("\"", "")

			if book_name in self.filters_books:
				print (book_name + " has been filtered.")
				continue

			if book_name + ".txt" in self.download_books:
				print (book_name + " has been downloaded. ")
				continue


			print ("book origin name: " + book)
			url = search_prefix + book + search_suffix

			time.sleep(random.randint(1, 2))
			# html = self.get_page(url)
			driver.get(url)
			html = driver.page_source

			"""
			if not html:  # 若在取得下一页链接,但是获取页面失败,即停止
				gevent.joinall(self.tasks_list)
			"""
			# 处理当页图书

			soup = BeautifulSoup(html, "lxml")
			li_list = soup.select('a.a-link-normal.a-text-normal')
			length = len(li_list)

			if length is 0:
				print ("search not found: " +  book)
				self.write_filter_file(book_name)
				continue

			li = li_list[0]

			href = li.get('href')
			book_url = href_prefix + href
			# print (str(href))

			path = "./content/" + book_name + ".txt"

			# print (link)

			# print ("book name:" + book_name)
			# print ("path:" + path)

			introduction = self.handle_introduction(book_url, book_name)

			if len(introduction) == 0:
				print (book_name + " introduction is null.")
				continue

			self.write_file(introduction, path)
			print("book " + book_name + " has download.")

	def handle_introduction(self, book_url, book_name):

		# print ("book url:" + book_url)

		driver.get(book_url)

		try:
			driver.switch_to.frame('bookDesc_iframe')
		except:
			# print ("description frame not found! ")
			self.write_filter_file(book_name)
			return ""

		soup = BeautifulSoup(driver.page_source, 'lxml')
		description = soup.select('div#iframeContent')

		introduction = ""

		for tag in description:
			str_info  = tag.get_text()
			# print ("introduction: " + str_info)
			introduction += str_info

		"""
		for tag in info:
			temp = tag.get_text()
			length = len(temp)

			print (length)


			if length > 50:
				introduction += temp
			

		br_info = soup.select('br')
		for tag in info:
			temp = tag.get_text()
			length = len(temp)

			if length > 50:
				introduction += temp


		introduction = self.introduction_filter(introduction)
		print (introduction)

		print ("filter done .")
		
		"""

		return introduction

	def write_file(self, introduction, path):
		f = file(path, 'w+')
		f.write(introduction)
		f.close()
		return

	def write_filter_file(self, book_name):
		f = file(FILTER_FILES, 'a+')
		f.write(book_name + "\n")
		f.close()
		return

	@staticmethod
	def down_img(img_url, path):
		time.sleep(random.randint(2, 3))  # 随机睡眠1,2秒避免访问频率过高
		img = requests.get(img_url).content  # 获取二进制资源
		with open(path, "wb")as f:  # 二进制写入
			f.write(img)


amazon = Amazon_books()
amazon.handle_page()