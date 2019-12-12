#coding=utf-8
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import requests, gevent, random, time
from bs4 import BeautifulSoup
from gevent import monkey

monkey.patch_all()

href_prefix = "https://www.amazon.com"

dictionary = ['with', 'from', 'to', 'that', 'in', 'one', 'to', 'it',
              'of', 'the', 'and', 'the', 'for', 'in', 'to', 'do',
              'this', 'hwo', 'to', 'in', 'for', 'by', 'an', 'a',
              'by', 'our', 'all', 'R.', 'is', 'out of', ]


class Amazon_books(object):
	def __init__(self):
		self.url = "https://www.amazon.com/gp/bestsellers/books/ref=sv_b_3#1"
		self.tasks_list = []  # 因为有页面请求等待时间,这个列表用来存储任务
		self.index = 1  # 记录图书序号
		# 开始解析页面,当有下一页时,改变self.url,再次调用handle_page,直到无下一页
		self.handle_page()

	def get_page(self, url):
		"""输入链接返回resopnse.text
		不设防网页,简单构造请求头即可"""
		headers = {
			"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36"}
		try:
			html = requests.get(url, headers=headers).text
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
		html = self.get_page(self.url)
		if not html:  # 若在取得下一页链接,但是获取页面失败,即停止
			gevent.joinall(self.tasks_list)
		# 处理当页图书
		soup = BeautifulSoup(html, "lxml")
		li_list = soup.select('li[class="zg-item-immersion"]')
		print ('init')
		for li in li_list:
			info = li.select('img')
			link_list = li.select('.a-link-normal')
			link = link_list[0]
			link = link.get('href')

			book = li.select('img')[0].attrs['alt']  # 书名
			book = book.replace(":", "")  # 书名去冒号,否则会路径错误
			book = book.replace("/", "")  # 书名去冒号,否则会路径错误

			path = "./content/" + book + ".txt"
			# print (link)
			print (book)
			print (path)

			introduction = self.handle_introduction(href_prefix + link)
			print (href_prefix + link)
			print ('------------')
			print (introduction)
			print ('------------')
			introduction = self.introduction_filter(introduction)
			print (introduction)
			print ('------------')

			print (introduction)
			if len(introduction) == 0:
				continue

			self.write_file(introduction, path)

			"""
			img_url = li.select('img')[0].attrs['src']  # 图片链接
			author = li.select('span[class="a-size-small a-color-base"]')[0].get_text()  # 作者名
			# 接下来就是获取链接与写入本地,需要传入链接和路径
			self.tasks_list.append(gevent.spawn(self.down_img, img_url, path))
			self.index += 1
			"""

			self.index += 1
		# 下一页问题
		a_last = soup.select('li[class="a-last"]')
		if a_last:
			# 如果有下一页,获取下一页地址
			a_href = a_last[0].select('a')[0].attrs['href']
			self.url = a_href  # 切换要处理的地址
			self.handle_page()
		else:
			# 如果没有,说明任务加载完毕
			gevent.joinall(self.tasks_list)

	def handle_introduction(self, book_url):
		wb_data = requests.get(book_url)
		soup = BeautifulSoup(wb_data.text, 'lxml')
		info = soup.select('b')
		introduction = ""
		for tag in info:
			temp = tag.get_text()
			length = len(temp)
			if length > 50:
				introduction += temp

		br_info = soup.select('br')
		for tag in info:
			temp = tag.get_text()
			length = len(temp)
			if length > 50:
				introduction += temp
		introduction = self.introduction_filter(introduction)
		return introduction

	def write_file(self, introduction, path):
		f = file(path, 'w+')
		f.write(introduction)
		return

	@staticmethod
	def down_img(img_url, path):
		time.sleep(random.randint(2, 3))  # 随机睡眠1,2秒避免访问频率过高
		img = requests.get(img_url).content  # 获取二进制资源
		with open(path, "wb")as f:  # 二进制写入
			f.write(img)


amazon = Amazon_books()
