import re
import time

import matplotlib.pyplot as plt


# 基类 以后可以再添加多种方法
class logFun ():
	# def basicconfig(self):
	#     pass
	# def info(self):
	#     pass
	def infoAcc (self):
		pass

	def infoLoss (self):
		pass


# 被代理类
class beProxy (logFun):
	# def basicconfig(self):
	#     logging.basicConfig(level=logging.INFO, filename='log/parameter_server.log', filemode='w',
	#                         format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
	# def info(self):
	# logging.info('round {}:accuracy={}'.format(v['current_round'], nn['sess'].run(nn['accuracy'], feed_dict={
	#     nn['xs']: nn['test_x'], nn['ys']: nn['test_y']})))

	# logging.info('round 0: accuracy={}'.format(nn['sess'].run(nn['accuracy'], feed_dict={
	#      nn['xs']: nn['test_x'], nn['ys']: nn['test_y']})))
	def infoAcc (self):
		fp = open ("log/parameter_server.log")
		accuracy = []
		for line in fp:
			m = re.search ('accuracy', line)
			if m:
				n = re.search ('[0]\.[0-9]+', line)
				if n is not None:
					accuracy.append (float (n.group ()))
		fp.close ()

		print (accuracy)
		print (type (accuracy [0]))

		plt.plot (accuracy, 'go')
		plt.plot (accuracy, 'r')
		plt.xlabel ('count1')
		plt.ylabel ('accuracy')
		plt.ylim (0, 1)
		printtime = time.strftime ('%Y-%m-%d-%H-%M-%S', time.localtime (time.time ()))
		plt.savefig ("./log/a-" + str (printtime) + ".jpg")
		plt.title ('Accuracy')

	def infoLoss (self):
		worker = [0, 1]
		for workernum in worker:
			fp = open ('log/worker_' + str (workernum) + '.log')

			loss = []

			for line in fp:
				m = re.search ('loss', line)

				if m:
					n = re.search ('[0]\.[0-9]+', line)
					if n is not None:
						loss.append (float (n.group ()))

			fp.close ()

			print (loss)
			print (type (loss [0]))

			plt.cla ()
			plt.plot (loss, 'go')
			plt.plot (loss, 'r')
			plt.xlabel ('count' + str (workernum))
			plt.ylabel ('loss' + str (workernum))
			plt.ylim (0, 1)
			plt.title ('Loss' + str (workernum))
			printtime = time.strftime ('%Y-%m-%d-%H-%M-%S', time.localtime (time.time ()))
			plt.savefig ('log/w_' + str (workernum) + '-' + printtime + ".jpg")


# 代理类
class proxy (logFun):
	def __init__ (self):
		self.log_Fun = beProxy ()

	def set_logFun (self, logFun):
		self.log_Fun = logFun

	# def basicconfig(self):
	#     self.log_Fun.basicconfig()
	# def info(self):
	#     self.log_Fun.info()
	def infoAcc (self):
		self.log_Fun.infoAcc ()

	def infoLoss (self):
		self.log_Fun.infoLoss ()


class logact (object):
	# def star1(self):
	#     a1 = proxy()
	#     a1.basicconfig()
	# def star2(self):
	#     a2 = proxy()
	#     a2.info()
	def star3 (self):
		a3 = proxy ()
		a3.infoAcc ()

	def star4 (self):
		a4 = proxy ()
		a4.infoLoss ()
