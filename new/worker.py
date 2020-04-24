import getopt
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request

import util
from nn import nn_lr
from values import values_w

nn = nn_lr.get_nn ()
v = values_w.get_values ()

try:
	options, args = getopt.getopt (sys.argv [1:], 'i:r:b:l:j:k',
		['worker_index=', 'round=', 'batch_size=', 'local_epoch_num=', 'start_data_index=',
		 'end_data_index='])
except getopt.GetoptError:
	sys.exit ()

self_index = -1
for option, value in options:
	if option in ('-i', '--worker_index'):
		self_index = int (value)
	elif option in ('-r', '--round'):
		v ['round'] = int (value)
	elif option in ('-b', '--batch_size'):
		v ['batch_size'] = int (value)
	elif option in ('-l', '--local_epoch_num'):
		v ['local_epoch_num'] = int (value)
	elif option in ('-j', '--start_data_index'):
		v ['start_index'] = int (value)
	elif option in ('-k', '--end_data_index'):
		v ['end_index'] = int (value)
if len (args) > 0:
	print ('error args: {0}'.format (args))

nn_lr.set_train_data_batch (v ['batch_size'], v ['round'], v ['start_index'], v ['end_index'])
nn_lr.set_train_lr (v ['learning_rate'])

logging.basicConfig (level=logging.INFO, filename='log/worker_' + str (self_index) + '.log', filemode='w',
	format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

this_port = v ['start_port'] + self_index

app = Flask (__name__)
executor = ThreadPoolExecutor (1)
write = io.BytesIO ()


@app.route ('/train', methods=['POST'])
def route_train ():
	w = util.parse_received_weight (request.files.get ('weights'))
	executor.submit (on_route_train, w)
	return 'continue training'


def on_route_train (received_w):
	util.assignment (nn ['assign_list'], received_w, nn ['sess'])
	final_loss = util.train (v ['local_epoch_num'], nn ['batch_num'], nn ['sess'], nn ['batch'], nn ['loss'],
		nn ['train_step'], nn ['xs'], nn ['ys'])

	v ['current_round'] += 1
	logging.info ('worker {} round {}:loss={}'.format (self_index, v ['current_round'], final_loss))

	latest_weights = nn ['sess'].run (nn ['weights'])
	util.send_weight_up_combine (write, latest_weights, v ['up_addr'])


@app.route ('/heart_beat', methods=['GET'])
def heart_beat ():
	return 'alive'


app.run (host='0.0.0.0', port=this_port, threaded=True)
