import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, request
from config import Configuration
from device_impl.base_device import Base_Device

import dml_utils
import worker_utils
import tensorflow as tf
from model.resnet import resnet10  # configurable parameter, from nns.whatever import nn.

config = Configuration('cifar10', 'iid')
device = Base_Device(config)

dirname = os.path.abspath(os.path.dirname(__file__))

dml_port = os.getenv('DML_PORT')
ctl_addr = os.getenv('NET_CTL_ADDRESS')
agent_addr = os.getenv('NET_AGENT_ADDRESS')
node_name = os.getenv('NET_NODE_NAME')

nn = resnet10(num_classes=10)
nn.build(input_shape=(None, 32, 32, 3))
input_shape = (None, 32, 32, 3)
log_file = os.path.abspath(os.path.join(dirname, '../dml_file/log/', node_name + '.log'))
worker_utils.set_log(log_file)
conf = {}
peer_list = []
# configurable parameter, specify the dataset path.
train_path = os.path.join(dirname, '../dataset/cifar10/iid')
train_images: np.ndarray
train_labels: np.ndarray
# configurable parameter, specify the dataset path.
test_path = os.path.join(dirname, '../dataset/cifar10/iid')
test_images: np.ndarray
test_labels: np.ndarray

app = Flask(__name__)
lock = threading.RLock()
executor = ThreadPoolExecutor(1)
print('Tensorflow Version: ', tf.__version__)


# if this is container, docker will send a GET to here every 30s
# this ability is defined in controller/base/node.py, Class Emulator, save_yml (), healthcheck.
@app.route('/hi', methods=['GET'])
def route_hi():
    # send a heartbeat to the agent.
    # when the agent receives the heartbeat of a container for the first time,
    # it will deploy the container's tc settings.
    # please ensure that your app implements this function, i.e.,
    # receiving docker healthcheck and sending heartbeat to the agent.
    worker_utils.heartbeat(agent_addr, node_name)
    return 'this is node ' + node_name + '\n'


@app.route('/conf/dataset', methods=['POST'])
def route_conf_d():
    f = request.files.get('conf').read()
    conf.update(json.loads(f))
    print('POST at /conf/dataset')

    global device

    device.load_dataset(conf['labeled_dataset'], conf['unlabeled_dataset'], conf['test_dataset'])

    return ''


@app.route('/conf/structure', methods=['POST'])
def route_conf_s():
    f = request.files.get('conf').read()
    conf.update(json.loads(f))
    print('POST at /conf/structure')

    global device
    device.initialize_connecting_device(conf['connect'])

    # filename = os.path.join (dirname, '../dml_file/conf', node_name + '_structure.conf')
    # with open (filename, 'w') as fw:
    # 	fw.writelines (json.dumps (conf, indent=2))

    # conf ['current_round'] = 0
    # peer_list.extend (list (conf ['connect'].keys ()))
    return ''


@app.route('/log', methods=['GET'])
def route_log():
    executor.submit(on_route_log)
    return ''


def on_route_log():
    worker_utils.send_log(ctl_addr, log_file, node_name)


@app.route('/start', methods=['GET'])
def route_start():
    print('GET at /start')
    executor.submit(on_route_start)
    global device
    executor.submit()
    return ''


def on_route_start():
    _, init_acc = dml_utils.test_on_batch(nn.model, test_images, test_labels, conf['batch_size'])
    msg = dml_utils.log_acc(init_acc, 0)
    worker_utils.send_print(ctl_addr, node_name + ': ' + msg)

    with lock:
        gossip()


def gossip():
    peer = dml_utils.random_selection(peer_list, 1)
    worker_utils.log('gossip to ' + peer[0])
    dml_utils.send_weights(nn.model.get_weights(), '/gossip', peer, conf['connect'])


@app.route('/gossip', methods=['POST'])
def route_gossip():
    print('POST at /gossip')
    weights = dml_utils.parse_weights(request.files.get('weights'))
    executor.submit(on_route_gossip, weights)
    return ''


def on_route_gossip(received_weights):
    with lock:
        new_weights = np.add(nn.model.get_weights(), received_weights) / 2
        dml_utils.assign_weights(nn.model, new_weights)

        conf['current_round'] += 1
        loss_list = dml_utils.train(nn.model, train_images, train_labels, conf['epoch'],
                                    conf['batch_size'])
        last_epoch_loss = loss_list[-1]
        msg = dml_utils.log_loss(last_epoch_loss, conf['current_round'])
        worker_utils.send_print(ctl_addr, node_name + ': ' + msg)

        _, acc = dml_utils.test_on_batch(nn.model, test_images, test_labels, conf['batch_size'])
        msg = dml_utils.log_acc(acc, conf['current_round'])
        worker_utils.send_print(ctl_addr, node_name + ': ' + msg)

        if conf['current_round'] < conf['sync']:
            gossip()


app.run(host='0.0.0.0', port=dml_port, threaded=True)
