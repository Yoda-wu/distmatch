from concurrent.futures import ThreadPoolExecutor
import os

import tensorflow as tf
from flask import request
import matplotlib.pyplot as plt
from base import Manager, Testbed
from base.utils import send_data


class DSSL_Manager(Manager):
    def __init__(self, testbed: Testbed):
        super().__init__(testbed)
        self.time_log = {}
        self.val_acc_log = {}
        self.sup_loss_log = {}
        self.uns_loss_log = {}
        self.manager_base_path = os.path.abspath(os.path.dirname(__file__))
        self.tb_writer = None
        self.executor = ThreadPoolExecutor(1)

        @testbed.flask.route('/routine-log', methods=['POST'])
        def on_routine_log():
            print('POST /routine-log')
            self.executor.submit(self.routine_log, request.json)
            return ''

    def create_tb_writer(self, log_dir):
        self.tb_writer = tf.summary.create_file_writer(log_dir)

    def routine_log(self, log):
        if log['node_name'] in self.val_acc_log.keys():
            self.time_log[log['node_name']].append(log['current_time'])
            self.val_acc_log[log['node_name']].append(log['val_acc'])
            self.sup_loss_log[log['node_name']].append(log['sup_loss'])
            self.uns_loss_log[log['node_name']].append(log['uns_loss'])
        else:
            self.time_log[log['node_name']] = [log['current_time']]
            self.val_acc_log[log['node_name']] = [log['val_acc']]
            self.sup_loss_log[log['node_name']] = [log['sup_loss']]
            self.uns_loss_log[log['node_name']] = [log['uns_loss']]
        total_num = 0
        total_acc_by_time = 0.0
        total_sup_loss_by_time = 0.0
        total_uns_loss_by_time = 0.0
        for acc_list, sup_loss_list, uns_loss_list in zip(self.val_acc_log.values(), self.sup_loss_log.values(),
                                                          self.uns_loss_log.values()):
            total_acc_by_time += acc_list[-1]
            total_sup_loss_by_time += sup_loss_list[-1]
            total_uns_loss_by_time += uns_loss_list[-1]
            total_num += 1
        print('Current Average Validation Accuracy: ', total_acc_by_time / total_num)
        with self.tb_writer.as_default():
            tf.summary.scalar('Validation Accuracy By Time', total_acc_by_time / total_num, step=log['current_time'])
            tf.summary.scalar('Supervised Loss By Time', total_sup_loss_by_time / total_num, step=log['current_time'])
            tf.summary.scalar('Unsupervised Loss By Time', total_uns_loss_by_time / total_num, step=log['current_time'])

    def on_route_start(self, req: request) -> str:
        for pn in self.pNode.values():
            send_data('GET', '/start', pn.ip, pn.port)
        for en in self.eNode.values():
            send_data('GET', '/start', en.ip, en.port)
        print('start training')
        return ''

    def on_route_finish(self, req: request) -> bool:
        """
		need the user to send message to here.
		"""
        return True

    def parse_log_file(self, req: request, filename: str):
        """
		parse log files into pictures.
		the log files format comes from worker/worker_utils.py, log_acc () and log_loss ().
		Aggregate: accuracy=0.8999999761581421, round=1,
		Train: loss=0.2740592360496521, round=1,
		we left a comma at the end for easy positioning and extending.
		"""
        log_file = os.path.join(self.logFileFolder, filename)

        os.path.join(self.logFileFolder, filename)
        acc_str = 'accuracy='
        loss_str = 'loss='
        acc_list = []
        loss_list = []
        with open (os.path.join (self.logFileFolder, filename), 'r') as f:
            for line in f:
                if line.find ('Aggregate') != -1:
                    acc_start_i = line.find (acc_str) + len (acc_str)
                    acc_end_i = line.find (',', acc_start_i)
                    acc = float (line [acc_start_i:acc_end_i])
                    acc_list.append (acc)
                elif line.find ('Train') != -1:
                    loss_start_i = line.find (loss_str) + len (loss_str)
                    loss_end_i = line.find (',', loss_start_i)
                    loss = float (line [loss_start_i:loss_end_i])
                    loss_list.append (loss)
        name = filename [:filename.find ('.log')]
        if acc_list:
            plt.plot (acc_list, 'go')
            plt.plot (acc_list, 'r')
            plt.xlabel ('round')
            plt.ylabel ('accuracy')
            plt.ylim (0, 1)
            plt.title ('Accuracy')
            plt.savefig (os.path.join (self.logFileFolder, 'png/', name + '-acc.png'))
            plt.cla ()
        if loss_list:
            plt.plot (loss_list, 'go')
            plt.plot (loss_list, 'r')
            plt.xlabel ('round')
            plt.ylabel ('loss')
            plt.ylim (0, loss_list [0] * 1.2)
            plt.title ('Loss')
            plt.savefig (os.path.join (self.logFileFolder, 'png/', name + '-loss.png'))
            plt.cla ()
