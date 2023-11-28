import abc
import gc
import logging
import math
import os
import random
import time
import tensorflow as tf
import threading
from typing import List

import numpy as np
from config import Configuration
from dml_app.device_impl.neighbor_device import Neighbor_Device
from dml_app.device_impl.preprocessor import Dataset_Preprocessor
from keras import optimizers
from dml_app.model.resnet import resnet10
from utils.numpy_helpers import np_load
from dml_app.worker_utils import send_data


class Base_Device(metaclass=abc.ABCMeta):
    def __init__(self, config: Configuration) -> None:
        # 原始数据集
        self.labeled_x = []
        self.labeled_y = []
        self.unlabeled_x = []
        self.unlabeled_y = []
        self.test_x = []
        self.test_y = []

        # 数据集特征
        self.input_shape = config.input_shape
        self.num_of_classes = config.num_of_classes

        # 网络类型
        self.network_type = config.network_type

        # 训练轮次
        self.current_round = 0

        # 每轮训练代数
        self.num_of_epochs_per_round = config.num_of_epochs_per_round

        # 连接的节点
        self.connecting_devices_list: List [Neighbor_Device] = []
        self.fetched_models = []

        # 优化器
        if config.optimizer == 'SGD':
            self.optimizer = optimizers.gradient_descent_v2.SGD(
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                nesterov=config.nesterov
            )
        elif config.optimizer == 'Adam':
            self.optimizer = optimizers.adam_v2.Adam(
                learning_rate=config.learning_rate,
                beta_1=config.beta_1,
                beta_2=config.beta_2
            )

        # 损失系数
        self.loss_lambda = config.loss_lambda
        self.loss_eta = config.loss_eta

        # 模型读写锁，禁止在写模型参数时读模型参数
        self.lock = threading.RLock()

        # 用于传递给其他节点的模型信息
        self.model_info = {}

        # 配置表
        self.config = config

        # 日志打印器
        self.logger = config.get_logger()

        # 训练日志
        self.training_logs = {
            'configuration': config.get_config_dict(),
            'log_time': [],
            'validation_accuracy': [],
            'supervised_loss': [],
            'unsupervised_loss': [],
            'test_accuracy': 0.0
        }

        # Tensorboard记录器
        self.tb_writer = tf.summary.create_file_writer(os.path.join(config.log_path, config.node_name))

        # 外部接口地址
        self.heartbeat_path = '/heartbeat?name=' + config.node_name
        self.log_path = '/log?name=' + config.node_name
    
    def initialize(self):
        self.__initialize_connecting_devices_list()
        self.__initialize_dataset()
        self.__initialize_model()
        self.__initialize_assistant_model()
        self.__update_model_info()
    
    def __initialize_connecting_devices_list(self):
        # 连接列表形式为{node_name: node_addr}
        for node_name, node_addr in self.config.connecting_devices_list.items():
            self.connecting_devices_list.append(Neighbor_Device(node_name, node_addr))

        self.logger.log(f'connecting devices list: {[node.get_node_name() for node in self.connecting_devices_list]}')
    
    def __initialize_dataset(self):
        labeled_dataset = np_load(self.config.labeled_dataset_path)
        unlabeled_dataset = np_load(self.config.unlabeled_dataset_path)
        test_dataset = np_load(self.config.test_dataset_path)

        self.labeled_x = labeled_dataset['x']
        self.labeled_y = labeled_dataset['y']
        self.unlabeled_x = unlabeled_dataset['x']
        self.unlabeled_y = unlabeled_dataset['y']
        self.test_x = test_dataset['x']
        self.test_y = test_dataset['y']

        self.dataset_preprocessor = Dataset_Preprocessor(self.config.dataset_name, self.config.weak_augment_helper, self.config.strong_augment_helper)
        self.labeled_batch_size = self.config.labeled_batch_size
        ratio = self.labeled_batch_size / self.labeled_x.shape[0]
        self.unlabeled_batch_size = math.ceil(self.unlabeled_x.shape[0] * ratio)

        self.testing_db = self.dataset_preprocessor.preprocess_labeled_dataset(
            self.test_x,
            self.test_y,
            batch_size=1000
        )
        self.testing_db = list(self.testing_db)

        self.logger.log('dataset loaded')

    def __initialize_model(self):
        '''
        初始化本设备的网络
        '''
        with self.lock:
            self.model = self.network_type(num_classes=self.config.num_of_classes)
            self.model.build(input_shape=self.config.input_shape)
        self.logger.log('model built')
    
    def __initialize_assistant_model(self):
        '''
        初始化本设备使用的协作网络，用于加载其他设备的网络
        '''
        self.assistant_model = self.network_type(num_classes=self.config.num_of_classes)
        self.assistant_model.build(input_shape=self.config.input_shape)
        self.logger.log('assistant model built')
    
    def train(self):
        start_time = time.time()
        for self.current_round in range(1, self.config.num_of_rounds + 1):
            self.logger.log(f'round {self.current_round} begin training')

            # 训练
            total_supervised_loss, total_num_of_labeled_data, total_unsupervised_loss, total_num_of_unlabeled_data = self.train_one_round()
            supervised_loss = total_supervised_loss/ total_num_of_labeled_data
            unsupervised_loss = total_unsupervised_loss/ total_num_of_unlabeled_data
            self.logger.log(f'round {self.current_round} supervised loss: {supervised_loss}')
            self.logger.log(f'round {self.current_round} unsupervised loss: {unsupervised_loss}')

            # 拉取模型
            self.fetch_model()

            # 更新协助模型
            self.update_helpers()

            # 聚合模型
            self.aggregate_model_with_neighbor()

            # 清理内存
            del self.fetched_models
            # self.clear_all_device()

            # 模型评估
            acc = self.evaluate(False)
            self.logger.log(f'round: {self.current_round} accuracy: {float(acc)}')
            
            # 日志记录
            current_time = time.time() - start_time
            self.training_logs['log_time'].append(current_time)
            self.training_logs['supervised_loss'].append(supervised_loss)
            self.training_logs['unsupervised_loss'].append(unsupervised_loss)
            self.training_logs['validation_accuracy'].append(acc)
            with self.tb_writer.as_default():
                tf.summary.scalar('Supervised Loss', supervised_loss, step=self.current_round)
                tf.summary.scalar('Unsupervised Loss', unsupervised_loss, step=self.current_round)
                tf.summary.scalar('Validation Accuracy', acc, step=self.current_round)
            self.logger.routine_log(self.current_round, current_time, supervised_loss, unsupervised_loss, acc)
            gc.collect()
        
        # 训练结束评估
        acc = self.evaluate(True)
        self.training_logs['test_accuracy'] = acc

    @abc.abstractclassmethod
    def train_one_round(self):
        pass

    def fetch_model(self):
        '''
        获取规定跳数以内的所有网络信息及设备字典
        更新self.device_list_to_fetch（仅第一轮）与self.fetched_models
        '''
        self.fetched_models = []
        # 保存所有网络文件及设备的引用
        # device_list_to_fetch = random.choices(self.connecting_devices_list, k = 5)
        for device in self.connecting_devices_list:
            device.sync_with_origin()
            self.fetched_models.append(
                {
                    'device': device,
                    'weights': device.get_model_weight()
                }
            )

    @abc.abstractclassmethod
    def update_helpers(self):
        pass

    def model_staleness(self, round):
        return 1.0

    def aggregate_model_with_neighbor(self):
        '''
        与邻居节点聚合模型
        '''

        # # 获取当前网络的参数
        # model_weights = self.get_dataset_size()
        # new_model_parameters = np.array(self.model.get_weights()) * model_weights
        # all_model_weights = model_weights

        # # 对网络的逐层求加权平均，得到新网络参数并加载到网络
        # for fetched_model in self.fetched_models:
        #     model_weights = self.model_staleness(fetched_model['device'].get_current_round()) * fetched_model['device'].get_dataset_size()
        #     new_model_parameters = np.add(new_model_parameters, np.array(fetched_model['weights']) * model_weights)
        #     all_model_weights += model_weights
        # new_model_parameters = np.divide(new_model_parameters, all_model_weights)

        # with self.lock:
        #     self.model.set_weights(new_model_parameters)
        
        # del new_model_parameters, all_model_weights

        all_model_parameters = [self.model.get_weights()]
        all_model_weights = [self.current_round * self.get_dataset_size()]
        # 获取邻居节点的网络参数
        for fetched_model in self.fetched_models:
            all_model_parameters.append(fetched_model['weights'])
            all_model_weights.append(fetched_model['device'].get_current_round() * fetched_model['device'].get_dataset_size())

        # 对网络的逐层求加权平均，得到新网络参数并加载到网络
        new_model_parameters = [
            np.average(
                [all_model_parameters[j][i] for j in range(len(all_model_parameters))],
                axis=0,
                weights=all_model_weights
            ) for i in range(len(all_model_parameters[0]))]

        with self.lock:
            self.model.set_weights(new_model_parameters)
        del all_model_parameters, all_model_weights, new_model_parameters
        self.logger.log(f'round: {self.current_round} aggregated')

    def clear_all_device(self):
        for device in self.connecting_devices_list:
            device.clear()

    def evaluate(self, total_test = False):
        '''
        使用指定测试集评估本机模型上的TOP-1精度
        '''
        acc = 0.0
        if total_test:
            acc = self.__total_evaluate(self.model, self.testing_db)
        else:
            acc = self.__partial_evaluate(self.model, self.testing_db)
        return acc
    
    def __partial_evaluate(self, model, testing_db):
        '''
        使用指定模型，在指定测试集上计算其TOP-1精度
        '''
        i = random.choice(range(len(testing_db)))
        x, y = testing_db[i]
        logits = model(x, training=False)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        acc = int(correct) / x.shape[0]
        del logits, prob, pred, correct
        return acc

    def __total_evaluate(self, model, testing_db):
        '''
        使用指定模型，在指定测试集上计算其TOP-1精度
        '''
        total_num = 0
        total_correct = 0
        for x,y in testing_db:
            logits = model(x, training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        del total_correct, total_num
        return acc
    
    def __update_model_info(self):
        with self.lock:
            if self.model_info == {} or self.model_info['round'] != self.current_round:
                weights = self.model.get_weights()
                self.model_info = {
                    'weight': [
                        layer.tolist() for layer in weights
                    ],
                    'round': self.current_round,
                    'dataset_size': self.get_dataset_size()
                }
        
    def get_dataset_size(self):
        '''
        获取设备数据集大小
        '''
        return len(self.labeled_x) + len(self.unlabeled_x)

    def on_heartbeat(self):
        send_data('GET', self.heartbeat_path, self.config.agent_addr)
        return 'this is node ' + self.config.node_name + '\n'

    def on_log(self):
        with open(self.logger.log_file, 'r') as f:
            send_data ('POST', self.log_path, self.config.ctl_addr, files={'log': f})
        
    def on_start(self):
        self.config.generate_config_dict()
        self.config.show_configuration()
        self.initialize()
        self.train()

    def on_get_model_info(self):
        self.__update_model_info()
        return self.model_info

    def on_get_current_round(self):
        return {'round': self.current_round}