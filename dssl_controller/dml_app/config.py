import logging
import os

from prettytable import PrettyTable
from dataset.augment import Rand_Augment, Weak_Augment

from model.resnet import resnet10
from utils.logger import Logger


class Configuration:
    def __init__(self) -> None:
        # 基地址
        self.base_path = os.path.abspath (os.path.dirname (__file__))

        # 环境变量
        # HTTP 监听端口
        self.dml_port = os.getenv ('DML_PORT')
        # 测试床控制器所在地址
        self.ctl_addr = os.getenv ('NET_CTL_ADDRESS')
        # 测试床工作节点所在地址
        self.agent_addr = os.getenv ('NET_AGENT_ADDRESS')
        # 节点命名
        self.node_name = os.getenv ('NET_NODE_NAME')
        
        # 批大小
        self.labeled_batch_size = 32

        # 优化器参数
        self.optimizer = 'SGD'
        if self.optimizer == 'SGD':
            self.learning_rate = 0.03
            self.momentum = 0.90
            self.nesterov = True
        elif self.optimizer == 'Adam':
            self.learning_rate = 0.01
            self.beta_1 = 0.9
            self.beta_2 = 0.999

        # 有监督损失系数
        self.loss_lambda = 1.0
        # 无监督损失系数
        self.loss_eta = 1.0

        # 训练轮次及每轮训练代数
        self.num_of_rounds = 200
        self.num_of_epochs_per_round = 1

        # 设备可获取的网络跳数
        self.num_of_jump = 1
    
    def config_dataset(self, dataset_conf: dict):
        # 数据集名
        self.dataset_name: str = dataset_conf['dataset_name']
        
        # 数据集分布
        self.data_distribution: str = dataset_conf['data_distribution']
        
        # 任务ID
        self.task_id: str = dataset_conf['task_id']

        # 训练算法
        self.dssl_algorithm: str = dataset_conf['dssl_algorithm']

        # 数据集基地址
        self.dataset_path = os.path.join(self.base_path, '..', 'dataset', self.dataset_name, self.data_distribution)

        # 数据集信息
        if self.dataset_name in ['cifar10', 'cifar100']:
            self.num_of_classes = 10
            self.input_shape = (None, 32, 32, 3)
            self.network_type = resnet10
            self.weak_augment_helper = Weak_Augment(self.dataset_name)
            self.strong_augment_helper = Rand_Augment(self.dataset_name)

        # 数据集地址
        self.labeled_dataset_path = os.path.join(self.dataset_path, dataset_conf['labeled_dataset'])
        self.unlabeled_dataset_path = os.path.join(self.dataset_path, dataset_conf['unlabeled_dataset'])
        self.test_dataset_path = os.path.join(self.dataset_path, dataset_conf['test_dataset'])

        if self.dssl_algorithm == 'my_dssl':
            from device_impl.my_dssl_device import My_DSSL_Device
            self.device_class = My_DSSL_Device
            # 伪标签接受概率下限与软标签最高概率接受下限
            self.accept_probability = 0.0
            self.deny_probability = 0.8
            self.num_of_helpers = 100
            # 预训练论次
            self.warming_up_rounds = 1
        # 纯有监督
        elif self.dssl_algorithm == 'dist_sgd':
            from device_impl.dist_sgd_device import Distrituted_SGD_Device
            self.device_class = Distrituted_SGD_Device
        # 分布式FixMatch
        elif self.dssl_algorithm == 'dist_fixmatch':
            from device_impl.dist_fixmatch_device import Distributed_FixMatch_Device
            self.device_class = Distributed_FixMatch_Device
            self.accept_probability = 0.95
        # 分布式FixMatch（无标记数据真实标签已知）
        elif self.dssl_algorithm == 'dist_fixmatch_uy_known':
            from device_impl.dist_fixmatch_uy_known_device import Distributed_FixMatch_Uy_Known_Device
            self.device_class = Distributed_FixMatch_Uy_Known_Device

        # 基础日志信息
        self.log_path = os.path.join(self.base_path, '..', 'dml_file', 'log', self.task_id)
        self.logger = Logger(self.ctl_addr, self.node_name, self.log_path)

        # 配置表
        self.config_dict = {}

    def config_structure(self, structure_conf:dict):
        self.connecting_devices_list = structure_conf['connect']

    def generate_config_dict(self):
        '''
        记录配置表，以便后续分析数据读取
        '''
        self.config_dict['task_id'] = self.task_id
        self.config_dict['dataset_name'] = self.dataset_name
        self.config_dict['data_distribution'] = self.data_distribution

        self.config_dict['dssl_algorithm'] = self.dssl_algorithm
        self.config_dict['device_class'] = self.device_class.__name__
        if self.dssl_algorithm == 'my_dssl':
            self.config_dict['accept_probability'] = self.accept_probability
            self.config_dict['deny_probability'] = self.deny_probability
            self.config_dict['num_of_helpers'] = self.num_of_helpers
            self.config_dict['warming_up_rounds'] = self.warming_up_rounds
        elif self.dssl_algorithm == 'gossip_fixmatch':
            self.config_dict['accept_probability'] = self.accept_probability

        self.config_dict['labeled_batch_size'] = self.labeled_batch_size
        self.config_dict['optimizer'] = self.optimizer
        if self.optimizer == 'SGD':
            self.config_dict['learning_rate'] = self.learning_rate
            self.config_dict['momentum'] = self.momentum
            self.config_dict['nesterov'] = self.nesterov
        elif self.optimizer == 'Adam':
            self.config_dict['learning_rate'] = self.learning_rate
            self.config_dict['beta_1'] = self.beta_1
            self.config_dict['beta_2'] = self.beta_2
        
        self.config_dict['loss_lambda'] = self.loss_lambda
        self.config_dict['loss_eta'] = self.loss_eta
        self.config_dict['num_of_rounds'] = self.num_of_rounds
        self.config_dict['num_of_epochs_per_round'] = self.num_of_epochs_per_round
        self.config_dict['num_of_jump'] = self.num_of_jump

        # 数据集划分方式
        if self.dataset_name in ['cifar10', 'cifar100']:
            self.config_dict['weak_augment_helper'] = self.weak_augment_helper.__class__.__name__
            self.config_dict['strong_augment_helper'] = self.strong_augment_helper.__class__.__name__


    def get_config_dict(self):
        return self.config_dict

    def get_logger(self):
        return self.logger

    def show_configuration(self):
        config_table = PrettyTable()
        config_table.add_column('Configuration Key', list(self.config_dict.keys()))
        config_table.add_column('Configuration Value', list(self.config_dict.values()))
        config_table = str(config_table)
        config_list = config_table.split('\n')
        self.logger.log('----------------------------CONFIGURATION----------------------------')
        for config_entry in config_list:
            self.logger.log(config_entry)
        self.logger.log('----------------------------CONFIGURATION----------------------------')
