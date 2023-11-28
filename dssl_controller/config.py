import os
import uuid

from prettytable import PrettyTable


class Configuration:
    def __init__(self, args) -> None:
        # 任务ID
        self.task_id = str(uuid.uuid1())

        # 物理节点IP
        self.testbed_ip = '222.201.187.51'
        self.worker1_ip = '222.201.187.52'
        self.worker2_ip = '222.201.187.50'

        # train/generate
        self.running_mode = args.running_mode

        # 使用的数据集名称（mnist, fmnist, cifar10）
        self.dataset_name = args.dataset

        # 数据集分布
        self.data_distribution = args.dist

        # 数据集及日志输出
        self.controller_base_path = os.path.abspath(os.path.dirname(__file__))
        self.dml_app_path = os.path.join(self.controller_base_path, 'dml_app')
        self.dml_dataset_path = os.path.join(self.controller_base_path, 'dataset')
        self.dataset_path = os.path.join(self.controller_base_path, 'dataset', self.dataset_name)
        self.links_name = 'links.json'
        self.node_info_name = 'node_info.json'
        self.dataset_conf_name = 'dataset_conf.json'
        self.structure_conf_name = 'structure_conf.json'

        # 设备总数及协作设备总数
        self.num_of_devices = 100
        # self.num_of_devices = 100
        if self.running_mode == 'generate':
            # 数据集生成参数
            if self.dataset_name == 'cifar10':
                from dataset_generators.cifar10_generator import Cifar10_Generator
                self.dataset_generator_class = Cifar10_Generator
                self.num_of_classes = 10
                self.num_of_labeled_samples_per_class = 500
            elif self.dataset_name == 'cifar100':
                from dataset_generators.cifar100_generator import Cifar100_Generator
                self.dataset_generator_class = Cifar100_Generator
                self.num_of_classes = 100
                self.num_of_labeled_samples_per_class = 500
            # 随机拓扑连接路径数
            self.num_of_random_paths = 1000
            # 拓扑设置
            self.topology = args.topology
        elif self.running_mode == 'train':
            # 分布式半监督学习算法
            self.dssl_algorithm = args.algorithm
        
        self.config_dict = {}
    
    def generate_config_dict(self):
        '''
        记录配置表，以便后续分析数据读取
        '''
        self.config_dict['task_id'] = self.task_id
        self.config_dict['running_mode'] = self.running_mode
        self.config_dict['dataset_name'] = self.dataset_name
        self.config_dict['data_distribution'] = self.data_distribution
        
        if self.running_mode == 'generate':
            self.config_dict['num_of_devices'] = self.num_of_devices
            self.config_dict['num_of_random_paths'] = self.num_of_random_paths
            self.config_dict['topology'] = self.topology
        elif self.running_mode == 'train':
            self.config_dict['dssl_algorithm'] = self.dssl_algorithm

    def get_config_dict(self):
        return self.config_dict

    def show_configuration(self):
        config_table = PrettyTable()
        config_table.add_column('Configuration Key', list(self.config_dict.keys()))
        config_table.add_column('Configuration Value', list(self.config_dict.values()))
        config_table = str(config_table)
        config_list = config_table.split('\n')
        print('----------------------------CONFIGURATION----------------------------')
        for config_entry in config_list:
            print(config_entry)
        print('----------------------------CONFIGURATION----------------------------')
