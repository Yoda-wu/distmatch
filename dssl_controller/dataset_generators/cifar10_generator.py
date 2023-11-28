import copy
import os
import time
import tensorflow_datasets as tfds
import numpy as np
import random
import config
import json
from utils.connection import Connection_Generator
from utils.numpy_helpers import np_save


class Cifar10_Generator:
    def __init__(self, config: config.Configuration) -> None:
        self.dataset_name = config.dataset_name
        self.data_distribution = config.data_distribution
        self.dataset_base_dir = os.path.join(config.dataset_path, self.data_distribution)
        # self.num_of_samples_per_class_in_testing_set = config.num_of_samples_per_class_in_testing_set
        self.num_of_labeled_samples_per_class = config.num_of_labeled_samples_per_class
        self.num_of_devices = config.num_of_devices
        self.num_of_random_paths = config.num_of_random_paths
        self.topology = config.topology
        self.topology_file_path = os.path.join(config.controller_base_path, config.links_name)
        self.dataset_conf_file_path = os.path.join(config.controller_base_path, config.dataset_conf_name)
        self.structure_conf_file_path = os.path.join(config.controller_base_path, config.structure_conf_name)

    def generate(self) -> None:
        print(f'Generating {self.dataset_name} with {self.data_distribution} in {self.dataset_base_dir}')
        start_time = time.time()
        # 生成连接拓扑
        self.__generate_device_connection()
        # 加载数据集
        training_set, testing_set = self.__load_dataset()
        # # 划分数据集为训练集和测试集
        # training_set, testing_set = self.__split_training_testing_set(dataset_by_index)
        # 划分训练集为有标签与无标签训练集
        labeled_dataset, unlabeled_dataset = self.__split_unlabeled_and_labeled(training_set)
        # 确定每一个设备上的数据分布
        devices_dataset_distribution = self.__generate_distribution(training_set['labels'].size)
        # 根据分布划分每个设备上的数据
        labeled_dataset_by_devices = self.__split_dataset_by_devices(labeled_dataset, devices_dataset_distribution)
        unlabeled_dataset_by_devices = self.__split_dataset_by_devices(unlabeled_dataset, devices_dataset_distribution)
        # 保存数据集
        testing_set['name'] = f'test_{self.dataset_name}'
        self.__save_dataset(testing_set)
        for device_id in range(self.num_of_devices):
            labeled_dataset_by_devices[device_id]['name'] = f'labeled_{self.dataset_name}_{device_id}'
            unlabeled_dataset_by_devices[device_id]['name'] = f'unlabeled_{self.dataset_name}_{device_id}'
            self.__save_dataset(labeled_dataset_by_devices[device_id])
            self.__save_dataset(unlabeled_dataset_by_devices[device_id])
        self.__save_conf()
        print(f'{self.dataset_name} with {self.data_distribution} generated ({time.time() - start_time}s)')

    def __save_conf(self):
        dataset_conf = {
            'n' + str(i): {
                'labeled_dataset': f'labeled_{self.dataset_name}_{i}.npy',
                'unlabeled_dataset': f'unlabeled_{self.dataset_name}_{i}.npy',
                'test_dataset': f'test_{self.dataset_name}.npy'
            }
            for i in range(self.num_of_devices)
        }

        structure_conf = {
            'sync': 200,
            'node_list': [
                {
                    'name': 'n' + str(i),
                    'epoch': 1
                } for i in range(self.num_of_devices)
            ]
        }

        dataset_conf_json = json.dumps(dataset_conf, indent=4)
        with open(self.dataset_conf_file_path, 'w') as f:
            f.write(dataset_conf_json)

        structure_conf_json = json.dumps(structure_conf, indent=4)
        with open(self.structure_conf_file_path, 'w') as f:
            f.write(structure_conf_json)

    def __generate_device_connection(self):
        # 拓扑设置
        connection_generator = Connection_Generator(self.num_of_random_paths, self.num_of_devices)
        if self.topology == 'fc':
            device_connecting_topology = connection_generator.generate_fully_connected_topology()
        elif self.topology == 'star':
            device_connecting_topology = connection_generator.generate_star_topology()
        elif self.topology == 'round':
            device_connecting_topology = connection_generator.generate_round_topology()
        elif self.topology == 'rt':
            device_connecting_topology = connection_generator.generate_random_tree_topology()
        elif self.topology == 'random':
            device_connecting_topology = connection_generator.generate_random_connected_topology()

        device_connecting_topology_json = {
            'n' + str(i): [
                {
                    'dest': 'n' + str(dest),
                    'bw': str(random.randint(1, 5)) + 'mbps'
                }
                for dest in device_connecting_topology[i]
            ]
            for i in range(self.num_of_devices)
        }

        device_connecting_topology_json = json.dumps(device_connecting_topology_json, indent=4)
        with open(self.topology_file_path, 'w') as f:
            f.write(device_connecting_topology_json)

        # device_connecting_topology = {
        #     'num_of_devices': self.num_of_devices,
        #     'num_of_random_paths': self.num_of_random_paths,
        #     'topology_name': self.topology,
        #     'topology': device_connecting_topology
        # }
        # np_save(self.dataset_base_dir, 'device_connection_topology.npy', device_connecting_topology)
        print('device connection topology generated')

    def __load_dataset(self):
        # ds, ds_info = tfds.load('stl10', split='unlabelled', with_info=True)
        # fig = tfds.show_examples(ds, ds_info)
        from tensorflow_datasets.core.utils import gcs_utils
        gcs_utils._is_gcs_disabled = True
        print('dataset load')
        train = tfds.load('cifar10', split='train', shuffle_files=True,try_gcs=False)
        print('dataset load')
        train = list(train)
        test = tfds.load('cifar10', split='test', shuffle_files=True)
        test = list(test)

        training_set = {
            'x': [],
            'y': [],
            'length': 0,
            'labels': []
        }
        testing_set = copy.deepcopy(training_set)

        for raw_dataset, dataset in [(train, training_set), (test, testing_set)]:
            for data in raw_dataset:
                dataset['x'].append(data['image'].numpy())
                dataset['y'].append(data['label'].numpy())
            dataset['x'] = np.array(dataset['x'])
            dataset['y'] = np.array(dataset['y'])
            dataset['length'] = len(dataset['x'])
            dataset['labels'] = np.unique(dataset['y'])
        print('dataset loaded')
        return training_set, testing_set

    # def __split_training_testing_set(self, dataset_by_index: dict):
    #     '''
    #     重新划分训练集与测试集
    #     参数格式
    #     dataset_by_index@dict
    #     {
    #         @label: @list[image]
    #     }
    #     返回格式
    #     training_set, testing_set@dict
    #     {
    #         'x': @list[image],
    #         'y': @list[label],
    #         'length': @int,        # 数据集长度
    #         'labels': @list[label] # 'y'中包含的标签种类
    #     }
    #     '''
    #     training_set_by_index = {}
    #     testing_set_by_index = {}
    #     for key, value in dataset_by_index.items():
    #         training_set_by_index[key] = value[self.num_of_samples_per_class_in_testing_set:]
    #     for key, value in dataset_by_index.items():
    #         testing_set_by_index[key] = value[:self.num_of_samples_per_class_in_testing_set]

    #     training_set = {
    #         'x': [],
    #         'y': [],
    #         'length': 0,
    #         'labels': None
    #     }
    #     for key, value in training_set_by_index.items():
    #         training_set['x'].extend(value)
    #         training_set['y'].extend([key for _ in range(len(value))])
    #         training_set['length'] += len(value)
    #     training_set['x'] = np.array(training_set['x'])
    #     training_set['y'] = np.array(training_set['y'])
    #     training_set['labels'] = np.unique(training_set['y'])

    #     testing_set = {
    #         'x': [],
    #         'y': [],
    #         'length': 0,
    #         'labels': None
    #     }
    #     for key, value in testing_set_by_index.items():
    #         testing_set['x'].extend(value)
    #         testing_set['y'].extend([key for _ in range(len(value))])
    #         testing_set['length'] += len(value)
    #     testing_set['x'] = np.array(testing_set['x'])
    #     testing_set['y'] = np.array(testing_set['y'])
    #     testing_set['labels'] = np.unique(testing_set['y'])

    #     return training_set, testing_set

    def __split_unlabeled_and_labeled(self, training_set):
        '''
        将训练集划分为有标签与无标签训练集
        参数格式
        training_set@dict
        {
            'x': @list[image],
            'y': @list[label],
            'length': @int,        # 数据集长度
            'labels': @list[label] # 'y'中包含的标签种类
        }
        返回格式
        labeled_dataset, unlabeled_dataset@dict
        {
            'data': @dict
            {
                @label: @dict
                {
                    'x': @list[image],
                    'y': @list[label]
                }
            }
            'length': @int        # 数据集长度
        }
        '''

        data_by_label = {}
        labels = training_set['labels']
        x = training_set['x']
        y = training_set['y']
        for label in labels:
            idx = np.where(y == label)[0]
            data_by_label[label] = {
                'x': x[idx],
                'y': y[idx]
            }
        labeled_dataset = {
            'data': {},
            'length': self.num_of_labeled_samples_per_class * labels.size
        }
        unlabeled_dataset = {
            'data': {},
            'length': 0
        }
        for label, data in data_by_label.items():
            labeled_dataset['data'][label] = {
                'x': data['x'][:self.num_of_labeled_samples_per_class],
                'y': data['y'][:self.num_of_labeled_samples_per_class]
            }
            unlabeled_dataset['data'][label] = {
                'x': data['x'][self.num_of_labeled_samples_per_class:],
                'y': data['y'][self.num_of_labeled_samples_per_class:]
            }
            unlabeled_dataset['length'] += len(unlabeled_dataset['data'][label]['x'])
        return labeled_dataset, unlabeled_dataset

    def __split_dataset_by_devices(self, dataset, distribution):
        '''
        为每个设备根据分布划分数据集
        参数格式
        dataset@dict
        {
            'data': @dict
            {
                @label: @dict
                {
                    'x': @list[image],
                    'y': @list[label]
                }
            }
            'length': @int        # 数据集长度
        }
        distribution@list[@list[@int len(num_of_labels_types)] len(num_of_device)]
        返回格式
        dataset_by_devices@dict
        {
            @device_id:@dict
            {
                'device_id': device_id,
                'labels': np.unique(y),
                'length': len(x),
                'x': @list[image],
                'y': @list[label]
            }
        }
        '''
        labels = list(dataset['data'].keys())
        num_of_samples_per_client = int(dataset['length'] / self.num_of_devices)
        offset_per_label = {label: 0 for label in labels}
        dataset_by_devices = {}
        for device_id in range(self.num_of_devices):
            x = []
            y = []
            # 根据设备上的数据分布选取数据集中所有样本对应的标签
            # freqs = np.random.choice(labels, num_of_samples_per_client, p=distribution[device_id])
            for label, data in dataset['data'].items():
                # num_instances = len(freqs[freqs == label]) 
                num_instances = int(num_of_samples_per_client * distribution[device_id][label])
                start = offset_per_label[label]
                end = offset_per_label[label] + num_instances
                x = [*x, *data['x'][start:end]]
                y = [*y, *data['y'][start:end]]
                offset_per_label[label] = end
            x, y = self.__shuffle(x, y)
            dataset_by_devices[device_id] = {
                'device_id': device_id,
                'labels': np.unique(y),
                'length': len(x),
                'x': x,
                'y': y
            }
        return dataset_by_devices

    def __shuffle(self, x, y):
        idx = np.arange(len(x))
        random.shuffle(idx)
        return np.array(x)[idx], np.array(y)[idx]

    def __generate_distribution(self,
                                num_of_types_of_labels: int, ):
        if self.data_distribution == 'iid':
            return [[1.0 / num_of_types_of_labels for _ in range(num_of_types_of_labels)]
                    for __ in range(self.num_of_devices)]
        elif self.data_distribution == 'noniid':
            ten_types_of_class_imbalanced_dist = [
                [0.50, 0.16, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.16],  # type 0
                [0.16, 0.50, 0.16, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04],  # type 1
                [0.04, 0.16, 0.50, 0.16, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02],  # type 2
                [0.02, 0.04, 0.16, 0.50, 0.16, 0.04, 0.02, 0.02, 0.02, 0.02],  # type 3
                [0.02, 0.02, 0.04, 0.16, 0.50, 0.16, 0.04, 0.02, 0.02, 0.02],  # type 4
                [0.02, 0.02, 0.02, 0.04, 0.16, 0.50, 0.16, 0.04, 0.02, 0.02],  # type 5
                [0.02, 0.02, 0.02, 0.02, 0.04, 0.16, 0.50, 0.16, 0.04, 0.02],  # type 6
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.16, 0.50, 0.16, 0.04],  # type 7
                [0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.16, 0.50, 0.16],  # type 8
                [0.16, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.04, 0.16, 0.50],  # type 9
            ]
            # 根据设备号循环使用上述10类分布
            distribution = [ten_types_of_class_imbalanced_dist[i % len(ten_types_of_class_imbalanced_dist)]
                            for i in range(self.num_of_devices)]
            return distribution

    def __save_dataset(self, dataset):
        np_save(self.dataset_base_dir, f"{dataset['name']}.npy", data=dataset)
        print(f"filename:{dataset['name']}, "
              f"labels:[{','.join(map(str, dataset['labels']))}], "
              f"num_examples:{len(dataset['x'])}")
