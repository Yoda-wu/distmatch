import json
import os
from typing import Dict
from base import default_testbed
from base.manager import NodeInfo
from base.utils import read_json
from config import Configuration
from dssl_manager import DSSL_Manager


class Conf:
    def __init__(self, name, sync, epoch):
        self.name = name
        self.sync = sync
        self.epoch = epoch
        self.connect = {}

    def __hash__(self):
        return hash(self.name)

    def to_json(self):
        return {
            'sync': self.sync,
            'epoch': self.epoch,
            'connect': self.connect,
        }


class Controller:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.num_of_devices = config.num_of_devices
        self.controller_base_path = config.controller_base_path
        self.dml_app_path = config.dml_app_path
        self.dml_dataset_path = config.dml_dataset_path
        self.topology_file_path = os.path.join(config.controller_base_path, config.links_name)
        self.testbed = default_testbed(ip=config.testbed_ip, dir_name=self.controller_base_path,
                                       manager_class=DSSL_Manager, host_port=9000)
        self.testbed.manager.create_tb_writer(
            os.path.join(config.controller_base_path, 'dml_file', 'log', config.task_id))

    def config_dataset(self):
        '''
        配置数据集，生成#(node_name)_dataset.conf文件
        '''
        with open(os.path.join(self.config.controller_base_path, self.config.dataset_conf_name), 'r') as f:
            dataset_conf = json.loads(f.read())

        for node_name in dataset_conf:
            node_conf = dataset_conf[node_name]
            node_conf['dataset_name'] = self.config.dataset_name
            node_conf['data_distribution'] = self.config.data_distribution
            node_conf['dssl_algorithm'] = self.config.dssl_algorithm
            node_conf['task_id'] = self.config.task_id
            conf_path = os.path.join(self.config.controller_base_path, 'dml_file', 'conf', node_name + '_dataset.conf')
            with open(conf_path, 'w') as f:
                f.writelines(json.dumps(node_conf, indent=2))

    def config_structure(self):
        '''
        配置连接节点，生成#(node_name)_structure.conf文件
        '''
        emulated_node, physical_node, all_node = self.load_node_info()
        conf_json = read_json(os.path.join(self.config.controller_base_path, self.config.structure_conf_name))
        link_json = read_json(os.path.join(self.config.controller_base_path, self.config.links_name))
        self.generate_structure_conf(all_node, conf_json, link_json)

    def initialize(self):
        nfsApp = self.testbed.add_nfs(tag='dml_app', path=self.dml_app_path)
        nfsDataset = self.testbed.add_nfs(tag='dataset', path=self.dml_dataset_path)

        emulator_1 = self.testbed.add_emulator('emulator-1', self.config.worker1_ip, cpu=128, ram=512, unit='G')
        for i in range(0, 50):
            en = self.testbed.add_emulated_node('n' + str(i), '/home/worker/dml_app',
                                                ['python3', 'app_main.py'], 'dml:v1.0', cpu=2, ram=8, unit='G',
                                                emulator=emulator_1)
            en.mount_local_path('./dml_file', '/home/worker/dml_file')
            en.mount_nfs(nfsApp, '/home/worker/dml_app')
            en.mount_nfs(nfsDataset, '/home/worker/dataset')

        emulator_2 = self.testbed.add_emulator('emulator-2', self.config.worker2_ip, cpu=128, ram=512, unit='G')
        for i in range(50, 100):
            en = self.testbed.add_emulated_node('n' + str(i), '/home/worker/dml_app',
                                                ['python3', 'app_main.py'], 'dml:v1.0', cpu=2, ram=8, unit='G',
                                                emulator=emulator_2)
            en.mount_local_path('./dml_file', '/home/worker/dml_file')
            en.mount_nfs(nfsApp, '/home/worker/dml_app')
            en.mount_nfs(nfsDataset, '/home/worker/dataset')

        # # 仅用一台3990x启动模拟器
        # emulator = self.testbed.add_emulator ('emulator', self.config.worker1_ip, cpu=128, ram=512, unit='G')
        # for i in range (0, 100):
        #     en = self.testbed.add_emulated_node ('n' + str (i), '/home/worker/dml_app',
        #         ['python3', 'app_main.py'], 'dml:v1.0', cpu=1, ram=4, unit='G', emulator=emulator)
        #     en.mount_local_path ('./dml_file', '/home/worker/dml_file')
        #     en.mount_nfs (nfsApp, '/home/worker/dml_app')
        #     en.mount_nfs (nfsDataset, '/home/worker/dataset')

        links_json = read_json(self.topology_file_path)
        self.testbed.load_link(links_json)
        self.testbed.save_config()
        # self.testbed.manager

    def initialize_physical(self):
        nfsApp = self.testbed.add_nfs(tag='dml_app', path=self.dml_app_path)
        nfsDataset = self.testbed.add_nfs(tag='dataset', path=self.dml_dataset_path)
        links_json =read_json(self.topology_file_path)
        self.testbed.load_link(links_json)
        self.testbed.save_config()


    def start(self):
        # 变更环境用这个
        # self.testbed.start (build_emulated_env=True)
        self.testbed.start()

    def load_node_info(self):
        """
        return three dicts: emulated node only, physical node only, and all node.
        """
        node_info_json = read_json(os.path.join(self.config.controller_base_path, self.config.node_info_name))

        emulated_node: Dict[str, NodeInfo] = {}
        physical_node: Dict[str, NodeInfo] = {}
        all_node: Dict[str, NodeInfo] = {}

        emulated_node_json = node_info_json['emulated_node']
        for name, val in emulated_node_json.items():
            emulated_node[name] = NodeInfo(name, val['ip'], val['port'])
            all_node[name] = emulated_node[name]

        physical_node_json = node_info_json['physical_node']
        for name, val in physical_node_json.items():
            physical_node[name] = NodeInfo(name, val['ip'], val['port'])
            all_node[name] = physical_node[name]

        return emulated_node, physical_node, all_node

    def generate_structure_conf(self, all_node, conf_json, link_json):
        node_conf_map = {}

        for node in conf_json['node_list']:
            name = node['name']
            assert name not in node_conf_map, Exception(
                'duplicate node: ' + name)
            conf = node_conf_map[name] = Conf(name, conf_json['sync'], node['epoch'])

            if name in link_json:
                link_list = link_json[name]
                for link in link_list:
                    dest = link['dest']
                    assert dest in all_node, Exception('no such node called ' + dest)
                    assert dest not in conf.connect, Exception(
                        'duplicate link from ' + name + ' to ' + dest)
                    conf.connect[dest] = all_node[dest].ip + ':' + str(all_node[dest].port)

        for name in node_conf_map:
            conf_path = os.path.join(self.config.controller_base_path, 'dml_file', 'conf', name + '_structure.conf')
            with open(conf_path, 'w') as f:
                f.writelines(json.dumps(node_conf_map[name].to_json(), indent=2))
