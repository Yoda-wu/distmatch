import random


class Connection_Generator:
    def __init__(self,
                 num_of_random_paths: int,
                 num_of_devices: int) -> None:
        # 随机生成连通图路径数应大于设备数
        if num_of_random_paths < num_of_devices - 1:
            raise ValueError('num_of_random_paths < num_of_devices')
        if num_of_random_paths > (num_of_devices * (num_of_devices - 1)) / 2:
            raise ValueError('num_of_random_paths > (num_of_devices * (num_of_devices - 1)) / 2')
        self.num_of_devices = num_of_devices
        self.num_of_random_paths = num_of_random_paths

    def generate_star_topology(self):
        topology = [[] for _ in range(self.num_of_devices)]
        for i in range(self.num_of_devices):
            self.__add_path(topology, 0, i)
        return topology

    def generate_fully_connected_topology(self):
        topology = [[] for _ in range(self.num_of_devices)]
        for i in range(self.num_of_devices):
            for j in range(self.num_of_devices):
                self.__add_path(topology, i, j)
        return topology

    def generate_round_topology(self):
        topology = [[] for _ in range(self.num_of_devices)]
        for i in range(self.num_of_devices):
            self.__add_path(topology, i, (i + self.num_of_devices + 1) % self.num_of_devices)
        return topology

    def generate_random_tree_topology(self):
        topology = [[] for _ in range(self.num_of_devices)]
        connected_ids = [0]
        not_connected_ids = [i for i in range(1, self.num_of_devices)]
        for _ in range(self.num_of_devices - 1):
            selected_device_i = random.choice(connected_ids)
            selected_device_j = random.choice(not_connected_ids)
            connected_ids.append(selected_device_j)
            not_connected_ids.remove(selected_device_j)
            self.__add_path(topology, selected_device_i, selected_device_j)
        return topology

    def generate_random_connected_topology(self):
        topology = self.generate_random_tree_topology()
        i = 0
        # 树的路径数为self.num_of_devices - 1
        while i < (self.num_of_random_paths - self.num_of_devices + 1):
            select_device_i = random.randint(0, self.num_of_devices - 1)
            select_device_j = random.randint(0, self.num_of_devices - 1)
            if self.__add_path(topology, select_device_i, select_device_j):
                i += 1
        return topology

    def __add_path(self,
                   topology: list,
                   device_i: int,
                   device_j: int):
        if device_i != device_j and device_i not in topology[device_j]:
            topology[device_i].append(device_j)
            topology[device_j].append(device_i)
            return True
        else:
            return False
