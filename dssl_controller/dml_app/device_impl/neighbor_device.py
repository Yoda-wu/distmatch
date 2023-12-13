import json

import numpy as np
from dml_app.utils.network_utils import send_data


class Neighbor_Device:
    def __init__(self, node_name, node_addr) -> None:
        self.node_name = node_name
        self.node_addr = node_addr
        self.model_weight = None
        self.current_round = 0
        self.dataset_size = 0

    def get_node_name(self):
        return self.node_name

    def get_node_addr(self):
        return self.node_addr

    def sync_with_origin(self):
        data = json.loads(send_data('GET', '/current-round', self.node_addr))
        current_round = data['round']
        if self.current_round != current_round:
            while True:
                if json.loads(send_data('GET', 'is_train_finish', self.node_addr))['is_train_finish']:
                    break
                print('wait for train finish')
            data = json.loads(send_data('GET', '/model-info', self.node_addr))
            self.model_weight = [np.array(layer) for layer in data['weight']]
            self.current_round = data['round']
            self.dataset_size = data['dataset_size']

    def get_model_weight(self):
        return self.model_weight

    def get_current_round(self):
        return self.current_round

    def get_dataset_size(self):
        return self.dataset_size
    
    def clear(self):
        self.model_weight = None