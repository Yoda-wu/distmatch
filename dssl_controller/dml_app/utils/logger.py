import logging
import os

from utils.network_utils import post_json, send_data

class Logger:
    def __init__(self, controller_addr: str, node_name: str, log_path: str) -> None:
        self.controller_addr = controller_addr
        self.node_name = node_name
        self.log_path = log_path
        self.log_file = os.path.join(self.log_path, self.node_name + '.log')
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(
            filename=self.log_file,
            format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
            level=logging.INFO
        )

    def log(self, message):
        print(message)
        logging.info(message)

    
    def send_log (self, filename: str):
        """
        send log file to controller.
        this request can be received by controller/ctl_utils.py, log_listener ().
        """
        with open (filename, 'r') as f:
            path = '/log?name=' + self.node_name
            send_data ('POST', path, self.controller_addr, files={'log': f})

    def routine_log(self, current_round, current_time, sup_loss, uns_loss, val_acc):
        path = '/routine-log'
        post_json(path, self.controller_addr, json={
            'node_name': self.node_name,
            'current_round': current_round,
            'current_time': int(current_time),
            'sup_loss': float(sup_loss),
            'uns_loss': float(uns_loss),
            'val_acc': float(val_acc)
        })