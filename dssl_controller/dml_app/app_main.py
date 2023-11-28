from concurrent.futures import ThreadPoolExecutor
import json
from flask import Flask, request
from config import Configuration
from device_impl.base_device import Base_Device

config = Configuration()
device: Base_Device = None
app = Flask(__name__)
executor = ThreadPoolExecutor(1)


# DML实现接口
@app.route('/hi', methods=['GET'])
def on_route_hi():
    return device.on_heartbeat()


# DML实现接口
@app.route('/log', methods=['GET'])
def on_route_log():
    print('GET at /log')
    executor.submit(device.on_log)
    return ''


# DML实现接口
# 根据#(node_name)_dataset.conf配置运行信息
@app.route('/conf/dataset', methods=['POST'])
def on_conf_dataset():
    print('POST at /conf/dataset')
    f = request.files.get('conf').read()
    dataset_conf = json.loads(f)
    config.config_dataset(dataset_conf)

    global device
    device = config.device_class(config)
    return ''


# DML实现接口
@app.route('/conf/structure', methods=['POST'])
def on_conf_structure():
    print('POST at /conf/structure')
    f = request.files.get('conf').read()
    structure_conf = json.loads(f)
    config.config_structure(structure_conf)
    return ''


# DML实现接口
# 开始训练
@app.route('/start', methods=['GET'])
def on_start():
    print('GET at /start')
    executor.submit(device.on_start)
    return ''


# 获取模型信息
@app.route('/model-info', methods=['GET'])
def on_get_model_info():
    print('GET at /model-info')
    return device.on_get_model_info()


# 获取模型信息
@app.route('/current-round', methods=['GET'])
def on_get_current_round():
    print('GET at /current-round')
    return device.on_get_current_round()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.dml_port, threaded=True)
