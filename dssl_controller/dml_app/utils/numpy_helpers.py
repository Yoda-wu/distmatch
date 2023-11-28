import os
import numpy as np

def np_save(base_dir, filename, data):
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)


def np_load(file_path):
    return np.load(file_path, allow_pickle=True).item()