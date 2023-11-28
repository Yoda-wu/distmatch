import tensorflow as tf
import numpy as np
from dataset.augment import Augment

class Dataset_Preprocessor:
    def __init__(self,
                 dataset_name:str,
                 weak_augment_helper:Augment,
                 strong_augment_helper:Augment) -> None:
        self.weak_augment_helper = weak_augment_helper
        self.strong_augment_helper = strong_augment_helper
        if dataset_name == 'cifar10':
            # CIFAR-10统计特征
            self.mean = np.array((0.4914, 0.4822, 0.4465), np.float32).reshape(1, 1, -1)
            self.std = np.array((0.2471, 0.2435, 0.2616), np.float32).reshape(1, 1, -1)
        elif dataset_name == 'mnist':
            self.mean = 0.1301
            self.std = 0.3081
        elif dataset_name == 'fmnist':
            self.mean = 0.0
            self.std = 1.0
        elif dataset_name == 'stl10':
            self.mean = np.array((0.5, 0.5, 0.5), np.float32).reshape(1, 1, -1)
            self.std = np.array((1.0, 1.0, 1.0), np.float32).reshape(1, 1, -1)

    def preprocess_labeled_dataset(self, x, y, batch_size, augment=False):
        # [-1~1]
        ax = x
        if augment:
            ax = self.weak_augment_helper.augment(x)
        labeled_train_db = tf.data.Dataset.from_tensor_slices((ax, y))
        labeled_train_db = labeled_train_db.shuffle(1000).map(
            lambda sx,y:(
                (tf.cast(sx, dtype=tf.float32) / 255. - self.mean) / self.std,
                tf.cast(y, dtype=tf.int32))
        ).batch(batch_size)
        return labeled_train_db

    def preprocess_unlabeled_dataset(self, ux, uy, batch_size):
        weak_augmented_ux = self.weak_augment_helper.augment(ux)
        strong_augmented_ux = self.strong_augment_helper.augment(ux)
        unlabeled_train_db = tf.data.Dataset.from_tensor_slices((ux, weak_augmented_ux, strong_augmented_ux, uy))
        unlabeled_train_db = unlabeled_train_db.shuffle(1000).map(
            lambda ux,wux,sux,uy: (
                (tf.cast(ux, dtype=tf.float32) / 255. - self.mean) / self.std,
                (tf.cast(wux, dtype=tf.float32) / 255. - self.mean) / self.std,
                (tf.cast(sux, dtype=tf.float32) / 255. - self.mean) / self.std,
                tf.cast(uy, dtype=tf.int32)
            )
        ).batch(batch_size)
        return unlabeled_train_db