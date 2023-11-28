import logging
import tensorflow as tf
from keras import losses
from config import Configuration
from device_impl.base_device import Base_Device


class Distrituted_SGD_Device(Base_Device):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)

        
    def update_helpers(self):
        '''
        获取其他设备的网络，根据选择算法确定协作网络
        Gossip空过该函数
        '''
        pass

    def train_one_round(self):
        labeled_train_db = self.dataset_preprocessor.preprocess_labeled_dataset(
            self.labeled_x,
            self.labeled_y,
            self.labeled_batch_size)
        unlabeled_train_db = self.dataset_preprocessor.preprocess_unlabeled_dataset(
            self.unlabeled_x,
            self.unlabeled_y,
            self.unlabeled_batch_size)
        
        for epoch in range(self.config.num_of_epochs_per_round):
            total_supervised_loss = 0.0
            total_num_of_labeled_data = 0
            total_unsupervised_loss = 0.0
            total_num_of_unlabeled_data = 0
            logging.info(f'round {self.current_round} begin train')
            for _, ((x, y), (ux, wux, sux, uy)) in enumerate(zip(labeled_train_db, unlabeled_train_db)):
                # 有监督训练，简单计算交叉熵
                with tf.GradientTape() as tape:
                    supervised_loss = self.__supervised_loss_fn(x, y)
                    unsupervised_loss = 0.0
                    loss = self.loss_lambda * supervised_loss + self.loss_eta * unsupervised_loss
  
                grads = tape.gradient(loss, self.model.trainable_variables)
                with self.lock:
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_supervised_loss += float(supervised_loss) * int(x.shape[0])
                total_num_of_labeled_data += int(x.shape[0])
                total_unsupervised_loss += float(unsupervised_loss) * int(ux.shape[0])
                total_num_of_unlabeled_data += int(ux.shape[0])

                del grads, unsupervised_loss, supervised_loss, loss
        del x, y, ux, wux, sux, uy, labeled_train_db, unlabeled_train_db
        
        return (total_supervised_loss, total_num_of_labeled_data, total_unsupervised_loss, total_num_of_unlabeled_data)

    def __supervised_loss_fn(self, x, y):
        '''
        有监督损失
        '''
        # [b, 32, 32, 3] => [b, 10]
        logits = self.model(x, training=True)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=self.num_of_classes)
        # 计算交叉熵
        supervised_loss = losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
        supervised_loss = tf.reduce_mean(supervised_loss)
        return supervised_loss