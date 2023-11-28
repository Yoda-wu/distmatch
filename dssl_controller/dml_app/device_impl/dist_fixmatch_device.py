import logging
import tensorflow as tf
from config import Configuration
from device_impl.base_device import Base_Device
from keras import losses


class Distributed_FixMatch_Device(Base_Device):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        # 伪标签接受概率
        self.accept_probability = config.accept_probability
        
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
            self.labeled_batch_size,
            augment=True)
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
                    unsupervised_loss = self.__unsupervised_loss_fn(wux, sux)
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

    def __unsupervised_loss_fn(self, wux, sux):
        '''
        无监督损失
        使用无增强样本进行标签预测，使用强增强样本进行训练
        '''
        # [b, 32, 32, 3] => [b, 10]
        prob_on_wux = tf.nn.softmax(self.model(wux, training=True))
        pse_uy = tf.one_hot(tf.argmax(prob_on_wux, axis=1), depth=self.num_of_classes).numpy()
        pseudo_mask = tf.cast((tf.reduce_max(prob_on_wux, axis=1) >= self.accept_probability), tf.float32)

        prob_on_sux = tf.nn.softmax(self.model(sux, training=True))

        # 预测伪标签并计算无监督损失，排除掉不确定性过大的伪标签
        ce_with_pse_uy = losses.categorical_crossentropy(
            pse_uy,
            prob_on_sux
        )
        unsupervised_loss = tf.reduce_mean(ce_with_pse_uy * pseudo_mask)
        return unsupervised_loss