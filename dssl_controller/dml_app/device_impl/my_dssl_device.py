import numpy as np
import tensorflow as tf
from config import Configuration
from dml_app.device_impl.base_device import Base_Device
from keras import losses


class My_DSSL_Device(Base_Device):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        # 协作网络文件列表
        self.model_on_helpers = []
        # 协作网络验证精度
        self.accuracy_of_helpers = []
        # 每个设备拥有的协作网络的个数
        self.max_num_of_helpers = config.num_of_helpers
        # 获取到的网络在本机数据集上的准确率
        # 表项结构为字典(设备ID: 设备精确度)
        self.accuracy_of_fetched_models = {}
        # 新聚合的模型权重
        self.helper_model_weight = None
        # 伪标签接受概率与拒绝概率
        self.accept_probability = config.accept_probability
        self.deny_probability = config.deny_probability
        # 预训练轮次
        self.warming_up_round = config.warming_up_rounds

    def initialize(self):
        super().initialize()
        self.validate_db = self.dataset_preprocessor.preprocess_labeled_dataset(self.labeled_x, self.labeled_y, 512, augment=True)

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
            for _, ((x, y), (ux, wux, sux, uy)) in enumerate(zip(labeled_train_db, unlabeled_train_db)):
                # 协助模型提供伪标签，并用于无监督训练
                pseudo_soft_label = self.__label_guess_on_helpers(ux)

                # 有监督训练，简单计算交叉熵
                with tf.GradientTape() as tape:
                    supervised_loss = self.__supervised_loss_fn(x, y)
                    # 有监督预训练轮次满足要求，开始同时进行无监督训练
                    if self.current_round > self.warming_up_round:
                        unsupervised_loss = self.__unsupervised_loss_fn(sux, pseudo_soft_label)
                    else:
                        unsupervised_loss = 0.0
                    loss = self.loss_lambda * supervised_loss + self.loss_eta * unsupervised_loss
  
                grads = tape.gradient(loss, self.model.trainable_variables)
                with self.lock:
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                total_supervised_loss += float(supervised_loss) * int(x.shape[0])
                total_num_of_labeled_data += int(x.shape[0])
                total_unsupervised_loss += float(unsupervised_loss) * int(ux.shape[0])
                total_num_of_unlabeled_data += int(ux.shape[0])

                del pseudo_soft_label, grads, unsupervised_loss, supervised_loss, loss
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
    
    def __unsupervised_loss_fn(self, au, pseudo_soft_label):
        '''
        无监督损失
        使用无增强样本进行标签预测，使用强增强样本进行训练
        '''
        # [b, 32, 32, 3] => [b, 10]
        probability_on_strong_augmented = tf.nn.softmax(self.model(au, training=True))

        # 预测伪标签并计算无监督损失，排除掉不确定性过大的伪标签
        ce_between_pseudo_and_strong_augmented = losses.categorical_crossentropy(
            pseudo_soft_label,
            probability_on_strong_augmented
        )
        pseudo_mask = tf.cast((tf.reduce_max(pseudo_soft_label, axis=1) >= self.deny_probability), tf.float32)
        unsupervised_loss = tf.reduce_mean(ce_between_pseudo_and_strong_augmented * pseudo_mask)

        return unsupervised_loss
    
    def __label_guess_on_helpers(self, ux):
        if self.current_round <= self.warming_up_round:
            return None
        
        # 使用协作节点的网络计算无标签样本的logit
        probability_and_acc_on_helpers = [
            {
                'accuracy': self.__helper_evaluate(self.model),
                'probability': tf.nn.softmax(self.model(ux, training=True))
            }
        ]
        for i in range(len(self.model_on_helpers)):
            self.assistant_model.set_weights(self.model_on_helpers[i])
            probability_and_acc_on_helpers.append(
                {
                    'accuracy': self.accuracy_of_helpers[i],
                    'probability': tf.nn.softmax(self.assistant_model(ux, training=True))
                }
            )
        
        pseudo_soft_label = []
        for probability_and_acc in probability_and_acc_on_helpers:
            # 获得独热标签
            probability = probability_and_acc['probability']
            acc = probability_and_acc['accuracy']
            pse_y = tf.one_hot(tf.argmax(probability, axis=1), depth=self.num_of_classes).numpy()
            # 获得独热标签置信度是否大于规定值
            pseudo_mask = tf.cast(
                (tf.reduce_max(probability, axis=1, keepdims=True) >= self.accept_probability),
                tf.float32
            ).numpy()
            pseudo_soft_label.append(pse_y * pseudo_mask * acc)
        pseudo_soft_label = tf.reduce_sum(np.array(pseudo_soft_label), axis=0).numpy()
        # 防止除以0
        pseudo_soft_label = pseudo_soft_label + 1e-5
        pseudo_soft_label = pseudo_soft_label / tf.reduce_sum(pseudo_soft_label, axis=1, keepdims=True).numpy()
        del probability_and_acc_on_helpers, probability, acc, pse_y, pseudo_mask
        return pseudo_soft_label
    
    def __helper_evaluate(self, model):
        '''
        有监督损失
        '''
        total_loss = 0.0
        total_num = 0
        for x,y in self.validate_db:
            # [b, 32, 32, 3] => [b, 10]
            logits = model(x, training=True)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=self.num_of_classes)
            # 计算交叉熵
            total_loss += tf.reduce_sum(losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            total_num += int(x.shape[0])
        avg_loss = total_loss / total_num
        return tf.exp(-avg_loss)

    
    def update_helpers(self):
        '''
        获取其他设备的网络，根据选择算法确定协作网络
        当前选择算法为使用有标签数据集进行评估，取最高精度的为协作网络
        '''
        if self.max_num_of_helpers == 0:
            return
        
        del self.model_on_helpers
        
        # 对每个邻居的网络在自己的有标签数据集上进行验证
        self.validate_db = self.dataset_preprocessor.preprocess_labeled_dataset(self.labeled_x, self.labeled_y, 512, augment=True)
        neighbor_acc_on_labeled_set = []
        for fetched_model in self.fetched_models:
            self.assistant_model.set_weights(fetched_model['weights'])
            acc = self.__helper_evaluate(self.assistant_model)
            self.accuracy_of_fetched_models[fetched_model['device'].get_node_name()] = acc
            neighbor_acc_on_labeled_set.append(acc)

        neighbor_acc_on_labeled_set = np.array(neighbor_acc_on_labeled_set)
        # 确定协作网络数量，记录正确率最高的数个网络的ID
        if self.max_num_of_helpers <= len(self.fetched_models):
            num_of_helpers = self.max_num_of_helpers
        else:
            num_of_helpers = len(self.fetched_models)
        helper_ids = neighbor_acc_on_labeled_set.argsort()[-num_of_helpers:]

        # 保存协作网络信息
        self.model_on_helpers = [self.fetched_models[id]['weights'] for id in helper_ids]
        self.accuracy_of_helpers = [neighbor_acc_on_labeled_set[id] for id in helper_ids]
        self.logger.log(
            'round: {} helpers: {} with accuracy {}'.format(
                self.current_round,
                [self.fetched_models[id]['device'].get_node_name() for id in helper_ids],
                [neighbor_acc_on_labeled_set[id] for id in helper_ids]
            )
        )
        del neighbor_acc_on_labeled_set