import tensorflow as tf
import keras
from keras import layers, Sequential


class LeNet(keras.Model):

    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3))
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')
        self.pool2 = layers.AveragePooling2D(2)
        # 定义全连接层
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation='relu')
        self.fc2 = layers.Dense(84, activation='relu')
        self.fc3 = layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # 定义前向传播
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def lenet5(num_classes):
    return LeNet(num_classes)
