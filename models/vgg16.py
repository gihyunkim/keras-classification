import keras
from layer import *

class VGGNET16:
    def __init__(self,input_shape, class_num, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.l2_reg = None

    def vggnet_stem(self, inputs):
        '''32 x 32'''
        block1 = conv_bn(inputs, filter_size=64, kernel_size=(3, 3), strides=1,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        block1 = conv_bn(block1, filter_size=64, kernel_size=(3, 3), strides=1,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        # block1_out = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1)

        '''16 x 16'''
        block2 = conv_bn(block1, filter_size=128, kernel_size=(3, 3), strides=1,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        block2 = conv_bn(block2, filter_size=128, kernel_size=(3, 3), strides=1,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        # block2_out = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block2)
        return block2

    def vggnet_body(self, inputs):
        '''8 x 8'''
        block_3 = vgg_block(inputs, filter_size=256, kernel_size=(3,3), strides=1,
                            padding="same", activation="relu", regularizer=self.l2_reg)

        '''4 x 4'''
        block_4 = vgg_block(block_3, filter_size=512,  kernel_size=(3,3), strides=1,
                            padding="same", activation="relu", regularizer=self.l2_reg)

        '''2 x 2'''
        block_5 = vgg_block(block_4, filter_size=512, kernel_size=(3,3), strides=1,
                            padding="same", activation="relu", regularizer=self.l2_reg)

        return block_5

    def vggnet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.vggnet_stem(inputs)

        '''body'''
        body_out = self.vggnet_body(stem)

        '''head'''
        flatten = keras.layers.Flatten()(body_out)
        fc1 = keras.layers.Dense(units=4096)(flatten)
        fc2 = keras.layers.Dense(units=4096)(fc1)
        fc3 = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(fc2)
        return keras.models.Model(inputs, fc3)