from models.layer import *
from keras.applications import MobileNet

class Mobilenet:
    def __init__(self,input_shape, class_num, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)

    def mobilenet_stem(self, inputs):
        '''64 x 64'''
        block1 = conv_bn(inputs, filter_size=32, kernel_size=(3, 3), strides=2,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        return block1

    def mobilenet_body(self, inputs):
        '''32 x 32'''
        layer1 = depthwise_separable_layer(inputs, filter_size=32, kernel_size=(3, 3), strides=1,
                                           activation="relu", alpha=2.0, regularizer=self.l2_reg)

        layer2 = depthwise_separable_layer(layer1, filter_size=64, kernel_size=(3, 3), strides=2,
                                           activation="relu", alpha=2.0, regularizer=self.l2_reg)

        '''16 x 16'''
        layer3 = depthwise_separable_layer(layer2, filter_size=128, kernel_size=(3, 3), strides=1,
                                           activation="relu", alpha=1.0, regularizer=self.l2_reg)

        layer4 = depthwise_separable_layer(layer3, filter_size=128, kernel_size=(3, 3), strides=2,
                                           activation="relu", alpha=2.0, regularizer=self.l2_reg)

        '''8 x 8'''
        layer5 = depthwise_separable_layer(layer4, filter_size=256, kernel_size=(3, 3), strides=1,
                                           activation="relu", alpha=1.0, regularizer=self.l2_reg)

        layer6 = depthwise_separable_layer(layer5, filter_size=256, kernel_size=(3, 3), strides=2,
                                           activation="relu", alpha=2.0, regularizer=self.l2_reg)
        '''4 x 4'''
        middle = layer6
        for _ in range(5):
            middle = depthwise_separable_layer(middle, filter_size=512, kernel_size=(3, 3), strides=1,
                                           activation="relu", alpha=1.0, regularizer=self.l2_reg)

        '''strides=2 in original'''
        layer7 = depthwise_separable_layer(middle, filter_size=512, kernel_size=(3, 3), strides=2,
                                           activation="relu", alpha=2.0, regularizer=self.l2_reg)

        '''strides=2 in original'''
        layer_out = depthwise_separable_layer(layer7, filter_size=1024, kernel_size=(3, 3), strides=2,
                                           activation="relu", alpha=1.0, regularizer=self.l2_reg)
        return layer_out

    def mobilenet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.resnet_stem(inputs)

        '''body'''
        body_out = self.resnet_body(stem)

        '''head'''
        g_avg = keras.layers.GlobalAveragePooling2D()(body_out)
        fc_out = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(g_avg)
        return keras.models.Model(inputs, fc_out)