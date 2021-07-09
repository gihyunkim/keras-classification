from models.layer import *

class MobilenetV2:
    def __init__(self,input_shape, class_num, alpha=1.0, exp=6, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.alpha = alpha
        self.exp = exp

    def mobilenetV2_stem(self, inputs):
        '''64 x 64'''
        layer = conv_bn(inputs, filter_size=32, kernel_size=(3, 3), strides=2,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        return layer

    def mobilenetV2_body(self, inputs):
        '''32 x 32'''
        layer1 = inverted_res_block(inputs, filter_size=16, strides=1, exp=1)

        '''16 x 16'''
        layer2 = inverted_res_block(layer1, filter_size=24, strides=2, exp=self.exp)
        layer2 = inverted_res_block(layer2, filter_size=24, strides=1, exp=self.exp)

        '''8 x 8'''
        layer3 = inverted_res_block(layer2, filter_size=32, strides=2, exp=self.exp)
        layer3 = inverted_res_block(layer3, filter_size=32, strides=1, exp=self.exp)
        layer3 = inverted_res_block(layer3, filter_size=32, strides=1, exp=self.exp)

        '''4 x 4'''
        layer4 = inverted_res_block(layer3, filter_size=64, strides=2, exp=self.exp)
        layer4 = inverted_res_block(layer4, filter_size=64, strides=1, exp=self.exp)
        layer4 = inverted_res_block(layer4, filter_size=64, strides=1, exp=self.exp)
        layer4 = inverted_res_block(layer4, filter_size=64, strides=1, exp=self.exp)

        layer5 = inverted_res_block(layer4, filter_size=96, strides=1, exp=self.exp)
        layer5 = inverted_res_block(layer5, filter_size=96, strides=1, exp=self.exp)
        layer5 = inverted_res_block(layer5, filter_size=96, strides=1, exp=self.exp)
        layer5 = inverted_res_block(layer5, filter_size=96, strides=1, exp=self.exp)

        '''2 x 2'''
        layer6 = inverted_res_block(layer5, filter_size=160, strides=2, exp=self.exp)
        layer6 = inverted_res_block(layer6, filter_size=160, strides=1, exp=self.exp)
        layer6 = inverted_res_block(layer6, filter_size=160, strides=1, exp=self.exp)

        layer_out = inverted_res_block(layer6, filter_size=320, strides=1, exp=self.exp)
        return layer_out

    def mobilenetV2(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.mobilenetV2_stem(inputs)

        '''body'''
        body_out = self.mobilenetV2_body(stem)

        '''head'''
        head = conv_bn(body_out, filter_size=1280, kernel_size=(3,3), regularizer=self.l2_reg)
        g_avg = keras.layers.GlobalAveragePooling2D()(head)
        fc_out = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(g_avg)
        return keras.models.Model(inputs, fc_out)