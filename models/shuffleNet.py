from models.layer import *

class ShuffleNet:
    def __init__(self,input_shape, class_num, group=1, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.repeat = [3, 7, 3] # repeat num on each stage
        self.group = group
        filter_dict = {1:14, 2:200, 3:240, 4:272, 8:384}
        self.filters = filter_dict[group]

        if not group in self.channel_dict.keys():
            print("number of group required to be one of (1, 2, 3, 4, 8)")
            exit(-1)

    def shufflenet_stem(self, inputs):
        '''64 x 64'''
        block = conv_bn(inputs, filter_size=24, kernel_size=(3, 3), strides=2,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        block_out = keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding="same")(block)
        return block_out

    def shufflenet_body(self, inputs):
        '''Stage 2'''
        layer1 = shuffle_block(inputs, filters=self.filters, kernel_size=(3,3), strides=2, padding="same",
                              activation="relu", regularizer=self.l2_reg, groups=self.group)
        for _ in range(self.repeat[0]):
            layer1 = shuffle_block(layer1, filters=self.filters, kernel_size=(3, 3), strides=1, padding="same",
                                  activation="relu", regularizer=self.l2_reg, groups=self.group)

        '''Stage 3'''
        layer2 = shuffle_block(layer1, filters=self.filters * 2, kernel_size=(3, 3), strides=2, padding="same",
                               activation="relu", regularizer=self.l2_reg, groups=self.group)
        for _ in range(self.repeat[1]):
            layer2 = shuffle_block(layer2, filters=self.filters * 2, kernel_size=(3, 3), strides=1, padding="same",
                                   activation="relu", regularizer=self.l2_reg, groups=self.group)

        '''Stage 4'''
        layer3 = shuffle_block(layer2, filters=self.filters * 4, kernel_size=(3, 3), strides=2, padding="same",
                               activation="relu", regularizer=self.l2_reg, groups=self.group)
        for _ in range(self.repeat[2]):
            layer3 = shuffle_block(layer3, filters=self.filters * 4, kernel_size=(3, 3), strides=1, padding="same",
                                   activation="relu", regularizer=self.l2_reg, groups=self.group)
        return layer3

    def shufflenet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.shufflenet_stem()

        '''body'''
        body_out = self.shufflenet_body(stem)

        '''head'''
        g_avg = keras.layers.GlobalAveragePooling2D()(body_out)
        fc_out = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(g_avg)
        return keras.models.Model(inputs, fc_out)