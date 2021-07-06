from models.layer import *

class Densenet:
    def __init__(self,input_shape, class_num, layer_num = 121, theta = 0.5, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.filter_size = 32 # filter sizes
        self.theta = theta
        possible_densenet = [121, 169, 201, 264] # supported resnet layer

        # layer list for each resnet size
        layer_size_list = [[6, 12, 24, 16], [6, 12, 32, 32], [6, 12, 48, 32],
                           [6, 12, 64, 48]]

        densenet_num = possible_densenet.index(layer_num)
        self.layer_sizes = layer_size_list[densenet_num]

        if not layer_num in possible_densenet:
            print("Not Supported Densenet Size")
            print("Supported Size: 121, 169, 201, 264")
            exit(-1)

    def densenet_stem(self, inputs):
        '''64 x 64'''
        block1 = conv_bn(inputs, filter_size=self.filter_size, kernel_size=(7, 7), strides=2,
                         padding="same", activation="relu", regularizer=self.l2_reg)

        '''32 x 32'''
        block1_out = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(block1)
        return block1_out

    def densenet_body(self, x):
        for s in range(len(self.layer_sizes)):
            for l in range(self.layer_sizes[s]):
                x = dense_block(x, filter_size=self.filter_size, kernel_size=(3,3), strides=1, padding="same",
                                        activation="relu", regularizer=self.l2_reg)
            if s!=len(self.layer_sizes)-1: # Except Last Layer
                x = transition_layer(x, filter_size=self.filter_size, strides=1, padding="same", activation="relu",
                                 regularizer=self.l2_reg, theta=self.theta)
        return x

    def densenet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.densenet_stem(inputs)

        '''body'''
        body_out = self.densenet_body(stem)

        '''head'''
        g_avg = keras.layers.GlobalAveragePooling2D()(body_out)
        fc_out = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(g_avg)
        return keras.models.Model(inputs, fc_out)