from models.layer import *

class Densenet:
    def __init__(self,input_shape, class_num, layer_num = 121, reduction = 0.5, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.filter_size = 32 # filter sizes
        self.reduction = reduction
        possible_densenet = [121, 169, 201, 264] # supported densenet layer

        # layer list for each densenet size
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
        block = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same"
                                     , kernel_regularizer=self.l2_reg)(inputs)

        '''32 x 32'''
        block_out = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(block)
        return block_out

    def densenet_body(self, x):
        block_size = len(self.layer_sizes)
        for s in range(block_size):
            for l in range(self.layer_sizes[s]):
                x = dense_block(x, filter_size=self.filter_size, kernel_size=(3,3), strides=1, padding="same",
                                        activation="relu", regularizer=self.l2_reg)
            if s!=block_size-1: # Except Last Block
                x = transition_layer(x, filter_size=self.filter_size, strides=1, padding="same", activation="relu",
                                 regularizer=self.l2_reg, reduction=self.reduction)
        return x

    def densenet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.densenet_stem(inputs)

        '''body'''
        body_out = self.densenet_body(stem)

        '''head'''
        head = keras.layers.BatchNormalization()(body_out)
        head = keras.layers.Activation("relu")(head)
        g_avg = keras.layers.GlobalAveragePooling2D()(head)
        fc_out = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(g_avg)
        return keras.models.Model(inputs, fc_out)