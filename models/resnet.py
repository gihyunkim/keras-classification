from models.layer import *

class Resnet:
    def __init__(self,input_shape, class_num, layer_num = 50, weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.filter_list = [64, 128, 256, 512] # filter sizes
        possible_resnet =[18, 34, 50, 101, 152] # supported resnet layer

        # layer list for each resnet size
        layer_size_list = [[2, 2, 2, 2], [3, 4, 6, 3], [3, 4, 6, 3],
                           [3, 4, 23, 3], [3, 8, 36, 3]]

        resnet_num = possible_resnet.index(layer_num)
        self.layer_sizes = layer_size_list[resnet_num]

        if not layer_num in possible_resnet:
            print("Not Supported Resnet Size")
            print("Supported Size: 18, 34, 50 ,101, 152")
            exit(-1)

        if layer_num < 50:
            self.block = residual_block
        else:
            self.block = residual_bottleneck_block

    def resnet_stem(self, inputs):
        '''32 x 32'''
        block1 = keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same"
                                     , kernel_regularizer=self.l2_reg)(inputs)

        '''16 x 16'''
        block1_out = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(block1)
        return block1_out

    def resnet_body(self, x):
        block_size = len(self.layer_sizes)
        for s in range(block_size):
            for l in range(self.layer_sizes[s]):
                if l==0:
                    x = self.block(x, filter_size=self.filter_list[s], kernel_size=(3,3), strides=1, padding="same",
                                        activation="relu", regularizer=self.l2_reg, first_layer=True)
                elif s!=block_size-1 and l==self.layer_sizes[s]-1:
                    x = self.block(x, filter_size=self.filter_list[s], kernel_size=(3,3), strides=2, padding="same",
                                        activation="relu", regularizer=self.l2_reg, first_layer=False)
                else:
                    x = self.block(x, filter_size=self.filter_list[s], kernel_size=(3,3), strides=1, padding="same",
                                        activation="relu", regularizer=self.l2_reg)
        return x

    def resnet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.resnet_stem(inputs)

        '''body'''
        body_out = self.resnet_body(stem)

        '''head'''
        head = keras.layers.BatchNormalization()(body_out)
        head = keras.layers.Activation("relu")(head)
        g_avg = keras.layers.GlobalAveragePooling2D()(head)
        fc_out = keras.layers.Dense(units=self.class_num, activation="softmax", use_bias=False)(g_avg)
        return keras.models.Model(inputs, fc_out)