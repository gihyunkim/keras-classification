from models.layer import *
import keras.backend as K

class MobilenetV3:
    def __init__(self,input_shape, class_num, alpha=1.0, mode="small", option="light", weight_decay=0.0001):
        self.input_shape = input_shape
        self.class_num = class_num
        self.l2_reg = keras.regularizers.l2(weight_decay)
        self.alpha = alpha

        if not mode in ["small", "large"]:
            print("mode should be 'small' or 'large'")
            exit(-1)
        if not option in ["light", "heavy"]:
            print("option should be 'light' or 'heavy'")
            exit(-1)

        self.mode = mode

        if option == "light":
            self.kernel_size = 3
            self.activation = "relu"
            self.se_ratio = None
        else:
            self.kernel_size = 5
            self.activation = hard_swish
            self.se_ratio = 0.25

    def mobilenetV3_stem(self, inputs):
        '''64 x 64'''
        layer = conv_bn(inputs, filter_size=16, kernel_size=(3, 3), strides=2,
                         padding="same", activation="relu", regularizer=self.l2_reg)
        return layer

    def mobilenetV3_Small_body(self, inputs):
        '''32 x 32'''
        layer = inverted_res_block_with_se(inputs, 16, (3, 3), strides=2, activation="relu", exp=1,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(inputs)

        layer = inverted_res_block_with_se(inputs, 24, (3, 3), strides=2, activation="relu", exp=72./16,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 24, (3, 3), strides=1, activation="relu", exp=88./24,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 40, self.kernel_size, strides=2, activation=self.activation, exp=4,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 40, self.kernel_size, strides=1, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 40, self.kernel_size, strides=1, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 48, self.kernel_size, strides=1, activation=self.activation, exp=3,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 48, self.kernel_size, strides=1, activation=self.activation, exp=3,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 96, self.kernel_size, strides=2, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 96, self.kernel_size, strides=1, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer_out = inverted_res_block_with_se(inputs, 96, self.kernel_size, strides=1, activation=self.activation, exp=6,
                                               se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)
        return layer_out

    def mobilenetV3_Large_body(self, inputs):
        '''32 x 32'''
        layer = inverted_res_block_with_se(inputs, 16, (3, 3), strides=1, activation="relu", exp=1,
                                           se_ratio=None, regularizer=self.l2_reg)(inputs)

        layer = inverted_res_block_with_se(inputs, 24, (3, 3), strides=2, activation="relu", exp=4,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 24, (3, 3), strides=1, activation="relu", exp=3,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 40, self.kernel_size, strides=2, activation="relu", exp=3,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 40, self.kernel_size, strides=1, activation="relu", exp=3,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 40, self.kernel_size, strides=1, activation="relu", exp=3,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 80, (3, 3), strides=2, activation=self.activation, exp=6,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 80, (3, 3), strides=1, activation=self.activation, exp=2.5,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 80, (3, 3), strides=2, activation=self.activation, exp=2.3,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 80, (3, 3), strides=1, activation=self.activation, exp=2.3,
                                           se_ratio=None, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 112, (3, 3), strides=1, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 112, (3, 3), strides=1, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 160, self.kernel_size, strides=2, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer = inverted_res_block_with_se(inputs, 160, self.kernel_size, strides=1, activation=self.activation, exp=6,
                                           se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        layer_out = inverted_res_block_with_se(inputs, 160, self.kernel_size, strides=1, activation=self.activation, exp=6,
                                               se_ratio=self.se_ratio, regularizer=self.l2_reg)(layer)

        return layer_out

    def mobilenetV3_head(self, inputs):
        last_filters = K.int_shape(inputs)[-1] * 6
        head = conv_bn(inputs, filter_size=last_filters, kernel_size=(1, 1), padding="same", activation=self.activation)
        head = keras.layers.GlobalAveragePooling2D()(head)
        head = keras.layers.Reshape((1, 1, -1))(head)

        head = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), padding="same", activation=self.activation)(head)
        head = keras.layers.Dropout(0.2)(head)
        head = keras.layers.Conv2D(filters=self.class_num, kernel_size=(1, 1), padding="same")(head)
        head = keras.layers.Flatten()(head)
        head_out = keras.layers.Activation("softmax")(head)
        return head_out

    def mobilenetV3(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        '''stem'''
        stem = self.mobilenetV3_stem(inputs)

        '''body'''
        if self.mode == "small":
            body_out = self.mobilenetV3_Small_body(stem)
        else:
            body_out = self.mobilenetV3_Large_body(stem)

        '''head'''
        head_out = self.mobilenetV3_head(body_out)
        return keras.models.Model(inputs, head_out)