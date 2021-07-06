import keras
from keras.regularizers import l2

def conv_bn(inputs,filter_size, kernel_size, strides=1, padding="same", activation="relu", regularizer=l2(0.001)):
    layer1 = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding=padding
                                 ,kernel_regularizer=regularizer)(inputs)
    layer2 = keras.layers.BatchNormalization()(layer1)
    out = keras.layers.Activation(activation=activation)(layer2)
    return out

'''Pre-activation'''
def bn_conv(inputs,filter_size, kernel_size, strides=1, padding="same", activation="relu", regularizer=l2(0.001)):
    layer1 = keras.layers.BatchNormalization()(inputs)
    layer2 = keras.layers.Activation(activation=activation)(layer1)
    out = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding=padding
                                 ,kernel_regularizer=regularizer)(layer2)
    return out


def vgg_block(inputs, filter_size, kernel_size,  strides=1, padding="same", activation="relu", regularizer=l2(0.001)):
    block = conv_bn(inputs, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                     padding=padding, activation=activation, regularizer=regularizer)
    block = conv_bn(block, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                     padding=padding, activation=activation, regularizer=regularizer)
    block = conv_bn(block, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                     padding=padding, activation=activation, regularizer=regularizer)
    block_out = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block)
    return block_out

'''Resblock Start'''
def residual_block(inputs, filter_size, kernel_size,  strides=1, padding="same", activation="relu", regularizer=l2(0.001)):
    residual = inputs
    block = bn_conv(inputs, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                    padding=padding, activation=activation, regularizer=regularizer)
    block = bn_conv(block, filter_size=filter_size, kernel_size=kernel_size, strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)
    if strides == 2:
        residual = keras.layers.Conv2D(filters=filter_size, kernel_size=(1,1), strides=strides,
                                    padding=padding, activation=activation, kernel_regularizer=regularizer)(residual)
        residual = keras.layers.BatchNormalization()(residual)
    block_out = keras.layers.Add()([residual, block])
    return block_out

def residual_bottleneck_block(inputs, filter_size, kernel_size,  strides=1, padding="same", activation="relu", regularizer=l2(0.001)):
    residual = inputs
    block = bn_conv(inputs, filter_size=filter_size, kernel_size=(1, 1), strides=strides,
                    padding=padding, activation=activation, regularizer=regularizer)
    block = bn_conv(block, filter_size=filter_size, kernel_size=kernel_size, strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)
    block = bn_conv(block, filter_size=filter_size*4, kernel_size=(1, 1), strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)
    if strides == 2:
        residual = keras.layers.Conv2D(filters=filter_size*4, kernel_size=(1,1), strides=strides,
                                    padding=padding, activation=activation, kernel_regularizer=regularizer)(residual)
        residual = keras.layers.BatchNormalization()(residual)
    block_out = keras.layers.Add()([residual, block])
    return block_out

'''Resblock End'''

'''DenseBlock Start'''
def dense_block(inputs, filter_size, kernel_size,  strides=1, padding="same", activation="relu", regularizer=l2(0.001)):
    dense = inputs
    block = bn_conv(inputs, filter_size=4*filter_size, kernel_size=(1,1), strides=strides,
                    padding=padding, activation=activation, regularizer=regularizer)
    block = bn_conv(block, filter_size=filter_size, kernel_size=kernel_size, strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)

    block_out = keras.layers.Concatenate()([dense, block])
    return block_out

def transition_layer(inputs, filter_size, strides=1, padding="same", activation="relu", regularizer=l2(0.001), theta=0.5):
    block = bn_conv(inputs, filter_size=round(theta*filter_size), kernel_size=(1,1), strides=strides,
                    padding=padding, activation=activation, regularizer=regularizer)
    block_out = keras.layers.AveragePooling2D(pool_size=2, strides=2)(block)
    return block_out
'''DenseBlock End'''

'''MobileNet Start'''
def depthwise_separable_layer(inputs, filter_size, kernel_size, strides=1, activation="relu", alpha=1.0, regularizer=l2(0.001)):
    filter_size = int(filter_size * alpha)
    depthwise_layer = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                                   padding="same", kernel_regularizer=regularizer)(inputs)
    depthwise_layer = keras.layers.BatchNormalization()(depthwise_layer)
    depthwise_layer = keras.layers.Activation(activation)(depthwise_layer)

    pointwise_layer = keras.layers.Conv2D(filters=filter_size, kernel_size=(1, 1), strides=1,
                                          padding="same", kernel_regularizer=regularizer)(depthwise_layer)
    pointwise_layer = keras.layers.BatchNormalization()(pointwise_layer)
    pointwise_layer = keras.layers.Activation(activation)(pointwise_layer)
    return pointwise_layer
