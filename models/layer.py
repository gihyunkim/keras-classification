import keras
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

'''Activation Start'''
def hard_sigmoid(x):
    hs_x = 0.2 * x + 0.5
    return K.clip(hs_x, 0, 1)

def hard_swish(x):
    hs = keras.layers.ReLU(max_value=6)(x+3) * (1/6)
    hs = keras.layers.Multiply()([x, hs])
    return hs
'''Activation End'''

def conv_bn(inputs,filter_size, kernel_size, strides=1, padding="same", activation="relu", use_bias=False, regularizer=l2(0.0001)):
    layer1 = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding=padding
                                 ,kernel_regularizer=regularizer, use_bias=use_bias)(inputs)
    layer2 = keras.layers.BatchNormalization()(layer1)
    out = keras.layers.Activation(activation=activation)(layer2)
    return out

'''Pre-activation'''
def bn_conv(inputs,filter_size, kernel_size, strides=1, padding="same", activation="relu", use_bias=False, regularizer=l2(0.0001)):
    layer1 = keras.layers.BatchNormalization()(inputs)
    layer2 = keras.layers.Activation(activation=activation)(layer1)
    out = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding=padding ,
                              kernel_regularizer=regularizer, use_bias=use_bias)(layer2)
    return out


def vgg_block(inputs, filter_size, kernel_size,  strides=1, padding="same", activation="relu", regularizer=l2(0.0001)):
    block = conv_bn(inputs, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                     padding=padding, activation=activation, regularizer=regularizer)
    block = conv_bn(block, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                     padding=padding, activation=activation, regularizer=regularizer)
    block = conv_bn(block, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                     padding=padding, activation=activation, regularizer=regularizer)
    block_out = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block)
    return block_out

'''Resblock Start'''
def residual_block(inputs, filter_size, kernel_size,  strides=1, padding="same", activation="relu",
                   regularizer=l2(0.0001), first_layer=False):
    preact = keras.layers.BatchNormalization()(inputs)
    preact = keras.layers.Activation(activation)(preact)

    if strides == 2:
        residual = keras.layers.MaxPooling2D(pool_size=1, strides=2)(inputs)
    else:
        residual = inputs

    block = keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size, strides=strides, padding=padding
                                 ,kernel_regularizer=regularizer)(preact)
    block = bn_conv(block, filter_size=filter_size, kernel_size=kernel_size, strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)

    block_out = keras.layers.Add()([residual, block])
    return block_out

def residual_bottleneck_block(inputs, filter_size, kernel_size, strides=1, padding="same", activation="relu",
                              regularizer=l2(0.0001), first_layer=False):
    preact = keras.layers.BatchNormalization()(inputs)
    preact = keras.layers.Activation(activation)(preact)

    if first_layer:
        residual = keras.layers.Conv2D(filters=filter_size*4, kernel_size=(1,1), strides=1,
                                    padding=padding, kernel_regularizer=regularizer)(preact)
    elif strides == 2:
        residual = keras.layers.MaxPooling2D(pool_size=1, strides=2)(inputs)
    else:
        residual = inputs

    block = keras.layers.Conv2D(filters=filter_size, kernel_size=(1, 1), strides=1, padding=padding
                                ,kernel_regularizer=regularizer)(preact)
    block = bn_conv(block, filter_size=filter_size, kernel_size=kernel_size, strides=strides,
                    padding=padding, activation=activation, regularizer=regularizer)
    block = bn_conv(block, filter_size=filter_size*4, kernel_size=(1, 1), strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)

    block_out = keras.layers.Add()([residual, block])
    return block_out

'''Resblock End'''

'''DenseBlock Start'''
def dense_block(inputs, filter_size, kernel_size, padding="same", activation="relu", regularizer=l2(0.0001)):
    dense = inputs
    block = bn_conv(inputs, filter_size=4*filter_size, kernel_size=(1,1), strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)
    block = bn_conv(block, filter_size=filter_size, kernel_size=kernel_size, strides=1,
                    padding=padding, activation=activation, regularizer=regularizer)

    block_out = keras.layers.Concatenate()([dense, block])
    return block_out

def transition_layer(inputs, filter_size, strides=1, padding="same", activation="relu", regularizer=l2(0.0001), reduction=0.5):
    block = bn_conv(inputs, filter_size=round(reduction*filter_size), kernel_size=(1,1), strides=strides,
                    padding=padding, activation=activation, regularizer=regularizer)
    block_out = keras.layers.AveragePooling2D(pool_size=2, strides=2)(block)
    return block_out
'''DenseBlock End'''

'''MobileNet Start'''
def depthwise_separable_layer(inputs, filter_size, kernel_size, strides=1, activation="relu", alpha=1.0, regularizer=l2(0.0001)):
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

def inverted_res_block(inputs, filter_size, kernel_size, strides=1, activation="relu", alpha=1.0, exp=6, regularizer=l2(0.0001)):
    exp_filter_size = int(filter_size * exp)

    '''expand'''
    expand_layer = conv_bn(inputs, filter_size=exp_filter_size, kernel_size=(1,1), strides=1,
                    padding="same", activation=activation, regularizer=regularizer)

    '''depth-wise'''
    filter_size = int(filter_size * alpha)
    depthwise_layer = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                                   padding="same", kernel_regularizer=regularizer)(expand_layer)
    depthwise_layer = keras.layers.BatchNormalization()(depthwise_layer)
    depthwise_layer = keras.layers.Activation(activation)(depthwise_layer)

    '''project'''
    pointwise_layer = keras.layers.Conv2D(filters=filter_size, kernel_size=(1, 1), strides=1,
                                          padding="same", kernel_regularizer=regularizer)(depthwise_layer)
    dw_layer = keras.layers.BatchNormalization()(pointwise_layer)

    if strides==1:
        out = keras.layers.Add()([inputs, dw_layer])
    else:
        out = dw_layer
    return out

def inverted_res_block_with_se(inputs, filter_size, kernel_size, strides=1, activation="relu", alpha=1.0, exp=6, se_ratio=None, regularizer=l2(0.0001)):
    exp_filter_size = int(filter_size * exp)

    '''expand'''
    expand_layer = conv_bn(inputs, filter_size=exp_filter_size, kernel_size=(1,1), strides=1,
                    padding="same", activation=activation, regularizer=regularizer)

    '''depth-wise'''
    depthwise_layer = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                                   padding="same", kernel_regularizer=regularizer)(expand_layer)
    depthwise_layer = keras.layers.BatchNormalization()(depthwise_layer)
    depthwise_layer = keras.layers.Activation(activation)(depthwise_layer)

    '''se-block'''
    if se_ratio:
        depthwise_layer = se_block(depthwise_layer, filters=exp_filter_size, se_ratio=se_ratio)

    '''project'''
    filter_size = int(filter_size * alpha)
    pointwise_layer = keras.layers.Conv2D(filters=filter_size, kernel_size=(1, 1), strides=1,
                                          padding="same", kernel_regularizer=regularizer)(depthwise_layer)
    out_layer = keras.layers.BatchNormalization()(pointwise_layer)

    if strides==1:
        out = keras.layers.Add()([inputs, out_layer])
    else:
        out = out_layer
    return out

def se_block(inputs, filters, se_ratio):
    gap = keras.layers.GlobalAveragePooling2D(inputs)
    gap = keras.layers.Reshape((1, 1, filters))(gap)
    se_layer = keras.layers.Conv2D(int(filters*se_ratio), kernel_size=(1,1), padding="same")(gap)
    se_layer = keras.layers.Activation("relu")(se_layer)
    se_layer = keras.layers.Conv2D(filters, kernel_size=(1,1), padding="same")(se_layer)
    se_layer = keras.layers.Activation(hard_sigmoid)(se_layer)
    return keras.layers.Multiply()([inputs, se_layer])
'''MobileNet End'''

'''ShuffleNet Start'''
def group_conv(inputs, filters, kernel_size, strides=1, padding="same", regularizer=l2(0.0001), groups=1):
    if groups == 1:
        out = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                    padding=padding, kernel_regularizer=regularizer)(inputs)
    else:
        input_channel = K.int_shape(inputs)[-1]
        group_channel = round(input_channel / groups)
        group_outputs = []
        for idx in range(groups):
            group_layer = keras.layers.Lambda(lambda x : x[:, :, :, group_channel*idx:group_channel*(idx+1)])(inputs)
            group_outputs.append(keras.layers.Conv2D(filters=round(filters/groups), kernel_size=kernel_size, strides=strides,
                                         padding=padding, kernel_regularizer=regularizer))(group_layer)
        out = keras.layers.Concatenate()(group_outputs)

        '''groups must be able to evenly devide filters(Low Generalization)'''
        # group_layers = list(keras.layers.Lambda(lambda x : tf.split(x, groups, axis=-1))(inputs))
        # group_outoputs = []
        # for group_layer in group_layers:
        #     group_outoputs.append(conv_bn(group_layer, filter_size=round(filters/groups), kernel_size=kernel_size, strides=strides,
        #                 padding=padding, activation=activation, regularizer=regularizer))
        # out = keras.layers.Concatenate()(group_outoputs)
    return out

def shuffle_block(inputs, filters, kernel_size, strides=1, padding="same", activation="relu",
                  regularizer=l2(0.00001), groups=1):
    residual = inputs

    '''Group Conv 1'''
    group_layer = group_conv(inputs, filters=filters, kernel_size=(1, 1), strides=1, padding="same", regularizer=regularizer)
    group_layer = keras.layers.BatchNormalization()(group_layer)
    group_layer = keras.layers.Activation(activation)(group_layer)

    '''Channel Shuffle'''
    in_shape = K.int_shape(group_layer)
    shuffle_layer = keras.layers.Reshape((in_shape[0], in_shape[1], in_shape[2], groups, -1))(group_layer)
    shuffle_layer = keras.layers.Permute((0,1,2,4,3))(shuffle_layer)
    shuffle_layer = keras.layers.Reshape((in_shape[0], in_shape[1], in_shape[2], -1))(shuffle_layer)

    '''Depthwise & Group Conv'''
    shuffle_out = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=regularizer)(shuffle_layer)
    shuffle_out = keras.layers.BatchNormalization()(shuffle_out)
    shuffle_out = group_conv(shuffle_out, filters=filters, kernel_size=(1, 1), strides=1, padding="same", regularizer=regularizer)
    shuffle_out = keras.layers.BatchNormalization()(shuffle_out)

    if strides == 2:
        residual = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding="same")(residual)
        out = keras.layers.Concatenate()([residual, shuffle_out])
    else:
        out = keras.layers.Add()([residual, shuffle_out])
    out = keras.layers.Activation(activation)(out)
    return out

'''ShuffleNet End'''