import keras
import os
import datetime
from utils import get_cifar100
from models.vgg16 import VGGNET16
from models.resnet import Resnet
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint


def train():
    '''model configuration'''
    input_shape = (32, 32, 3)
    class_num = 100
    epochs = 1000
    batch_size = 64
    lr = 1e-4
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = keras.metrics.CategoricalAccuracy()
    model_name = "resnet"

    '''call back'''
    log_dir = "./logs/%s/%s"%(model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    weight_save_dir = "./save_weights/"
    if not os.path.isdir(weight_save_dir):
        os.mkdir(weight_save_dir)
    weight_save_file = "%s/%s_{epoch:05d}.h5"%(weight_save_dir, model_name)

    '''get datasets'''
    x_train, y_train, x_test, y_test = get_cifar100()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = keras.utils.to_categorical(y_train, class_num)
    y_test = keras.utils.to_categorical(y_test, class_num)

    '''train'''
    rs = Resnet(input_shape=input_shape, class_num=class_num, layer_num=34)
    model = rs.resnet()
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(log_dir),
                         ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
                         ModelCheckpoint(weight_save_file, monitor="val_loss", save_best_only=True)])

if __name__ == "__main__":
    train()